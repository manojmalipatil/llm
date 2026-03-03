import asyncio
import os
import sqlite3
import uuid
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv

# LiveKit Imports
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import sarvam, groq, silero

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GrievanceBot")


# --- Load Language Configuration ---
def load_language_config(config_path="language_config.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"[CONFIG] Loaded language config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"[ERROR] Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] Invalid JSON in config file: {e}")
        raise

LANGUAGE_CONFIG = load_language_config()
SUPPORTED_LANGUAGES = LANGUAGE_CONFIG["supported_languages"]
GRIEVANCE_PROMPTS = LANGUAGE_CONFIG["grievance_prompts"]


# ---------------------------------------------------------------------------
# SQLite Database Manager
# ---------------------------------------------------------------------------
class DatabaseManager:
    def __init__(self, db_path="grievance.db"):
        self.db_path = db_path
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                transcript  TEXT NOT NULL,
                language    TEXT,
                created_at  TEXT NOT NULL,
                status      TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"[DB] SQLite ready at {self.db_path}")

    async def save_grievance(self, transcript: str, language: str = "en") -> str | None:
        if not transcript.strip():
            logger.warning("[DB] Empty transcript — skipping save")
            return None

        record_id    = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()

        def _write():
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO grievances (id, timestamp, transcript, language, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (record_id, current_time, transcript, language, current_time),
                )
                conn.commit()
                logger.info(f"[DB] Saved grievance ID: {record_id} (lang={language})")
                return record_id
            except Exception as e:
                logger.error(f"[DB] Error saving grievance: {e}")
                return None
            finally:
                conn.close()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _write)


# ---------------------------------------------------------------------------
# Conversation Tracker
# ---------------------------------------------------------------------------
class GrievanceTracker:
    def __init__(self):
        self.grievance_text: list[str] = []
        self.selected_language: str | None = None

    def set_language(self, language_code: str):
        self.selected_language = language_code
        logger.info(f"[TRACKER] Language → {language_code}")

    def add_user_message(self, text: str):
        self.grievance_text.append(f"Employee: {text}")

    def add_agent_message(self, text: str):
        self.grievance_text.append(f"Agent: {text}")

    def get_full_grievance(self) -> str:
        return "\n".join(self.grievance_text)


# ---------------------------------------------------------------------------
# Agent Entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    logger.info(f"[ROOM] Connecting to room: {ctx.room.name}")

    for key in ["GROQ_API_KEY", "SARVAM_API_KEY"]:
        if not os.getenv(key):
            raise ValueError(f"[ERROR] {key} is required but not set in environment")

    db_manager        = DatabaseManager()
    grievance_tracker = GrievanceTracker()
    should_end_call   = asyncio.Event()

    await ctx.connect()

    # -----------------------------------------------------------------------
    # Receive language from web UI via data channel
    # -----------------------------------------------------------------------
    selected_lang_code   = "en"
    selected_sarvam_code = "en-IN"
    language_received    = asyncio.Event()

    def on_data_received(data_packet: rtc.DataPacket):
        nonlocal selected_lang_code, selected_sarvam_code
        try:
            msg = json.loads(data_packet.data.decode("utf-8"))
            if msg.get("type") == "language_selected":
                lang_code   = msg.get("language_code", "en")
                sarvam_code = msg.get("sarvam_code", "en-IN")

                for _, lang_data in SUPPORTED_LANGUAGES.items():
                    if lang_data["code"] == lang_code:
                        selected_lang_code   = lang_code
                        selected_sarvam_code = lang_data.get("sarvam_tts_code", sarvam_code)
                        break
                else:
                    selected_lang_code   = lang_code
                    selected_sarvam_code = sarvam_code

                grievance_tracker.set_language(selected_lang_code)
                logger.info(f"[LANG] Set from UI → {selected_lang_code} / TTS: {selected_sarvam_code}")
                language_received.set()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"[DATA] Could not parse data message: {e}")

    ctx.room.on("data_received", on_data_received)

    logger.info("[LANG] Waiting for language selection (up to 8 s)...")
    try:
        await asyncio.wait_for(language_received.wait(), timeout=8.0)
        logger.info(f"[LANG] ✓ Received: {selected_lang_code}")
    except asyncio.TimeoutError:
        logger.warning("[LANG] ⚠ Timeout — defaulting to English")
        grievance_tracker.set_language("en")

    ctx.room.off("data_received", on_data_received)

    # -----------------------------------------------------------------------
    # Grievance collection session
    # -----------------------------------------------------------------------
    logger.info(f"[STAGE] Starting grievance collection — lang: {selected_lang_code}")

    prompt = GRIEVANCE_PROMPTS.get(
        selected_lang_code,
        GRIEVANCE_PROMPTS.get("en", "You are a helpful grievance assistant. Collect the employee's grievance in detail.")
    )

    vad = silero.VAD.load(
        min_speech_duration=0.3,
        min_silence_duration=0.5,
        activation_threshold=0.5,
        sample_rate=16000,
    )

    session = AgentSession(
        vad=vad,
        stt=groq.STT(model="whisper-large-v3-turbo", language=selected_lang_code),
        llm=groq.LLM(model="llama-3.3-70b-versatile", temperature=0.3),
        tts=sarvam.TTS(target_language_code=selected_sarvam_code, speaker="anushka"),
    )

    @function_tool
    async def end_call(confirmation: str = "yes"):
        """
        End the grievance collection call.
        Call this ONLY when the user explicitly says they are done,
        wants to hang up, or has no more grievances to share.
        """
        logger.info("[TOOL] end_call triggered")
        should_end_call.set()
        return ""

    grievance_agent = Agent(instructions=prompt, tools=[end_call])

    @session.on("conversation_item_added")
    def on_item(event):
        if event.item.text_content:
            role = event.item.role
            text = event.item.text_content
            logger.info(f"[CONV] {role}: {text}")
            if role == "user":
                grievance_tracker.add_user_message(text)
            elif role == "assistant":
                grievance_tracker.add_agent_message(text)

    session_task = asyncio.create_task(
        session.start(agent=grievance_agent, room=ctx.room)
    )

    await asyncio.sleep(0.5)

    try:
        session.allow_interruptions = False 
        logger.info("[STAGE] Generating initial greeting (Interruptions Disabled)")
        await session.generate_reply()
        while session.is_speaking: 
            await asyncio.sleep(0.1)
        logger.info("[STAGE] ✓ Initial greeting sent and completed")
    except Exception as e:
        logger.error(f"[STAGE] Error generating greeting: {e}")
    finally:
        session.allow_interruptions = True
        logger.info("[STAGE] Interruptions re-enabled for user input")

    logger.info("[STAGE] Collecting grievance — waiting for end_call...")
    await should_end_call.wait()

    logger.info("[STAGE] Call ending — waiting for farewell audio...")
    await asyncio.sleep(12.0)

    # -----------------------------------------------------------------------
    # Cleanup & save
    # -----------------------------------------------------------------------
    logger.info("[CLEANUP] Closing session...")
    try:
        await session.aclose()
    except Exception as e:
        logger.error(f"[CLEANUP] Error closing session: {e}")

    session_task.cancel()
    try:
        await session_task
    except asyncio.CancelledError:
        pass

    full_log = grievance_tracker.get_full_grievance()
    if full_log.strip():
        logger.info(f"[CLEANUP] Saving transcript ({len(full_log)} chars) to SQLite...")
        record_id = await db_manager.save_grievance(full_log, selected_lang_code)
        logger.info(f"[CLEANUP] ✓ Saved — ID: {record_id}")
    else:
        logger.warning("[CLEANUP] ⚠ No content to save")

    await ctx.room.disconnect()
    logger.info("[END] ✓ Session complete")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))