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
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.speech_handle import SpeechHandle
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

# Build a direct map: short code (e.g. "hi") → Sarvam BCP-47 code (e.g. "hi-IN")
# This is derived entirely from language_config.json so it stays in sync automatically.
LANG_CODE_TO_SARVAM: dict[str, str] = {
    lang_data["code"]: lang_data["sarvam_tts_code"]
    for lang_data in SUPPORTED_LANGUAGES.values()
}
logger.info(f"[CONFIG] Sarvam code map: {LANG_CODE_TO_SARVAM}")


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
                emp_id      TEXT,
                created_at  TEXT NOT NULL,
                status      TEXT DEFAULT 'pending',
                is_confidential INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"[DB] SQLite ready at {self.db_path}")

    async def save_grievance(self, transcript: str, language: str = "en", emp_id: str = "Unknown", is_confidential: bool = False) -> str | None:
        if not transcript.strip():
            logger.warning("[DB] Empty transcript — skipping save")
            return None

        record_id    = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()
        confidential_int = 1 if is_confidential else 0

        def _write():
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO grievances (id, timestamp, transcript, language, emp_id, created_at, is_confidential)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (record_id, current_time, transcript, language, emp_id, current_time, confidential_int),
                )
                conn.commit()
                logger.info(f"[DB] Saved grievance ID: {record_id} (lang={language}, emp_id={emp_id}, confidential={confidential_int})")
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
    # Receive language from token metadata
    # -----------------------------------------------------------------------
    selected_lang_code   = "en"
    selected_sarvam_code = "en-IN"
    employee_id          = "Unknown"
    is_confidential      = False

    logger.info("[LANG] Waiting for remote participant to join and provide language metadata...")
    participant_joined = asyncio.Event()
    found_participant = None

    for p in ctx.room.remote_participants.values():
        found_participant = p
        participant_joined.set()
        break

    def on_participant_connected(p: rtc.RemoteParticipant):
        nonlocal found_participant
        found_participant = p
        participant_joined.set()

    ctx.room.on("participant_connected", on_participant_connected)

    try:
        await asyncio.wait_for(participant_joined.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("[LANG] ⚠ Timeout waiting for participant — defaulting to English.")
    finally:
        ctx.room.off("participant_connected", on_participant_connected)

    if found_participant and found_participant.metadata:
        try:
            meta = json.loads(found_participant.metadata)
            if "employee_id" in meta:
                employee_id = meta["employee_id"]
            if "is_confidential" in meta:
                is_confidential = meta["is_confidential"]
            if "language_code" in meta:
                lang_code = meta["language_code"]
                selected_lang_code = lang_code
                selected_sarvam_code = LANG_CODE_TO_SARVAM.get(lang_code, f"{lang_code}-IN")
                if lang_code not in LANG_CODE_TO_SARVAM:
                    logger.warning(
                        f"[LANG] '{lang_code}' not in language_config.json — "
                        f"using fallback TTS code '{selected_sarvam_code}'"
                    )
        except json.JSONDecodeError:
            pass

    grievance_tracker.set_language(selected_lang_code)
    logger.info(f"[LANG] Set from metadata → {selected_lang_code} / TTS: {selected_sarvam_code} / emp_id: {employee_id} / confidential: {is_confidential}")

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
        stt=sarvam.STT(language=selected_sarvam_code),
        llm=groq.LLM(model="llama-3.3-70b-versatile", temperature=0.3),
        tts=sarvam.TTS(target_language_code=selected_sarvam_code, speaker="anushka"),
    )

    @function_tool
    async def end_call(ctx: RunContext, confirmation: str = "yes"):
        """
        End the grievance collection call.
        Call this ONLY when the user explicitly says they are done,
        wants to hang up, or has no more grievances to share.
        """
        logger.info("[TOOL] end_call triggered — signalling Stage 3")
        should_end_call.set()
        
        # We add "in the user's language" to guarantee the LLM doesn't summarize in English.
        return (
            "System Instruction: The grievance has been successfully recorded. "
            "Please summarize the recorded grievance in 1-2 sentences in the user's language, "
            "say a polite goodbye, and do NOT call any tools."
        )

    grievance_agent = Agent(instructions=prompt, tools=[end_call])
    # NOTE: farewell_agent removed entirely.

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

    # -----------------------------------------------------------------------
    # STAGE 1 — Initial greeting
    # -----------------------------------------------------------------------
    try:
        session.allow_interruptions = False
        logger.info("[STAGE] Generating initial greeting (Interruptions Disabled)")
        greeting_handle = await session.generate_reply()

        try:
            await asyncio.wait_for(greeting_handle.wait_for_playout(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("[STAGE] ⚠ Timeout waiting for greeting playout — continuing anyway")

        logger.info("[STAGE] ✓ Initial greeting finished playing")

    except Exception as e:
        logger.error(f"[STAGE] Error generating greeting: {e}")
    finally:
        session.allow_interruptions = True
        logger.info("[STAGE] Interruptions re-enabled for grievance collection")

    # -----------------------------------------------------------------------
    # STAGE 2 — Grievance collection
    # -----------------------------------------------------------------------
    logger.info("[STAGE] Collecting grievance — waiting for end_call...")
    await should_end_call.wait()

    # -----------------------------------------------------------------------
    # STAGE 3 — Farewell playout
    # -----------------------------------------------------------------------
    logger.info("[STAGE] Tool called. Waiting for natural farewell summary to complete...")
    
    # Disable interruptions so the user can't cut off the farewell summary
    session.allow_interruptions = False

    participant_disconnected = asyncio.Event()
    
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(p: rtc.RemoteParticipant):
        participant_disconnected.set()

    try:
        # 25 seconds gives the LLM and TTS plenty of time to process and speak the final sentence.
        await asyncio.wait_for(participant_disconnected.wait(), timeout=25.0)
        logger.info("[STAGE] ✓ User disconnected naturally")
    except asyncio.TimeoutError:
        logger.info("[STAGE] ⚠ Time elapsed for farewell — forcing disconnect")
    finally:
        session.allow_interruptions = True

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
        record_id = await db_manager.save_grievance(
            transcript=full_log,
            language=selected_lang_code,
            emp_id=employee_id,
            is_confidential=is_confidential
        )
        logger.info(f"[CLEANUP] ✓ Saved — ID: {record_id}")
    else:
        logger.warning("[CLEANUP] ⚠ No content to save")

    await ctx.room.disconnect()
    logger.info("[END] ✓ Session complete")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))