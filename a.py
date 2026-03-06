import asyncio
import os
import sqlite3
import uuid
import json
import wave
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

# --- Recordings Directory ---
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
logger.info(f"[RECORDING] Recordings directory ready: {RECORDINGS_DIR}")


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
# Audio Recorder — captures participant audio to a WAV file
# ---------------------------------------------------------------------------
class AudioRecorder:
    """
    Subscribes to a RemoteParticipant's audio track and writes all
    received PCM frames to a WAV file.  The file is finalised (and
    its path returned) when stop() is called.
    """

    SAMPLE_RATE  = 16_000   # Hz  — must match the audio stream
    NUM_CHANNELS = 1        # mono

    def __init__(self, record_id: str):
        self.record_id   = record_id
        self.output_path = os.path.join(RECORDINGS_DIR, f"{record_id}.wav")
        self._frames: list[bytes] = []
        self._stream: rtc.AudioStream | None = None
        self._task:   asyncio.Task  | None = None
        self._stopped = False

    # ------------------------------------------------------------------
    def start(self, track: rtc.Track):
        """Attach to an audio track and begin buffering frames."""
        self._stream = rtc.AudioStream(
            track,
            sample_rate=self.SAMPLE_RATE,
            num_channels=self.NUM_CHANNELS,
        )
        self._task = asyncio.create_task(self._record_loop())
        logger.info(f"[RECORDING] ▶ Started recording → {self.output_path}")

    async def _record_loop(self):
        try:
            async for event in self._stream:
                if self._stopped:
                    break
                if isinstance(event, rtc.AudioFrameEvent):
                    self._frames.append(bytes(event.frame.data))
        except Exception as e:
            logger.error(f"[RECORDING] Error in record loop: {e}")

    # ------------------------------------------------------------------
    async def stop(self) -> str | None:
        """Stop recording and flush the WAV file.  Returns the file path."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if not self._frames:
            logger.warning("[RECORDING] No audio frames captured — skipping WAV write")
            return None

        def _write_wav():
            with wave.open(self.output_path, "wb") as wf:
                wf.setnchannels(self.NUM_CHANNELS)
                wf.setsampwidth(2)   # 16-bit PCM = 2 bytes per sample
                wf.setframerate(self.SAMPLE_RATE)
                for frame in self._frames:
                    wf.writeframes(frame)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_wav)
        logger.info(f"[RECORDING] ■ Saved WAV → {self.output_path} "
                    f"({len(self._frames)} frames)")
        return self.output_path


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
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                transcript      TEXT NOT NULL,
                language        TEXT,
                created_at      TEXT NOT NULL,
                status          TEXT DEFAULT 'pending',
                recording_path  TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"[DB] SQLite ready at {self.db_path}")

    async def save_grievance(
        self,
        transcript:     str,
        language:       str = "en",
        recording_path: str | None = None,
        record_id:      str | None = None,   # ← pass in so recorder & DB share the same ID
    ) -> str | None:
        if not transcript.strip():
            logger.warning("[DB] Empty transcript — skipping save")
            return None

        if record_id is None:
            record_id = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()

        def _write():
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO grievances
                        (id, timestamp, transcript, language, created_at, recording_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (record_id, current_time, transcript, language,
                     current_time, recording_path),
                )
                conn.commit()
                logger.info(f"[DB] Saved grievance ID: {record_id} "
                            f"(lang={language}, recording={recording_path})")
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

    # Generate the shared record ID up-front so recorder & DB use the same UUID
    record_id = str(uuid.uuid4())
    logger.info(f"[SESSION] Record ID for this call: {record_id}")

    audio_recorder: AudioRecorder | None = None

    await ctx.connect()

    # -----------------------------------------------------------------------
    # Receive language from token metadata
    # -----------------------------------------------------------------------
    selected_lang_code   = "en"
    selected_sarvam_code = "en-IN"

    logger.info("[LANG] Waiting for remote participant to join and provide language metadata...")
    participant_joined = asyncio.Event()
    found_participant  = None

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
            if "language_code" in meta:
                lang_code = meta["language_code"]
                for _, lang_data in SUPPORTED_LANGUAGES.items():
                    if lang_data["code"] == lang_code:
                        selected_lang_code   = lang_code
                        selected_sarvam_code = lang_data.get("sarvam_tts_code", "en-IN")
                        break
                else:
                    selected_lang_code = lang_code
        except json.JSONDecodeError:
            pass

    grievance_tracker.set_language(selected_lang_code)
    logger.info(f"[LANG] Set from metadata → {selected_lang_code} / TTS: {selected_sarvam_code}")

    # -----------------------------------------------------------------------
    # Subscribe to participant audio track for recording
    # -----------------------------------------------------------------------
    def on_track_subscribed(
        track:       rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        nonlocal audio_recorder
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_recorder = AudioRecorder(record_id)
            audio_recorder.start(track)

    ctx.room.on("track_subscribed", on_track_subscribed)

    # Also check for tracks that are already published
    if found_participant:
        for pub in found_participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_recorder = AudioRecorder(record_id)
                audio_recorder.start(pub.track)
                break

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

    # Stop recording and get WAV path
    recording_path = None
    if audio_recorder:
        recording_path = await audio_recorder.stop()

    # Save transcript + recording path under the same record_id
    full_log = grievance_tracker.get_full_grievance()
    if full_log.strip():
        logger.info(f"[CLEANUP] Saving transcript ({len(full_log)} chars) to SQLite...")
        saved_id = await db_manager.save_grievance(
            full_log,
            selected_lang_code,
            recording_path=recording_path,
            record_id=record_id,
        )
        logger.info(f"[CLEANUP] ✓ Saved — ID: {saved_id}, recording: {recording_path}")
    else:
        logger.warning("[CLEANUP] ⚠ No content to save")

    await ctx.room.disconnect()
    logger.info("[END] ✓ Session complete")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))