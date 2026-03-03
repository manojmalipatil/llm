"""
backend.py — Token server for Samasta Grievance Bot
Provides:
  POST /api/token           → generate LiveKit access token
  GET  /api/grievances      → list saved grievances (admin)
  GET  /api/grievances/{id} → single grievance detail (admin)
  GET  /health              → health check
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit.api import AccessToken, VideoGrants

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GrievanceServer")

app = FastAPI(title="Samasta Grievance Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL        = os.getenv("LIVEKIT_URL")
ADMIN_SECRET       = os.getenv("ADMIN_SECRET", "changeme")
DB_PATH            = os.getenv("DB_PATH", "grievance.db")


# ─────────────────────────────────────────────
# DB INIT ON STARTUP
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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
    logger.info(f"[DB] SQLite ready at {DB_PATH}")


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class TokenRequest(BaseModel):
    employee_id:   str
    language_code: Optional[str] = "en"
    room_name:     Optional[str] = None

class TokenResponse(BaseModel):
    token:         str
    room_name:     str
    livekit_url:   str
    language_code: str


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def verify_admin(x_admin_secret: str = Header(...)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/token", response_model=TokenResponse)
async def generate_token(req: TokenRequest):
    """Generate a LiveKit access token for an employee session."""
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    room_name = req.room_name or (
        f"grievance-{req.employee_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    )

    try:
        token = (
            AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(req.employee_id)
            .with_name(f"Employee {req.employee_id}")
            .with_grants(VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            ))
            .with_metadata(json.dumps({"language_code": req.language_code}))
            .to_jwt()
        )
        logger.info(f"[TOKEN] employee={req.employee_id} room={room_name} lang={req.language_code}")
        return TokenResponse(
            token=token,
            room_name=room_name,
            livekit_url=LIVEKIT_URL or "",
            language_code=req.language_code,
        )
    except Exception as e:
        logger.error(f"[TOKEN] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Token generation failed: {e}")


@app.get("/api/grievances")
async def list_grievances(
    limit: int = 50,
    language: Optional[str] = None,
    _: None = Depends(verify_admin),
):
    """Admin: list all grievances."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if language:
            rows = conn.execute(
                "SELECT id, timestamp, language, status, transcript FROM grievances "
                "WHERE language = ? ORDER BY created_at DESC LIMIT ?",
                (language, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, language, status, transcript FROM grievances "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return {"count": len(rows), "grievances": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/grievances/{grievance_id}")
async def get_grievance(
    grievance_id: str,
    _: None = Depends(verify_admin),
):
    """Admin: fetch a single grievance by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM grievances WHERE id = ?", (grievance_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Grievance not found")
        return dict(row)
    finally:
        conn.close()


# Serve frontend
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse("static/grievance-call.html")
except Exception:
    @app.get("/")
    async def root():
        return {"message": "Backend running. Place grievance-call.html in /static/"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)