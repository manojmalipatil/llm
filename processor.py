import os
import sqlite3
import json
import time
from datetime import datetime
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# New Google GenAI SDK imports
from google import genai
from google.genai import types

# Load variables from the .env file
load_dotenv() 

# =========================
# Configuration
# =========================
DB_PATH = "grievance.db"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize the new Client
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash" 

# =========================
# Helper Functions
# =========================
def translate_to_english(text, language):
    """Translate text to English using Deep Translator."""
    if not text: return ""
    if language and language.lower() in ["en", "english"]: return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def send_to_gemini(transcript):
    """Send to Gemini for structured analysis using the new SDK."""
    
    system_instruction = """
    You are an AI that extracts structured data from grievance transcripts.
    Always respond with valid JSON.
    """

    user_prompt = f"""
    Analyze this transcript:
    "{transcript}"
    
    Extract the following fields:
    1. "name": Person's name (or "Unknown").
    2. "emp_id": Employee ID like EMP123 (or "Unknown").
    3. "location": Branch, City, or Office (or "Unknown").
    4. "category": Classification of the grievance (e.g., POSH, Managerial, Data, Hygiene, Compensation, Workplace Environment, Conflict, Career, Attendance).
    5. "priority": High, Medium, or Low.
    6. "sentiment": Positive, Neutral, or Negative.
    7. "summary": Summary of the grievance.
    8. "tags": A list of keywords (up to 5 relevant keywords).
    9. "department": Department (Extract specific department e.g., Sales, IT, HR, Logistics. If NOT found, return "General").

    Return JSON format only:
    {{
        "name": "...", 
        "emp_id": "...", 
        "location": "...", 
        "category": "...", 
        "priority": "...", 
        "sentiment": "...", 
        "summary": "...", 
        "tags": ["..."], 
        "department": "..."
    }}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text)

    except Exception as e:
        print(f"!! Gemini API Error: {e}")
        return None

# =========================
# Main Logic (Now Continuous)
# =========================
def process_grievances_continuously():
    print("🚀 Continuous Grievance Processor Started.")
    print("Polling database for 'Pending' grievances every 5 seconds...\n")
    
    while True:
        try:
            # 1. Open DB connection per cycle to avoid locking the DB
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 2. Query ONLY 'Pending' status
            cursor.execute("SELECT id, transcript, language FROM grievances WHERE LOWER(TRIM(status)) = 'pending'")
            rows = cursor.fetchall()

            # 3. If no new grievances, sleep and try again
            if not rows:
                conn.close()
                time.sleep(5)
                continue

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(rows)} pending grievances. Processing...")

            for idx, (gid, transcript, language) in enumerate(rows, 1):
                print(f"  -> Processing ID {gid}...", end=" ", flush=True)

                en_text = translate_to_english(transcript, language)
                analysis = send_to_gemini(en_text)

                if not analysis:
                    print("❌ Failed (Skipping)")
                    continue

                llm_location = analysis.get("location", "Unknown")
                tags_data = analysis.get("tags", [])
                tags_str = ",".join(tags_data) if isinstance(tags_data, list) else str(tags_data)

                try:
                    # Updates data and sets status to 'Recorded'
                    cursor.execute("""
                        UPDATE grievances SET
                            translated_transcript = ?,
                            category = ?,
                            priority = ?,
                            sentiment = ?,
                            summary = ?,
                            tags = ?,
                            department = ?,
                            processed_at = ?,
                            analysis_json = ?,
                            name = ?,
                            emp_id = ?, 
                            location = ?,
                            status = 'Recorded'
                        WHERE id = ?
                    """, (
                        en_text,
                        analysis.get("category", "Uncategorized"),
                        analysis.get("priority", "Medium"),
                        analysis.get("sentiment", "Neutral"),
                        analysis.get("summary", ""),
                        tags_str,
                        analysis.get("department", "General"),
                        datetime.now().isoformat(),
                        json.dumps(analysis),
                        analysis.get("name", "Unknown"),
                        analysis.get("emp_id", "Unknown"),
                        llm_location,
                        gid
                    ))
                    
                    conn.commit()
                    print(f"✅ Done. Loc: {llm_location}")
                    
                except sqlite3.Error as e:
                    print(f"❌ Database Error for ID {gid}: {e}")

                # API rate limit buffer
                time.sleep(1)

            conn.close()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch complete. Resuming polling...\n")

        except Exception as e:
            print(f"‼️ Critical loop error: {e}")
            time.sleep(5) # Prevent rapid error looping if DB is locked

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not found. Please check your .env file.")
    else:
        process_grievances_continuously()