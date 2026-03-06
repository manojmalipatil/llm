import sqlite3

DB_PATH = "grievance.db"
GRIEVANCE_ID = "17774d2d-7495-4d10-b43a-814e03f49e4c"  # 👈 change this to the ID you want to delete

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Execute the delete statement
cursor.execute("""
    DELETE FROM grievances
    WHERE id = ?
""", (GRIEVANCE_ID,))

cursor.execute("ALTER TABLE grievances ADD COLUMN recording_path TEXT")

# Check how many rows were actually deleted
deleted_count = cursor.rowcount

# ⚠️ CRITICAL: You must commit the transaction to save the deletion!
conn.commit()

print("\n" + "=" * 60)
if deleted_count > 0:
    print(f"✅ SUCCESSFULLY DELETED GRIEVANCE (ID: {GRIEVANCE_ID})")
else:
    print(f"❌ No grievance found with ID = {GRIEVANCE_ID}. Nothing was deleted.")
print("=" * 60 + "\n")

conn.close()