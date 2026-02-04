import sqlite3
import os

DB_FILE = 'ZYLO_chat.db'

def update_schema():
    if not os.path.exists(DB_FILE):
        print(f"Database {DB_FILE} not found!")
        return

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # Check if columns exist
        c.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in c.fetchall()]
        
        if 'storage_quota' not in columns:
            print("Adding storage_quota column...")
            # Default 1GB = 1073741824 bytes
            c.execute("ALTER TABLE users ADD COLUMN storage_quota INTEGER DEFAULT 1073741824")
        
        if 'is_premium' not in columns:
            print("Adding is_premium column...")
            c.execute("ALTER TABLE users ADD COLUMN is_premium INTEGER DEFAULT 0")
            
        conn.commit()
        print("Schema update complete.")
        
    except Exception as e:
        print(f"Error updating schema: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    update_schema()
