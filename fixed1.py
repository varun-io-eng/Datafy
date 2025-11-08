import sqlite3

conn = sqlite3.connect("user_data.db")
cursor = conn.cursor()

# Create table with correct schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    display_name TEXT
);
""")

conn.commit()
conn.close()
print("✅ 'users' table created successfully with password_hash column.")
