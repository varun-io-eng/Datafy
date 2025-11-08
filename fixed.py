import sqlite3

conn = sqlite3.connect("user_data.db")
cursor = conn.cursor()

# Drop old table safely
cursor.execute("DROP TABLE IF EXISTS users;")

# Recreate it with the correct schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    display_name TEXT
);
""")

conn.commit()
conn.close()

print("✅ Fixed: 'users' table recreated with password_hash column.")
