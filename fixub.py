import sqlite3

# Connect to your app's database
conn = sqlite3.connect("user_data.db")
cursor = conn.cursor()

# Drop the old table if it exists (no password_hash column)
cursor.execute("DROP TABLE IF EXISTS users;")

# Recreate with the correct schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    display_name TEXT
);
""")

conn.commit()
conn.close()

print("✅ users table recreated successfully with password_hash and display_name columns.")
