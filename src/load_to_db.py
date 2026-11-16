"""
load_to_db.py
-------------
Loads the synthetic two-wheeler loan dataset into a SQLite database (loans.db).
Uses Python's built-in sqlite3 — no external installation required.
"""

import sqlite3
import pandas as pd
import os

DB_PATH  = "loans.db"
CSV_PATH = os.path.join("data", "two_wheeler_loans.csv")

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Connect & write to SQLite ─────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

# Replace the table if re-running this script
df.to_sql("loans", conn, if_exists="replace", index=False)

# Verify row count
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM loans")
count = cursor.fetchone()[0]
print(f"Rows in SQLite 'loans' table: {count}")

# Create a simple index for faster queries
cursor.execute("CREATE INDEX IF NOT EXISTS idx_city_tier ON loans(city_tier)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_employment ON loans(employment_type)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_defaulted  ON loans(defaulted)")
conn.commit()

conn.close()
print(f"Database saved -> {DB_PATH}")
