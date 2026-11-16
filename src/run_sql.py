"""
run_sql.py
----------
Runs every query in sql/analysis.sql against loans.db and prints the results.
Saves each result to reports/ as a CSV for future reference.
"""

import sqlite3
import pandas as pd
import os

DB_PATH  = "loans.db"
SQL_PATH = os.path.join("sql", "analysis.sql")
REPORTS  = "reports"
os.makedirs(REPORTS, exist_ok=True)

# ── Parse SQL file into individual queries ─────────────────────────────────────
with open(SQL_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

# Split on the "Query N:" comment lines to get labelled blocks
import re
blocks = re.split(r"--\s*[-]+\s*Query\s+\d+[:.]", raw)
# Fallback: split on double newline + comment header
if len(blocks) <= 1:
    blocks = [b.strip() for b in raw.split(";") if b.strip()]
else:
    blocks = [b.strip() for b in blocks[1:]]  # drop preamble

# Clean each block: remove trailing semicolons and comment-only lines
def clean_query(text):
    lines = [l for l in text.splitlines() if not l.strip().startswith("--")]
    sql   = "\n".join(lines).strip().rstrip(";").strip()
    return sql

queries = [clean_query(b) for b in blocks if clean_query(b)]

# Query labels for display
labels = [
    "Q1 : Overall Default Rate",
    "Q2 : Default Rate by City Tier",
    "Q3 : Default Rate by Employment Type",
    "Q4 : EMI-to-Income Risk Buckets",
    "Q5 : Top 10% Riskiest Customers (NTILE)",
    "Q6 : Above-Average EMI Burden Defaults (Subquery)",
    "Q7 : Monthly Default Trend",
    "Q8 : City-Tier x Employment Risk Rank",
]

# ── Run queries ────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

for i, (sql, label) in enumerate(zip(queries, labels)):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    try:
        df = pd.read_sql_query(sql, conn)
        print(df.to_string(index=False))
        # Save result
        safe_label = label.replace(" ", "_").replace(":", "").replace("/", "-")
        df.to_csv(os.path.join(REPORTS, f"{safe_label}.csv"), index=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  SQL was:\n{sql[:300]}")

conn.close()
print(f"\n\nAll query results saved to {REPORTS}/")
