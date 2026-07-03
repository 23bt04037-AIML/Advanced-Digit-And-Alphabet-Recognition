"""
migrate_sqlite_to_mysql.py
──────────────────────────
Copies data from the existing SQLite database (digit_recognition_db.db)
into the MySQL database configured in .env  (DB_HOST / DB_NAME / etc.).

Usage:
    cd <project-root>
    python deployment/migrate_sqlite_to_mysql.py
"""
import os, sys, json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ── Source: SQLite ────────────────────────────────────────────────────────────
SQLITE_PATH = ROOT / "digit_recognition_db.db"
if not SQLITE_PATH.exists():
    print(f"SQLite DB not found at {SQLITE_PATH} – nothing to migrate.")
    sys.exit(0)

sqlite_conn = sqlite3.connect(SQLITE_PATH)
sqlite_conn.row_factory = sqlite3.Row

# ── Target: MySQL ─────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "digit_recognition_db")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD", "")

MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

try:
    engine = create_engine(MYSQL_URL, pool_pre_ping=True)
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    print(f"✅  Connected to MySQL: {DB_HOST}/{DB_NAME}")
except Exception as e:
    print(f"❌  Cannot connect to MySQL: {e}")
    sys.exit(1)


def migrate_table(sqlite_table: str, mysql_table: str, col_map: dict | None = None):
    """Copy all rows from sqlite_table → mysql_table."""
    rows = sqlite_conn.execute(f"SELECT * FROM {sqlite_table}").fetchall()
    if not rows:
        print(f"   (empty) {sqlite_table}")
        return

    Session = sessionmaker(bind=engine)
    session = Session()
    inserted = 0
    for row in rows:
        d = dict(row)
        if col_map:
            d = {col_map.get(k, k): v for k, v in d.items()}
        # JSON fields stored as text in SQLite → keep as-is; MySQL JSON col handles it
        cols  = ", ".join(f"`{k}`" for k in d)
        vals  = ", ".join(f":{k}" for k in d)
        try:
            session.execute(text(f"INSERT IGNORE INTO {mysql_table} ({cols}) VALUES ({vals})"), d)
            inserted += 1
        except Exception as ex:
            print(f"   ⚠  Row skip ({sqlite_table} id={d.get('id','?')}): {ex}")

    session.commit()
    session.close()
    print(f"   ✅  {sqlite_table} → {mysql_table} : {inserted}/{len(rows)} rows")


print("\n── Migrating tables ─────────────────────────────────────────────────")

# Check which tables exist in SQLite
existing = {r[0] for r in sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
print(f"   Found SQLite tables: {existing}")

for sqlite_tbl, mysql_tbl in [
    ("prediction_logs", "prediction_logs"),
    ("audit_logs",       "audit_logs"),
    ("runtime_logs",     "runtime_logs"),
    ("model_metrics",    "model_metrics"),
]:
    if sqlite_tbl in existing:
        migrate_table(sqlite_tbl, mysql_tbl)
    else:
        print(f"   (skip) {sqlite_tbl} – not in SQLite DB")

sqlite_conn.close()
print("\n── Migration complete ────────────────────────────────────────────────")
print("   Open phpMyAdmin → digit_recognition_db to verify the data.")
