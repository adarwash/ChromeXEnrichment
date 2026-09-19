"""SQLite persistence shared across threads/processes. No stale cache fallback."""
import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS budgets (key TEXT PRIMARY KEY, day INTEGER, used INTEGER);
                CREATE TABLE IF NOT EXISTS rates (key TEXT PRIMARY KEY, next_at REAL NOT NULL, blocked_until REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, request TEXT NOT NULL, result TEXT NOT NULL, created REAL NOT NULL);
            ''')
        Path(path).chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def cached(self, key, now=None):
        with self.connect() as db:
            row = db.execute("SELECT payload,expires FROM cache WHERE key=?", (key,)).fetchone()
        if row and row[1] > (time.time() if now is None else now):
            return json.loads(row[0])
        return None

    def cache(self, key, payload, ttl):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?)", (key, json.dumps(payload), time.time() + ttl))

    def reserve(self, key, spacing, now=None):
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT next_at,blocked_until FROM rates WHERE key=?", (key,)).fetchone()
            next_at, blocked = row if row else (0, 0)
            if blocked > now:
                return {"blocked_until": blocked, "wait": 0}
            at = max(now, next_at)
            # Bound queues: callers can resubmit instead of exhausting worker threads.
            if at - now > 15:
                return {"blocked_until": at, "wait": 0}
            db.execute("INSERT OR REPLACE INTO rates VALUES(?,?,?)", (key, at + spacing, blocked))
            return {"blocked_until": 0, "wait": at - now}

    def budget(self, key, limit, now=None):
        now = time.time() if now is None else now
        day = int(now // 86400)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT day,used FROM budgets WHERE key=?", (key,)).fetchone()
            used = row[1] if row and row[0] == day else 0
            if used >= limit:
                return (day + 1) * 86400
            db.execute("INSERT OR REPLACE INTO budgets VALUES(?,?,?)", (key, day, used + 1))
        return 0

    def block(self, key, until):
        with self.connect() as db:
            db.execute("INSERT INTO rates VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET blocked_until=MAX(blocked_until,excluded.blocked_until)", (key, 0, until))

    def save(self, record_id, request, result):
        with self.connect() as db:
            db.execute("INSERT INTO records VALUES(?,?,?,?)", (record_id, json.dumps(request), json.dumps(result), time.time()))

    def get(self, record_id):
        with self.connect() as db:
            row = db.execute("SELECT request,result FROM records WHERE id=?", (record_id,)).fetchone()
        return {"request": json.loads(row[0]), "result": json.loads(row[1])} if row else None

    def prune(self, retention_days=30):
        with self.connect() as db:
            db.execute("DELETE FROM cache WHERE expires < ?", (time.time(),))
            db.execute("DELETE FROM records WHERE created < ?", (time.time() - retention_days * 86400,))
            db.execute("DELETE FROM budgets WHERE day < ?", (int(time.time() // 86400) - 2,))
            db.execute("DELETE FROM rates WHERE next_at < ? AND blocked_until < ?", (time.time() - 86400, time.time() - 86400))
