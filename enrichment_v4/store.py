"""SQLite persistence shared across threads/processes. No stale cache fallback."""
import json
import re
from datetime import datetime, timezone
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
                CREATE TABLE IF NOT EXISTS search_usage (key TEXT PRIMARY KEY, month TEXT, used INTEGER);
                CREATE VIRTUAL TABLE IF NOT EXISTS catalogue USING fts5(url UNINDEXED, title, text, source UNINDEXED, indexed_at UNINDEXED, tokenize='porter unicode61');
                CREATE TABLE IF NOT EXISTS catalogue_urls (url TEXT PRIMARY KEY, row_id INTEGER UNIQUE);
                INSERT OR IGNORE INTO catalogue_urls SELECT url,MIN(rowid) FROM catalogue GROUP BY url;
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

    def monthly_search_budget(self, key, limit, now=None):
        month = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).strftime("%Y-%m")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT month,used FROM search_usage WHERE key=?", (key,)).fetchone()
            used = row[1] if row and row[0] == month else 0
            if used >= limit:
                return False
            db.execute("INSERT OR REPLACE INTO search_usage VALUES(?,?,?)", (key, month, used + 1))
        return True

    def index_candidates(self, items, source="operator_import", preserve_existing=False):
        from .http import canonical_url
        with self.connect() as db:
            for item in items:
                url = canonical_url(item["url"])
                existing = db.execute("SELECT row_id FROM catalogue_urls WHERE url=?", (url,)).fetchone()
                if preserve_existing and existing:
                    continue
                if existing:
                    db.execute("DELETE FROM catalogue WHERE rowid=?", (existing[0],))
                row = db.execute("INSERT INTO catalogue VALUES(?,?,?,?,?)", (url, item.get("title", "")[:500], item.get("text", "")[:12000], source[:200], time.time()))
                db.execute("INSERT OR REPLACE INTO catalogue_urls VALUES(?,?)", (url, row.lastrowid))
        return len(items)

    def search_catalog(self, query, count=10, offset=0):
        # Literal quoted tokens only: never pass client FTS syntax into MATCH.
        sites = re.findall(r"site:([a-z0-9.-]+)", query, re.I)
        query = re.sub(r"site:[a-z0-9.-]+", " ", query, flags=re.I)
        stop = {"and", "or", "the", "contact", "united", "kingdom", "uk", "gb", "limited", "ltd"}
        quoted = re.findall(r'"([^"\n]+)"', query)
        if quoted:
            # Enrichment queries lead with a full company name; the following
            # phone/registry ID is only a hint and may be absent from the catalogue.
            query = quoted[0]
        tokens = list(dict.fromkeys(t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in stop))[:32]
        if not tokens:
            return []
        expression = " AND ".join('"' + t + '"' for t in tokens)
        # This is candidate retrieval only; ranking does not establish identity.
        with self.connect() as db:
            rows = db.execute("SELECT url,title,text,source,indexed_at FROM catalogue WHERE catalogue MATCH ? ORDER BY rank LIMIT ? OFFSET ?", (expression, min(count * 10, 200), offset * count)).fetchall()
        from .http import publisher
        return [{"url": r[0], "title": r[1], "snippet": r[2][:1000], "catalogue_source": r[3], "indexed_at": float(r[4]), "status": "discovered_unverified"} for r in rows if not sites or publisher(r[0]) in sites][:count]

    def catalog_stats(self):
        with self.connect() as db:
            count = db.execute("SELECT count(*) FROM catalogue").fetchone()[0]
            usage = db.execute("SELECT key,month,used FROM search_usage").fetchall()
        return {"candidate_urls": count, "external_search_usage": [{"provider": row[0], "month_utc": row[1], "http_requests": row[2]} for row in usage]}

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
            db.execute("DELETE FROM catalogue_urls WHERE row_id IN (SELECT rowid FROM catalogue WHERE CAST(indexed_at AS REAL) < ?)", (time.time() - 90 * 86400,))
            db.execute("DELETE FROM catalogue WHERE CAST(indexed_at AS REAL) < ?", (time.time() - 90 * 86400,))
            db.execute("DELETE FROM cache WHERE expires < ?", (time.time(),))
            db.execute("DELETE FROM records WHERE created < ?", (time.time() - retention_days * 86400,))
            db.execute("DELETE FROM budgets WHERE day < ?", (int(time.time() // 86400) - 2,))
            db.execute("DELETE FROM rates WHERE next_at < ? AND blocked_until < ?", (time.time() - 86400, time.time() - 86400))
