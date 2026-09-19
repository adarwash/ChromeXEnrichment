"""Zero-fee candidate search. Search results are never verification evidence."""
from __future__ import annotations
import hashlib
import http.client
import json
import re
import time
from urllib.parse import urlencode, urlsplit
from .http import FetchError, MAX_BYTES, canonical_url, remaining, retry_delay
from .providers import BraveSearch


class LocalSearch:
    name = "local"
    configured = True

    def __init__(self, store):
        self.store = store

    def search(self, query, count=10, offset=0, refresh=False):
        rows = self.store.search_catalog(query, count, offset)
        return rows, {"provider": self.name, "scope": "imported_and_previously_fetched_urls_only", "queried_at": time.time(),
                      "notes": ["Local candidates need fresh source verification; this is not a whole-web index."]}


class SearXNGSearch:
    """Dedicated client to an operator-controlled loopback service, not an SSRF exception.

    End users cannot supply the endpoint or a destination URL. No redirects,
    credentials, proxy environment variables or arbitrary internal hosts are used.
    Returned public URLs still go through Fetcher's normal public-only policy.
    """
    name = "searxng"
    configured = True

    def __init__(self, store, endpoint="http://127.0.0.1:8888", spacing=10.0, daily_limit=100):
        p = urlsplit(endpoint)
        if p.scheme != "http" or p.hostname != "127.0.0.1" or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment or not 1024 <= (p.port or 80) <= 65535:
            raise ValueError("SEARXNG_URL must be http://127.0.0.1:<port>, port 1024-65535")
        self.store, self.endpoint, self.port = store, endpoint.rstrip("/"), p.port
        self.spacing, self.daily_limit = max(10.0, spacing), max(0, daily_limit)

    def _request(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=min(20, remaining()))
        try:
            conn.request("GET", path, headers={"Accept": "application/json", "Accept-Encoding": "identity"})
            resp = conn.getresponse()
            data, size, deadline = [], 0, time.monotonic() + min(20, remaining())
            while True:
                if time.monotonic() > deadline:
                    raise FetchError("searxng_response_deadline")
                part = resp.read1(min(65536, MAX_BYTES + 1 - size))
                if not part:
                    break
                size += len(part)
                if size > MAX_BYTES:
                    raise FetchError("searxng_response_too_large")
                data.append(part)
            return resp.status, dict((k.lower(), v) for k, v in resp.getheaders()), b"".join(data)
        except (OSError, http.client.HTTPException):
            raise FetchError("searxng_unavailable") from None
        finally:
            conn.close()

    def search(self, query, count=10, offset=0, refresh=False):
        # SearXNG has its own query language (!engine, !!bang redirects, :language).
        # Send plain candidate queries; callers cannot select/bypass engine policy.
        query = re.sub(r"!+", " ", query)
        query = re.sub(r"(?<!\S):\S+", " ", query)
        query = " ".join(query.split())
        path = "/search?" + urlencode({"q": query, "format": "json", "categories": "general", "language": "en-GB", "pageno": offset + 1, "safesearch": 1})
        key = "searxng:" + hashlib.sha256((self.endpoint + path).encode()).hexdigest()
        if not refresh:
            cached = self.store.cached(key)
            if cached:
                return cached["rows"][:count], cached["evidence"]
        rate_key = "searxng-search"
        slot = self.store.reserve(rate_key, self.spacing)
        if slot["blocked_until"]:
            raise FetchError("searxng_cooldown", retry_at=slot["blocked_until"])
        if slot["wait"] >= remaining():
            raise FetchError("request_budget_exhausted")
        if slot["wait"]:
            time.sleep(slot["wait"])
        reset_at = self.store.budget(rate_key, self.daily_limit)
        if reset_at:
            raise FetchError("searxng_daily_budget_exhausted", retry_at=reset_at)
        try:
            status, headers, raw = self._request(path)
        except FetchError:
            self.store.block(rate_key, time.time() + 300)
            raise
        if status != 200:
            until = time.time() + retry_delay(headers.get("retry-after"), 3600 if status in (401, 403) else 300)
            self.store.block(rate_key, until)
            raise FetchError("searxng_http_" + str(status), retry_at=until)
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError()
        except (ValueError, TypeError):
            raise FetchError("searxng_invalid_response") from None
        failures = payload.get("unresponsive_engines") or []
        if failures:
            # Stop repeated searches into an upstream block; partial candidates
            # may be returned but cannot count as source verification.
            self.store.block(rate_key, time.time() + 300)
            if not payload["results"]:
                raise FetchError("searxng_engines_unavailable", retry_at=time.time() + 300)
        rows, seen = [], set()
        for r in payload["results"][:100]:
            if not isinstance(r, dict) or not isinstance(r.get("url"), str):
                continue
            try:
                url = canonical_url(r["url"])
            except FetchError:
                continue
            if url in seen:
                continue
            seen.add(url)
            rows.append({"url": url, "title": str(r.get("title", ""))[:500], "snippet": str(r.get("content", ""))[:1000], "status": "discovered_unverified"})
        evidence = {"provider": self.name, "fetched_at": time.time(), "upstream_failures": failures,
                    "notes": ["Self-hosted metasearch remains subject to upstream availability and blocking."]}
        self.store.cache(key, {"rows": rows, "evidence": evidence}, 86400)
        return rows[:count], evidence


class CandidateSearch:
    """Always search our catalogue; optional upstream is explicitly configured."""
    def __init__(self, store, upstream=None):
        self.local, self.upstream = LocalSearch(store), upstream
        self.name = upstream.name if upstream else "local"
        self.configured = True

    def search(self, query, count=10, offset=0, refresh=False):
        local, evidence = self.local.search(query, count, offset, refresh)
        if not self.upstream:
            return local, evidence
        try:
            remote, remote_evidence = self.upstream.search(query, count, offset, refresh)
        except FetchError as exc:
            if not local:
                raise
            return local, {**evidence, "upstream_error": exc.as_dict()}
        seen = set()
        rows = []
        for r in local + remote:
            if r["url"] not in seen:
                seen.add(r["url"])
                rows.append(r)
        return rows[:count], {"provider": self.name, "local": evidence, "upstream": remote_evidence}


def build_search(store, fetcher, provider="local", brave_key="", searxng_url="http://127.0.0.1:8888", searxng_daily_limit=100, scrapling_options=None):
    if provider == "local":
        upstream = None
    elif provider == "searxng":
        upstream = SearXNGSearch(store, searxng_url, daily_limit=searxng_daily_limit)
    elif provider == "brave":
        upstream = BraveSearch(fetcher, brave_key)
    elif provider == "scrapling":
        from .scrapling_search import ScraplingSearch
        upstream = ScraplingSearch(store, **(scrapling_options or {}))
    else:
        raise ValueError("SEARCH_PROVIDER must be local, searxng, scrapling or brave")
    return CandidateSearch(store, upstream)
