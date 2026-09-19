"""Polite bounded fetching, persistent backoff and DNS-pinned public connections."""
from __future__ import annotations
import base64
import contextvars
import hashlib
import http.client
import ipaddress
import json
import random
import socket
import ssl
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit, urljoin
from urllib.robotparser import RobotFileParser
import tldextract

_extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
API_HOSTS = {"api.company-information.service.gov.uk", "api.search.brave.com"}
USER_AGENT = "ChromeXEnrichment/4.0"
MAX_BYTES = 2_000_000
DEADLINE = contextvars.ContextVar("request_deadline", default=None)


def remaining():
    deadline = DEADLINE.get()
    left = deadline - time.monotonic() if deadline else 600
    if left <= 0:
        raise FetchError("request_budget_exhausted")
    return left


class FetchError(Exception):
    def __init__(self, code, url="", retry_at=None):
        super().__init__(code)
        self.code, self.url, self.retry_at = code, url, retry_at

    def as_dict(self):
        return {"code": self.code, "url": self.url, "retry_at": self.retry_at}


def publisher(url):
    host = (urlsplit(url if "://" in url else "https://" + url).hostname or "").lower()
    result = _extract(host)
    return result.top_domain_under_public_suffix or host


def canonical_url(url):
    if "://" not in url:
        url = "https://" + url
    if len(url) > 2048 or any(ord(c) < 33 for c in url) or "\\" in url:
        raise FetchError("invalid_url")
    try:
        p = urlsplit(url)
        if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
            raise ValueError()
        if p.port not in (None, 80 if p.scheme == "http" else 443):
            raise ValueError()
        host = p.hostname.encode("idna").decode().lower().rstrip(".")
        if ":" in host or host in ("localhost",) or "." not in host:
            raise ValueError()
        return urlunsplit((p.scheme, host, p.path or "/", p.query, ""))
    except ValueError:
        raise FetchError("unsafe_url") from None


def public_addresses(host, port):
    try:
        addresses = sorted({a[4][0] for a in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
        if not addresses or any(not ipaddress.ip_address(a).is_global or ipaddress.ip_address(a).is_multicast or ipaddress.ip_address(a).is_reserved for a in addresses):
            raise FetchError("non_public_address")
        return addresses
    except (socket.gaierror, ValueError):
        raise FetchError("dns_failed") from None


def _public_connection(address, timeout=15, source_address=None, **kwargs):
    # Resolve once, validate ALL answers, and connect to that numeric answer.
    # HTTPSConnection still uses the original hostname for SNI/certificate checks.
    host, port = address
    addresses = public_addresses(host, port)
    return socket.create_connection((addresses[0], port), timeout, source_address)


def retry_delay(value, default=300, now=None):
    now = time.time() if now is None else now
    try:
        return max(default, float(value))
    except (ValueError, TypeError):
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(default, parsed.timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return default


class Fetcher:
    def __init__(self, store, user_agent=USER_AGENT, website_spacing=3.0, search_spacing=1.1, search_monthly_limit=0):
        self.store = store
        self.search_monthly_limit = max(0, int(search_monthly_limit))
        self.user_agent = user_agent
        self.website_spacing = max(3.0, website_spacing)
        self.search_spacing = max(1.1, search_spacing)

    def _wire(self, url, headers):
        p = urlsplit(url)
        cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
        kwargs = {"context": ssl.create_default_context()} if p.scheme == "https" else {}
        conn = cls(p.hostname, timeout=min(15, remaining()), **kwargs)
        conn._create_connection = _public_connection
        try:
            conn.request("GET", urlunsplit(("", "", p.path or "/", p.query, "")), headers=headers)
            resp = conn.getresponse()
            # Identity encoding and streaming byte cap prevent decompression bombs.
            chunks, size, deadline = [], 0, time.monotonic() + min(30, remaining())
            while True:
                if time.monotonic() > deadline:
                    raise FetchError("response_deadline", url)
                part = resp.read1(min(65536, MAX_BYTES + 1 - size))
                if not part:
                    break
                size += len(part)
                if size > MAX_BYTES:
                    raise FetchError("response_too_large", url)
                chunks.append(part)
            data = b"".join(chunks)
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
        except FetchError:
            raise
        except (OSError, http.client.HTTPException):
            raise FetchError("network_error", url) from None
        finally:
            conn.close()

    def get(self, url, *, kind="web", headers=None, ttl=3600, refresh=False, robots=True):
        remaining()
        url = canonical_url(url)
        initial_host = urlsplit(url).hostname
        if kind != "web" and initial_host not in API_HOSTS:
            raise FetchError("unknown_api", url)
        hdr = {"User-Agent": self.user_agent, "Accept-Encoding": "identity", "Accept": "application/json" if kind != "web" else "text/html,text/plain"}
        hdr.update(headers or {})
        # Include auth identity in cache key, never its raw secret.
        key = hashlib.sha256((url + json.dumps(hdr, sort_keys=True)).encode()).hexdigest()
        if not refresh:
            cached = self.store.cached(key)
            if cached:
                return cached
        current = url
        for hop in range(5):
            host = urlsplit(current).hostname
            if kind != "web" and host != initial_host:
                raise FetchError("api_cross_origin_redirect", current)
            if kind == "web" and robots:
                self._robots(current, refresh=refresh)
            rate_key = "api:" + host if kind != "web" else "web:" + publisher(current)
            spacing = 1.0 if host == "api.company-information.service.gov.uk" else self.search_spacing if kind != "web" else self.website_spacing
            for attempt in range(3):
                slot = self.store.reserve(rate_key, spacing)
                if slot["blocked_until"]:
                    raise FetchError("source_cooldown", current, slot["blocked_until"])
                if slot["wait"] >= remaining():
                    raise FetchError("request_budget_exhausted", current)
                if slot["wait"]:
                    time.sleep(slot["wait"])
                remaining()
                reset_at = self.store.budget(rate_key, 5000 if kind != "web" else 200)
                if reset_at:
                    raise FetchError("daily_source_budget_exhausted", current, reset_at)
                if host == "api.search.brave.com" and not self.store.monthly_search_budget("brave", self.search_monthly_limit):
                    raise FetchError("external_search_monthly_budget_exhausted", current)
                try:
                    status, response_headers, body = self._wire(current, hdr)
                except FetchError as exc:
                    if exc.code == "network_error" and attempt < 2:
                        time.sleep(2 ** attempt + random.random())
                        continue
                    if exc.code == "network_error":
                        self.store.block(rate_key, time.time() + 300)
                    raise
                if status in (401, 403, 429) or status in (503, 502, 504) and response_headers.get("retry-after"):
                    delay = retry_delay(response_headers.get("retry-after"), 3600 if status in (401, 403) else 300)
                    until = time.time() + delay
                    self.store.block(rate_key, until)
                    raise FetchError(f"http_{status}", current, until)
                if status in (500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt + random.random())
                    continue
                break
            if status in (301, 302, 303, 307, 308):
                destination = canonical_url(urljoin(current, response_headers.get("location", "")))
                if urlsplit(current).scheme == "https" and urlsplit(destination).scheme != "https":
                    raise FetchError("https_downgrade", current)
                current = destination
                continue
            if status == 404:
                raise FetchError("http_404", current)
            if status != 200:
                if status >= 500:
                    self.store.block(rate_key, time.time() + 300)
                raise FetchError(f"http_{status}", current)
            content_type = response_headers.get("content-type", "").lower()
            if kind == "web" and not any(t in content_type for t in ("text/html", "text/plain", "application/xhtml+xml")):
                raise FetchError("unsupported_content_type", current)
            if response_headers.get("content-encoding", "identity") not in ("", "identity"):
                raise FetchError("unsupported_encoding", current)
            text = body.decode("utf-8", errors="replace")
            if kind == "web" and any(x in text.lower() for x in ("verify you are human", "cf-chl-", "unusual traffic", "captcha challenge")):
                until = time.time() + 3600
                self.store.block(rate_key, until)
                raise FetchError("bot_challenge", current, until)
            result = {"url": current, "fetched_at": time.time(), "text": text, "sha256": hashlib.sha256(body).hexdigest(), "content_type": content_type}
            self.store.cache(key, result, ttl)
            return result
        raise FetchError("redirect_limit", url)

    def _robots(self, url, refresh=False):
        p = urlsplit(url)
        robots_url = urlunsplit((p.scheme, p.netloc, "/robots.txt", "", ""))
        missing_key = "robots-missing:" + robots_url
        if self.store.cached(missing_key):
            return
        try:
            # Robots rules are cached even during contact revalidation.
            result = self.get(robots_url, robots=False, ttl=3600)
        except FetchError as exc:
            if exc.code == "http_404":
                self.store.cache(missing_key, {"missing": True}, 3600)
                return
            raise FetchError("robots_unavailable:" + exc.code, url, exc.retry_at) from None
        rules = RobotFileParser()
        rules.parse(result["text"].splitlines())
        if not rules.can_fetch(self.user_agent, url):
            raise FetchError("robots_disallowed", url)
        delay = rules.crawl_delay(self.user_agent) or rules.crawl_delay("*") or 0
        rate = rules.request_rate(self.user_agent) or rules.request_rate("*")
        spacing = max(delay, rate.seconds / rate.requests if rate and rate.requests else 0)
        if spacing:
            # Reserve an additional site-wide slot to honor a longer robots policy.
            slot = self.store.reserve("robots:" + publisher(url), spacing)
            if slot["blocked_until"]:
                raise FetchError("robots_rate_limited", url, slot["blocked_until"])
            if slot["wait"] >= remaining():
                raise FetchError("request_budget_exhausted", url)
            if slot["wait"]:
                time.sleep(slot["wait"])

    def json(self, url, *, token="", provider="ch", refresh=False, ttl=3600):
        headers = {"Authorization": "Basic " + base64.b64encode((token + ":").encode()).decode()} if provider == "ch" else {"X-Subscription-Token": token}
        response = self.get(url, kind="api", headers=headers, refresh=refresh, ttl=ttl)
        try:
            payload = json.loads(response["text"])
            if not isinstance(payload, dict):
                raise ValueError()
        except ValueError:
            raise FetchError("invalid_json", response["url"]) from None
        return payload, {k: v for k, v in response.items() if k != "text"}
