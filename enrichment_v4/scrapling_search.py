"""Optional browser discovery. Rendered search snippets are never evidence."""
from __future__ import annotations
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from bs4 import BeautifulSoup
from .http import FetchError, MAX_BYTES, canonical_url, publisher, remaining, retry_delay

ENGINES = {"google": "https://www.google.com/search", "bing": "https://www.bing.com/search",
           "duckduckgo": "https://html.duckduckgo.com/html/"}
_BROWSER_SLOT = threading.BoundedSemaphore(1)


def result_url(href, engine):
    """Unwrap known tracking links without visiting them or guessing a URL."""
    url = urljoin(ENGINES[engine], href)
    p = urlsplit(url)
    params = parse_qs(p.query)
    if p.hostname in ("www.google.com", "google.com") and p.path == "/url":
        url = (params.get("q") or params.get("url") or [""])[0]
    elif p.hostname in ("duckduckgo.com", "html.duckduckgo.com") and "uddg" in params:
        url = params["uddg"][0]
    elif p.hostname == "www.bing.com" and p.path == "/ck/a":
        value = params.get("u", [""])[0]
        if not value.startswith("a1"):
            raise FetchError("unrecognised_search_link")
        try:
            value = value[2:]
            url = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
        except (ValueError, UnicodeError):
            raise FetchError("unrecognised_search_link") from None
    # Explicit schemes prevent malformed/relative redirect values becoming hosts.
    if not url.startswith(("https://", "http://")):
        raise FetchError("unrecognised_search_link")
    url = canonical_url(url)
    host = urlsplit(url).hostname
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise FetchError("numeric_search_link")
    if host.endswith((".local", ".internal", ".localhost")) or publisher(url) in {"google.com", "bing.com", "duckduckgo.com", "microsoft.com", "gstatic.com"}:
        raise FetchError("non_business_search_link")
    return url


def parse_results(html, engine):
    soup = BeautifulSoup(html, "html.parser")
    rows, seen = [], set()
    if engine == "bing":
        links = soup.select("li.b_algo h2 a[href]")
    elif engine == "duckduckgo":
        links = soup.select("a.result__a[href]")
    else:
        # An organic heading inside a link; never all links on a SERP.
        links = [h.find_parent("a", href=True) for h in soup.select("#search h3")]
    for link in links[:100]:
        if not link:
            continue
        try:
            url = result_url(link["href"], engine)
        except (FetchError, ValueError):
            continue
        title = link.get_text(" ", strip=True)[:500]
        if not title or url in seen:
            continue
        seen.add(url)
        rows.append({"url": url, "title": title, "snippet": "", "status": "discovered_unverified"})
    if not rows:
        # Consent pages, DOM changes and empty searches require explicit review;
        # never interpret them as proof that a business does not exist.
        raise FetchError("scrapling_no_recognised_results")
    return rows[:20]


class ScraplingSearch:
    name = "scrapling"
    configured = True

    def __init__(self, store, python="/opt/chromex-enrichment-v4/scrapling/bin/python", engine="bing", daily_limit=25, solve_cloudflare=False):
        if engine not in ENGINES:
            raise ValueError("SCRAPLING_ENGINE must be bing, google or duckduckgo")
        if not python or not Path(python).is_absolute():
            raise ValueError("SCRAPLING_PYTHON must be the absolute path to the optional browser venv Python")
        self.store, self.python, self.engine = store, python, engine
        self.daily_limit, self.solve_cloudflare = max(0, int(daily_limit)), solve_cloudflare

    def _render(self, url):
        worker = Path(__file__).with_name("scrapling_worker.py")
        # Never give the browser API keys, proxy settings, the application's HOME,
        # or its database path. A new temporary browser profile per query.
        with tempfile.TemporaryDirectory(prefix="chromex-browser-") as home:
            env = {"PATH": "/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8"}
            env["PLAYWRIGHT_BROWSERS_PATH"] = str(Path(self.python).parent.parent / "browsers")
            payload = json.dumps({"url": url, "engine": self.engine, "solve_cloudflare": self.solve_cloudflare})
            with tempfile.TemporaryFile() as output:
                try:
                    proc = subprocess.Popen([self.python, "-I", str(worker)], stdin=subprocess.PIPE, stdout=output,
                                            stderr=subprocess.DEVNULL, env=env, start_new_session=True)
                except OSError:
                    raise FetchError("scrapling_not_installed") from None
                try:
                    proc.communicate(payload.encode(), timeout=min(75, remaining()))
                except subprocess.TimeoutExpired:
                    raise FetchError("scrapling_timeout") from None
                finally:
                    # Reap the worker AND its Chromium descendants on every path.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                output.seek(0)
                raw = output.read(4 * MAX_BYTES + 1)
                if len(raw) > 4 * MAX_BYTES:
                    raise FetchError("scrapling_response_too_large")
                try:
                    result = json.loads(raw)
                    if not isinstance(result, dict):
                        raise ValueError()
                except (ValueError, TypeError):
                    raise FetchError("scrapling_worker_failed") from None
                if result.get("error"):
                    raise FetchError(str(result["error"])[:100])
                if proc.returncode != 0:
                    raise FetchError("scrapling_worker_failed")
                return result

    def search(self, query, count=10, offset=0, refresh=False):
        remaining()
        # One organic-results page per query; no automatic next-page scraping.
        if offset:
            return [], {"provider": self.name, "engine": self.engine, "notes": ["Browser discovery supports the first results page only."]}
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            raise FetchError("scrapling_invalid_query")
        query = " ".join(query.split())
        url = ENGINES[self.engine] + "?" + urlencode({"q": query})
        key = "scrapling:" + hashlib.sha256((url + str(self.solve_cloudflare)).encode()).hexdigest()
        if not refresh:
            cached = self.store.cached(key)
            if cached:
                return cached["rows"][:count], cached["evidence"]
        if not _BROWSER_SLOT.acquire(blocking=False):
            raise FetchError("scrapling_busy")
        rate_key = "scrapling-search"  # Shared across engines; changing engine does not bypass a cooldown.
        try:
            slot = self.store.reserve(rate_key, 30.0)
            if slot["blocked_until"]:
                raise FetchError("scrapling_cooldown", retry_at=slot["blocked_until"])
            if slot["wait"] >= remaining():
                raise FetchError("request_budget_exhausted")
            if slot["wait"]:
                time.sleep(slot["wait"])
            reset_at = self.store.budget(rate_key, self.daily_limit)
            if reset_at:
                raise FetchError("scrapling_daily_budget_exhausted", retry_at=reset_at)
            try:
                response = self._render(url)
                status = response.get("status")
                if status != 200:
                    until = time.time() + retry_delay(response.get("retry_after"), 3600)
                    self.store.block(rate_key, until)
                    raise FetchError("scrapling_http_" + str(status), retry_at=until)
                html = response.get("html", "")
                if not isinstance(html, str) or len(html.encode()) > MAX_BYTES:
                    raise FetchError("scrapling_response_too_large")
                final = urlsplit(response.get("url", ""))
                if final.hostname != urlsplit(url).hostname or final.path.rstrip("/") != urlsplit(url).path.rstrip("/"):
                    raise FetchError("scrapling_consent_or_redirect")
                visible = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
                if any(text in visible for text in ("unusual traffic", "verify you are human", "captcha challenge", "confirm you are human", "before you continue to google", "one last step")):
                    raise FetchError("scrapling_bot_challenge")
                rows = parse_results(html, self.engine)
            except FetchError as exc:
                until = exc.retry_at or time.time() + 3600
                self.store.block(rate_key, until)
                raise FetchError(exc.code, retry_at=until) from None
            evidence = {"provider": self.name, "engine": self.engine, "url": url, "fetched_at": time.time(),
                        "notes": ["Browser search candidates only. No live CAPTCHA success or contact accuracy guarantee."]}
            self.store.cache(key, {"rows": rows, "evidence": evidence}, 86400)
            return rows[:count], evidence
        finally:
            _BROWSER_SLOT.release()
