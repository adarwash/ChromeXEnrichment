"""Short-lived, secret-free Scrapling subprocess. Only fixed search hosts may load."""
from __future__ import annotations
import contextlib
import ipaddress
import json
import re
import resource
import socket
import sys
from urllib.parse import urlsplit

HOSTS = {
    "bing": {"www.bing.com", "bing.com", "r.bing.com", "th.bing.com"},
    "google": {"www.google.com", "google.com", "www.gstatic.com", "ssl.gstatic.com", "consent.google.com"},
    "duckduckgo": {"html.duckduckgo.com", "duckduckgo.com", "links.duckduckgo.com"},
}
ROOTS = {"bing": ("www.bing.com", "/search"), "google": ("www.google.com", "/search"), "duckduckgo": ("html.duckduckgo.com", "/html/")}
PROFILES = {
    "chrome_windows": {"ua_platform": "Windows NT 10.0; Win64; x64", "navigator_platform": "Win32", "platform": "Windows", "platform_version": "15.0.0", "architecture": "x86", "width": 1920, "height": 1080, "scale": 1},
    "chrome_macos": {"ua_platform": "Macintosh; Intel Mac OS X 10_15_7", "navigator_platform": "MacIntel", "platform": "macOS", "platform_version": "14.0.0", "architecture": "arm", "width": 1440, "height": 900, "scale": 2},
    "chrome_linux": {"ua_platform": "X11; Linux x86_64", "navigator_platform": "Linux x86_64", "platform": "Linux", "platform_version": "", "architecture": "x86", "width": 1366, "height": 768, "scale": 1},
}


def profile_identity(profile, browser_version):
    """Keep UA and client hints tied to the installed Chromium version."""
    config = PROFILES[profile]
    match = re.fullmatch(r"(?:HeadlessChrome|Chrome|Chromium)/(\d+\.\d+\.\d+\.\d+)", browser_version)
    if not match:
        raise ValueError("scrapling_unknown_browser_version")
    version = match[1]
    major = version.split('.')[0]
    return {
        "userAgent": f"Mozilla/5.0 ({config['ua_platform']}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36",
        "platform": config["navigator_platform"],
        "acceptLanguage": "en-GB,en;q=0.9",
        "userAgentMetadata": {
            "brands": [{"brand": "Chromium", "version": major}, {"brand": "Google Chrome", "version": major}],
            "fullVersionList": [{"brand": "Chromium", "version": version}, {"brand": "Google Chrome", "version": version}],
            "fullVersion": version, "platform": config["platform"], "platformVersion": config["platform_version"],
            "architecture": config["architecture"], "bitness": "64", "model": "", "mobile": False, "wow64": False,
        },
    }


def pinned_hosts(engine, solve_cloudflare=False):
    rules, allowed = [], set()
    hosts = HOSTS[engine] | ({"challenges.cloudflare.com"} if solve_cloudflare else set())
    for host in sorted(hosts):
        try:
            addresses = {r[4][0] for r in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        except OSError:
            continue
        if not addresses or any(not ipaddress.ip_address(a).is_global or ipaddress.ip_address(a).is_multicast or ipaddress.ip_address(a).is_reserved for a in addresses):
            continue
        address = sorted(addresses, key=lambda a: (":" in a, a))[0]
        rules.append("MAP " + host + " " + ("[" + address + "]" if ":" in address else address))
        allowed.add(host)
    if ROOTS[engine][0] not in allowed:
        raise ValueError("scrapling_search_dns_unavailable")
    # Chromium connects to these validated numeric answers, without resolving
    # them again. Unknown hosts cannot resolve, including browser background DNS.
    return ", ".join(rules + ["MAP * ~NOTFOUND"]), allowed


def allowed_request(url, hosts):
    try:
        p = urlsplit(url)
        return p.scheme == "https" and p.hostname in hosts and p.port in (None, 443) and not p.username and not p.password and "\\" not in url
    except ValueError:
        return False


def browser_setup(allowed, profile="native"):
    def setup(page):
        # Scrapling catches ordinary setup errors and continues; close the
        # context before rethrowing so a failed guard cannot permit navigation.
        try:
            count = 0

            def route_request(route):
                nonlocal count
                count += 1
                request = route.request
                if count > 80 or request.method not in ("GET", "POST") or not allowed_request(request.url, allowed):
                    route.abort()
                else:
                    route.continue_()

            page.context.route("**/*", route_request)
            page.context.route_web_socket("**/*", lambda ws: ws.close())
            if profile != "native":
                session = page.context.new_cdp_session(page)
                version = session.send("Browser.getVersion")["product"]
                session.send("Emulation.setUserAgentOverride", profile_identity(profile, version))
        except Exception:
            page.context.close()
            raise
    return setup


def render(payload):
    engine, url = payload["engine"], payload["url"]
    p = urlsplit(url)
    if engine not in ROOTS or (p.hostname, p.path) != ROOTS[engine] or not allowed_request(url, HOSTS[engine]):
        raise ValueError("scrapling_invalid_search_url")
    solve = payload.get("solve_cloudflare") is True
    profile = payload.get("browser_profile", "native")
    if profile != "native" and profile not in PROFILES:
        raise ValueError("scrapling_invalid_browser_profile")
    rules, allowed = pinned_hosts(engine, solve)
    context_options = {"service_workers": "block", "accept_downloads": False, "chromium_sandbox": True}
    profile_options = {}
    if profile != "native":
        config = PROFILES[profile]
        dimensions = {k: config[k] for k in ("width", "height")}
        context_options.update(viewport=dimensions, screen=dict(dimensions), device_scale_factor=config["scale"], is_mobile=False, has_touch=False)
        profile_options = {"locale": "en-GB", "timezone_id": "Europe/London"}
    from scrapling.fetchers import StealthyFetcher
    response = StealthyFetcher.fetch(
        url, headless=True, timeout=60000, retries=1, google_search=False,
        solve_cloudflare=solve, block_webrtc=True, network_idle=False,
        page_setup=browser_setup(allowed, profile),
        extra_flags=["--host-resolver-rules=" + rules, "--no-proxy-server", "--disable-quic",
                     "--disable-background-networking", "--dns-prefetch-disable"],
        additional_args=context_options, **profile_options,
    )
    html = response.html_content
    if len(html.encode()) > 2_000_000:
        raise ValueError("scrapling_response_too_large")
    return {"url": response.url, "status": response.status, "html": html, "browser_profile": profile,
            "retry_after": next((v for k, v in response.headers.items() if k.lower() == "retry-after"), None)}


def main():
    resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))
    try:
        payload = json.loads(sys.stdin.read(4096))
        with contextlib.redirect_stdout(sys.stderr):
            result = render(payload)
    except ImportError:
        result = {"error": "scrapling_not_installed"}
    except ValueError as exc:
        result = {"error": str(exc) if str(exc).startswith("scrapling_") else "scrapling_worker_failed"}
    except Exception:
        result = {"error": "scrapling_browser_failed"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
