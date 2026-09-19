"""CSV candidate import. Local files only; rows never become verified facts."""
from __future__ import annotations
import argparse
import csv
import http.client
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from .http import canonical_url, FetchError


def header_key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def csv_candidates(path, url_column=None, title_column=None, source_url_column=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = {header_key(k): k for k in reader.fieldnames or []}

        def choose(explicit, aliases, required=False):
            if explicit:
                if explicit not in (reader.fieldnames or []):
                    raise ValueError("Missing CSV column: " + explicit)
                return explicit
            match = next((names[k] for k in aliases if k in names), None)
            if required and not match:
                raise ValueError("Specify --url-column; no website column was recognised")
            return match

        url_key = choose(url_column, ["website", "locationwebsite", "webaddress", "websiteurl", "url", "domain"], True)
        title_key = choose(title_column, ["companyname", "locationname", "name", "providername", "title"])
        reference_key = choose(source_url_column, ["cqcurl", "cqcwebpage", "sourceurl", "referenceurl"])
        text_keys = [v for k, v in names.items() if k in {"companyname", "locationname", "providername", "companynumber", "companyregistrationnumber", "postcode", "locationpostcode", "city", "town", "locationcity", "county", "region", "location", "sector", "industry", "servicetype", "servicetypes", "sic", "siccodes", "phone", "phonenumber", "telephone"}]
        for line, row in enumerate(reader, 2):
            title = (row.get(title_key) or "")[:500] if title_key else ""
            text = " ".join(str(row.get(k) or "") for k in text_keys)[:12000]
            for key in dict.fromkeys(k for k in (url_key, reference_key) if k):
                value = (row.get(key) or "").strip()
                if not value:
                    continue
                try:
                    url = canonical_url(value)
                except FetchError:
                    print(f"Skipped invalid website at CSV line {line}", file=sys.stderr)
                    continue
                yield {"url": url, "title": title, "text": text}


def send_batch(endpoint, api_key, items, source):
    p = urlsplit(endpoint)
    # Never send a credential to plaintext remote HTTP or follow redirects.
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment:
        raise ValueError("Expected API origin URL")
    if p.scheme == "http" and p.hostname != "127.0.0.1":
        raise ValueError("Remote API imports require HTTPS")
    cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
    conn = cls(p.hostname, p.port, timeout=30)
    try:
        conn.request("POST", "/catalogue/import", body=json.dumps({"items": items, "source": source}), headers={"X-API-Key": api_key, "Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read(100000)
        if response.status != 200:
            raise RuntimeError("Import rejected: HTTP " + str(response.status))
        data = json.loads(raw)
        if data.get("indexed") != len(items):
            raise RuntimeError("Import count did not match submitted batch")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Import business URL candidates from CSV; no automatic calling approval")
    parser.add_argument("csv_file")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--url-column")
    parser.add_argument("--title-column")
    parser.add_argument("--source-url-column")
    parser.add_argument("--source", default="operator_csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count URL rows without API calls")
    args = parser.parse_args()
    key = os.getenv("ENRICHMENT_API_KEY", "")
    if not args.dry_run and len(key) < 24:
        parser.error("Set ENRICHMENT_API_KEY in the environment")
    total, batch = 0, []
    try:
        for candidate in csv_candidates(args.csv_file, args.url_column, args.title_column, args.source_url_column):
            total += 1
            if not args.dry_run:
                batch.append(candidate)
                if len(batch) == 250:
                    send_batch(args.api, key, batch, args.source)
                    batch.clear()
        if batch:
            send_batch(args.api, key, batch, args.source)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"candidate_url_rows": total, "dry_run": args.dry_run, "call_ready": False}))


if __name__ == "__main__":
    main()
