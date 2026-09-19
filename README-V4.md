# ChromeX Enrichment V4

V4 adds a strict UK company/contact verification service and a self-contained Ubuntu installer. It is an intentionally conservative successor to V3's verification pipeline: incomplete or conflicting evidence produces `needs_review`, never a degraded calling record.

**Status: implementation and offline regression tests complete; live business accuracy has not yet been measured.** API keys and a labelled pilot are needed before production rollout. This is not a claim of 100% accuracy. Public records can lag changes, websites can be stale, and matching a published telephone does not prove who will answer it.

## What changed from V3

The V3 source reviewed was commit `a0ff73b` (the installer is the authoritative source reviewed). Risks addressed:

- Name scoring could accept a Companies House candidate at 50 points; real directors could then be attached to a different business.
- The API requested only the first 100 officers. Officer/profile errors could look like empty successful results.
- `"director" in role` included corporate/nominee roles. Some web fallbacks inferred roles from an officer's whole appointments page rather than the target company appointment.
- Verifier failures could produce a degraded `/enrich-verified` response assembled from discovery results, including the supplied phone and unverified people.
- Search-engine HTML scraping and several crawling paths bypassed a shared request policy.

V4 replaces these decisions with explicit evidence rules. V3 and its service are preserved for comparison and rollback. V4 runs separately on port **8001** and does not install/change Ollama, GPU drivers or models. The verification decision is deterministic and does not depend on an LLM. Existing V3 AI extraction, address parsing, SMTP checks and GPU/model-control endpoints are not reimplemented in V4. Do not blend V3's unverified data into V4's calling output.

## Conditions for `call_ready: true`

All must pass:

1. Retrieve the exact Companies House company profile by company number; status must be `active`. Name search returns candidates and never selects a fuzzy match. Supplied names/postcodes must match the legal name/registered office or the result goes to review.
2. Confirm the website through a contextual disclosure of both the legal company name and the same registration number. Different numbers, ambiguous legal entities and unconfirmed websites go to review. A supplied domain is only a hint.
3. Read **all officer pages**. Changed totals, duplicate appointments, failed pages or invalid pagination prevent approval. Only current natural-person `director` appointments are eligible. Resigned, ceased, future, corporate, nominee, secretary and LLP-member roles are excluded. Pre-1992 appointments are handled separately.
4. Cross-reference at least one exact full director name and an adjacent director title on that confirmed website. News/blog/archive pages do not qualify. Former/retired/negated roles and duplicate names go to review. Initials, omitted middle names and nicknames are not automatically accepted. `director_name` can require a specific target.
5. Cross-check the business phone on that website and at least one approved external publisher whose page is tied to the same legal entity. The external page must show its exact company number and legal name, or the full legal name plus registered postcode plus a link to the confirmed official domain. Conflicting phones prevent approval. Multiple verified numbers require a selected `phone_number`.
6. All evidence used must have been fetched within 24 hours. `/call-check/{record_id}` bypasses company, officer and page caches and repeats the checks immediately before calling. Source cooldowns and robots rules still apply.

This normally means three publishers overall: Companies House, the company website and a corroborating external publisher. Director evidence specifically uses the registry and company website; external phone corroboration is a separate check. The system does not claim three independent confirmations of every field.

`registered_directors` contains current appointments for the candidate legal entity, including people lacking website corroboration. **Never use that list for automatic calling.** `directors` contains people that passed the website cross-check. Only `call_contact` is the guarded calling payload, and it is `null` unless the whole record is ready. A business switchboard is not labelled as a director's personal number. Birth dates and residential addresses are not copied into calling records. Published Companies House identity-verification dates, when available, are reported separately from the appointment decision.

Emails are labelled `published_on_official_website`, with deliverability `unknown`. No invented addresses, assumed personal emails, SMTP probing or unsupported deliverability claims.

## Cross-reference sources and request limits

Default external publisher groups: CQC (`cqc.org.uk`), NHS (`nhs.uk`), Yell (`yell.com`) and Thomson Local (`thomsonlocal.com`). These are candidates for contact corroboration, not guarantees that every page is accessible or contains sufficient evidence. Their `robots.txt`, source policies and response status still determine access. Sector-specific coverage varies; many businesses will correctly require review.

Configure `SOURCE_GROUPS_JSON` as a domain-to-ownership-group mapping after reviewing source permissions, ownership and suitability. Use one group for commonly owned domains. Registry republishers must not be promoted as independent verification of directors. Request callers cannot add trusted sources; they can supply candidate `corroboration_urls` for configured publishers only. Same-site pages and identical normalized page text do not multiply corroboration. These checks cannot establish every source's underlying data independence or catch every copied statement.

Requests use:

- Companies House API, with at least one second between requests, below its published 600 requests per five minutes limit for this installation. A key used by other applications still needs a shared upstream quota plan.
- Brave Search API for discovery; search snippets are never accepted as facts. Search spacing defaults to 1.1 seconds and can be increased with `SEARCH_SPACING_SECONDS` for your plan. Obtain a plan permitting your enrichment use and any retained results.
- Website requests at least three seconds apart per registrable publisher, honoring longer robots delays/request rates. No browser impersonation, CAPTCHA solving, proxy rotation or block bypass.
- SQLite-backed cooldowns shared by threads and surviving restarts. HTTP 401/403/429 and challenge pages open a cooldown. `Retry-After` seconds/date values are honored; 429 is not immediately retried.
- At most two retries for transient network/5xx failures, exponential delays and jitter, followed by a cooldown on exhausted server/network failures.
- 200 website requests per publisher per UTC day, including robots requests; 5,000 API requests per provider per day. These are conservative ceilings, not guaranteed provider allowances. Four concurrent jobs, at most 18 website fetch attempts per company and a 180-second company request budget.
- Cached successful registry results for one hour, web pages for six hours and robots rules for one hour. Expired cache entries are never used as fallback evidence. A refresh can still be blocked by rate limits and return review.
- Public-only DNS validation and connection pinning; private/link-local addresses, unsafe ports, credential-bearing URLs, HTTPS downgrades and cross-host authenticated API redirects are blocked. Response bytes and read duration are bounded.

No implementation can guarantee that an IP will never be restricted. If a source denies access, the record reports the failure and remains unapproved where required evidence is missing.

Official provider documentation: [Companies House limits](https://developer.company-information.service.gov.uk/developer-guidelines/), [officer endpoint](https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/reference/officers/list), [officer roles and appointment fields](https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/resources/officerlist?v=latest), [Brave Search API](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started).

## Install

Ubuntu 24.04+ with Python 3.12+, root and systemd. Review the script first. The installer embeds all V4 runtime source, the pinned dependency lock and this guide. It does not download application code from a moving branch.

```bash
# Offline source extraction; no system changes or network access:
bash EnrichmentV4.sh --extract ./v4-review

# Isolated installation:
sudo bash EnrichmentV4.sh
sudoedit /etc/chromex-enrichment-v4.env
sudo systemctl restart chromex-enrichment-v4
```

Set `COMPANIES_HOUSE_API_KEY` and `BRAVE_SEARCH_API_KEY` in that root-only environment file. A random `ENRICHMENT_API_KEY` is generated on first install; keep it for the CRM's authenticated requests. Keys are not printed. Existing configuration is preserved during upgrades; update it explicitly with `sudoedit`.

A Companies House key is required for verification. A Brave key enables automatic website discovery/corroboration and sector discovery. Without Brave, an explicit `domain`, legal company identity and sufficient `corroboration_urls` can still be verified. Without Companies House, records cannot pass the strict gate.

Service: `chromex-enrichment-v4.service`. Application: `/opt/chromex-enrichment-v4/current`. State: `/var/lib/chromex-enrichment-v4/enrichment.sqlite3`. Logs: `journalctl -u chromex-enrichment-v4`. It runs as the dedicated `chromex-enrichment` user, binds loopback only and requires an API key for data endpoints. Put your authenticated TLS reverse proxy in front if clients are remote. The installer stages versioned releases and restores the previous V4 release/service on failed startup. It does not stop V3 or kill unrelated port owners.

The SQLite database contains cached source responses and audit evidence, including public registry data. Its directory is restricted to the service user. Back it up securely along with the configuration file. Audit records are retained for 30 days, pruned at startup and during jobs; expired cache entries are pruned at the same time. Backup retention must be managed separately.

## API and calling integration

Send `X-API-Key: <your ENRICHMENT_API_KEY>` or an equivalent Bearer token. `/health` is public. `/docs` describes the V4 schema.

```json
POST /enrich-verified
{
  "company_number": "01234567",
  "company_name": "EXAMPLE LIMITED",
  "domain": "https://example.co.uk",
  "phone_number": "020 7946 0958",
  "director_name": "Jane Example",
  "corroboration_urls": ["https://cqc.org.uk/location/example"]
}
```

This is an illustrative request, not a real verified business. Company name, website, director and phone must actually agree with the sources. A trading name or trading postcode that differs from the registry will require review; do not work around that by assuming two legal entities are equivalent.

Response envelope:

```json
{
  "schema_version": "4.0",
  "status": "needs_review",
  "call_ready": false,
  "degraded": false,
  "verified_record": {
    "record_id": "...",
    "blocking_reasons": ["supplied_phone_not_cross_verified"],
    "call_contact": null,
    "field_evidence": {},
    "source_errors": []
  }
}
```

| Route | Purpose |
|---|---|
| `POST /enrich-verified` or `/enrich` | Strict single-company check |
| `POST /batch-enrich` | `{"items": [<enrich request>, ...]}`, at most 20; sequential bounded processing |
| `POST /discover-businesses` or `/crawl-businesses` | `{"query":"care homes", "location":"Hampshire", "max_results":5, "offset":0, "enrich":true}`; deduplicated website candidates, optionally verified individually |
| `GET /records/{record_id}` | Retrieve audit evidence; expiry removes calling approval |
| `POST /call-check/{record_id}` | Fresh repeat verification before a call; creates a new audit record |
| `GET /system-info` | Configured providers, policy and approved publisher groups, without keys |

The caller must:

1. Keep discovery and review results outside the automatic calling queue.
2. Immediately before dialing, call `/call-check/{record_id}`.
3. Require successful HTTP status, `call_ready === true`, and a non-null `verified_record.call_contact` from that **new response**.
4. Use only its `business_phone` and `verified_director_names`. Do not keep dialing an older ready response after an error or review result.
5. Keep your existing suppression, do-not-call and marketing eligibility checks. `call_ready` is data quality only.

**Schema migration is required.** V4 keeps several familiar route names and the `/enrich-verified` envelope, but is not a drop-in V3 response replacement. Field confidence percentages are removed; evidence and explicit check states replace them. `company`/`phone` remain accepted aliases for `company_name`/`phone_number`. Legacy model/quality/ScrapeGraph/fallback flags return validation errors rather than weakening the gate. V4 is scoped to UK registered companies; sole traders, non-UK entities and unsupported role types remain review work.

## Validation and rollout

Run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-v4-dev.txt
.venv/bin/python -m pytest -q
python3 scripts/build_v4_installer.py
bash -n EnrichmentV4.sh
bash EnrichmentV4.sh --extract /tmp/chromex-v4-review
```

Regression coverage includes positive verified records; wrong/ambiguous company names; wrong website/phone; renamed and resigned people; corporate/nominee directors; duplicate names; officer pagination and partial failures; stale evidence; shared publishers and copied text; missing keys; rate limits; robots denial; private DNS; authentication; and a director resigning between enrichment and pre-call refresh. Tests use fixtures and mocked HTTP, not real registry credentials. The root/systemd installation and real source coverage still need staging validation on the target Ubuntu host.

Before replacing the live queue, run V4 in shadow mode on a labelled sample of at least 100 businesses that includes known wrong-company, old-director, shared-office, trading-name, multi-site and changed-phone cases. Independently check legal entity, current director and business number. Record ready/review coverage, false approvals by field, source failures, request cost and latency. Investigate every false approval. Keep the conservative rules unless that evidence supports a reviewed change. This is how accuracy should be established; a high score from an AI model is not a measurement.
