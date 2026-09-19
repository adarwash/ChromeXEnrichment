# Zero-budget search for ChromeX Enrichment V4.2

V4.2 runs without a paid search subscription. It can provide a SerpApi-style candidate-search endpoint for your own business catalogue and, optionally, a self-hosted SearXNG search companion or a local Scrapling headless browser. It does not create an index comparable to Google's for free. Searches find possible pages; the existing company/director/phone checks decide whether those pages support a calling record.

## Recommended setup

Use your existing Linux server, the Companies House public API, direct website fetching and a local SQLite search catalogue. Import the website links you already own in CRM exports or permitted public datasets. Companies House is used for legal entities and officer appointments, not assumed to contain business phone numbers or websites. Enrichment searches the local catalogue and directly fetches candidate pages for verification. This has no search-provider fee; server electricity, bandwidth and maintenance still exist.

For new web discoveries beyond the imported catalogue, opt into the included SearXNG companion. It makes ordinary metasearch requests to configured engines. It can be rate-limited or blocked and has no guaranteed throughput. Local candidates remain usable if upstream search fails. Stop or defer unavailable sources; do not rotate public instances, proxies or accounts to evade a block.

The default configuration is:

```ini
SEARCH_PROVIDER="local"
EXTERNAL_SEARCH_MONTHLY_LIMIT="0"
SEARXNG_URL="http://127.0.0.1:8888"
SEARXNG_DAILY_LIMIT="100"
```

An existing Brave key does **not** activate Brave. `SEARCH_PROVIDER=brave` is explicit, and outgoing Brave HTTP requests also require a positive `EXTERNAL_SEARCH_MONTHLY_LIMIT`. The counter is persistent, shared by threads and counts retries and redirects as outgoing requests. Cached hits do not consume it. Zero permits no new Brave requests. This is an application request cap, not a provider billing control: free credits, billing-cycle dates and other applications using the same account must be checked at the provider. No service is subscribed to or charged by the installer.

## Import free candidate data

`POST /catalogue/import` accepts up to 500 URL records, authenticated with the existing enrichment API key:

```json
{
  "source": "our_crm_export",
  "items": [
    {
      "url": "https://example.co.uk",
      "title": "Example Care Limited",
      "text": "care home Hampshire PO15 7AD company 01234567"
    },
    {
      "url": "https://cqc.org.uk/location/example",
      "title": "Example Care Limited",
      "text": "care home Hampshire PO15 7AD"
    }
  ]
}
```

The examples are fictional. Imported names/numbers/phones are only search hints, never verification evidence. CSV uploads do not clear contacts for calling. Duplicate URLs update the existing candidate; candidates expire after 90 days unless reimported. Previously verified links can be reused as candidates but must pass normal verification again. Imported sector/location metadata is preserved when the same URL is rediscovered.

To import CSV on the installed server:

```bash
cd /opt/chromex-enrichment-v4/current
# Set ENRICHMENT_API_KEY securely in this shell to match the service.
# Check recognised columns first; this makes no network requests:
venv/bin/python -m enrichment_v4.catalogue /path/to/businesses.csv --dry-run

# Import after inspecting the dry-run count:
venv/bin/python -m enrichment_v4.catalogue /path/to/businesses.csv --source our_crm
```

Recognised website columns include `Website`, `Location Website`, `Web Address`, `URL` and `Domain`; matching ignores case/spaces. Company/location names, postcode, county, sector and related fields are indexed. A `CQC URL`, `Source URL` or `Reference URL` column can add the corroborating page as a separate candidate. For other schemas specify `--url-column`, `--title-column` and `--source-url-column` using the exact CSV headers. The importer sends batches of 250 to the API, reads keys from the environment and refuses remote plaintext HTTP. It does not automatically download data or follow API redirects.

`GET /catalogue/status` shows candidate URL count and optional paid-provider HTTP usage. `/discover-businesses` supports the same sector/location request as before. In local mode it searches the imported catalogue only; an empty catalogue returns no new businesses. Full-text matching requires the discovery terms and uses stemming, e.g. `care homes` can match `care home`. Directory pages remain corroborating candidates rather than being treated as official company sites.

### Care-sector source

[CQC publishes downloadable care-directory data](https://www.cqc.org.uk/about-us/transparency/using-cqc-data), including website links, under the Open Government Licence. Download the current CSV and map its actual headers with the importer. Keep CQC attribution when reusing its data. Website links from an official list are useful discovery hints, but a care-location name, registered manager and legal company director are different concepts. The current page also reports delays in some directory updates; always do live checks before calling. The live CQC CSV was not imported or validated in this development environment.

## Optional self-hosted search

An existing Docker and Compose installation is required. From a checkout or extracted installer payload:

```bash
bash deploy/searxng/setup.sh
```

The helper pulls the official SearXNG image, records its resolved digest and generates a secret in a local `.env`. It starts the companion bound to **127.0.0.1:8888**. Repeat runs reuse that digest; upgrades require reviewing and changing it. The provided settings enable only Bing and DuckDuckGo, JSON output, upstream suspension and bounded timeouts. There is no paid API configuration, Google CAPTCHA solver, stealth extension or proxy rotation. Configure additional engines only after reviewing their operation and access policies.

Then edit `/etc/chromex-enrichment-v4.env`:

```ini
SEARCH_PROVIDER="searxng"
SEARXNG_URL="http://127.0.0.1:8888"
SEARXNG_DAILY_LIMIT="100"
EXTERNAL_SEARCH_MONTHLY_LIMIT="0"
```

Restart V4. Its adapter makes at most one SearXNG query every ten seconds and at most 100 uncached queries per UTC day by default. Each SearXNG query can result in a request to each configured engine, so 100 is not an upstream request count. Results are cached for 24 hours as **candidates only**. A failed engine puts the adapter into a persistent cooldown; it does not continuously hammer other engines. Returned website links still pass through the public-only V4 fetcher. Only the fixed loopback SearXNG origin is permitted; API callers cannot choose an arbitrary internal server or redirect.

Source: [SearXNG overview](https://docs.searxng.org/user/about.html), [JSON search API](https://docs.searxng.org/dev/search_api.html), [container setup](https://docs.searxng.org/admin/installation-docker.html), [engine selection](https://docs.searxng.org/admin/settings/settings.html), [upstream suspension](https://docs.searxng.org/admin/settings/settings_search.html). This companion uses free software; upstream availability is not guaranteed. It was not launched here because this environment has no Docker runtime.

## Optional Scrapling browser discovery

[Scrapling](https://github.com/D4Vinci/Scrapling) is free, BSD-3-Clause software. This integration pins Scrapling **0.4.15** and its Python dependencies in a separate environment. No paid search API, solver subscription or proxy account is required. The normal V4 install stays lightweight; installing this option downloads Chromium and its operating-system dependencies.

On your Ubuntu server, from the extracted source or installed release:

```bash
cd /opt/chromex-enrichment-v4/current
sudo bash deploy/scrapling/setup.sh
```

Then edit `/etc/chromex-enrichment-v4.env` and restart V4:

```ini
SEARCH_PROVIDER="scrapling"
SCRAPLING_PYTHON="/opt/chromex-enrichment-v4/scrapling/bin/python"
SCRAPLING_ENGINE="bing"
SCRAPLING_DAILY_LIMIT="25"
SCRAPLING_SOLVE_CLOUDFLARE="0"
EXTERNAL_SEARCH_MONTHLY_LIMIT="0"
```

```bash
sudo systemctl restart chromex-enrichment-v4
```

Set `SCRAPLING_ENGINE` to `google`, `bing` or `duckduckgo`. This is an operator setting; callers cannot choose engines or arbitrary browser URLs. Your existing `/discover-businesses` and enrichment requests use the new provider automatically. The local catalogue is still searched and remains usable during upstream failures. Switch `SEARCH_PROVIDER` back to `local` to disable browser discovery. No changes to the calling schema are required.

The browser runs on your Linux machine using [StealthyFetcher](https://scrapling.readthedocs.io/en/latest/fetching/stealthy.html). Its optional `solve_cloudflare` handler is enabled only by `SCRAPLING_SOLVE_CLOUDFLARE="1"`. It is a Cloudflare-specific capability, **not a demonstrated Google reCAPTCHA solver or an IP-block guarantee**. The `google_search` library parameter merely sets a Referer; our adapter explicitly constructs the search URL and disables that parameter.

Operational bounds:

- One selected engine, one organic-results page per query; no automatic engine switching, next-page scraping or paid fallback.
- One browser process per V4 worker, a 75-second total process deadline, at least 30 seconds between uncached attempts, and 25 attempts per UTC day by default. Busy/closely spaced work returns an explicit error for later retry. A query may load up to 80 browser subrequests; 25 queries does not mean 25 network requests.
- A 24-hour candidate cache. Persisted budget and cooldown state survives restarts and is shared across engine choices. Unresolved challenges, consent redirects, unknown page layouts and browser failures open a one-hour cooldown; longer `Retry-After` values are honored.
- Public DNS answers for a small fixed engine/asset host list are pinned into Chromium's resolver rules. Other hosts, unsafe schemes/ports, service workers, WebSockets and downloads are blocked. Business links are parsed but never visited by the search browser.
- A fresh temporary profile and a minimal environment without API keys or inherited proxy configuration. Chromium's sandbox stays enabled; run V4 as its dedicated non-root user on a host supporting Chromium sandboxing. If the sandbox cannot start, the attempt fails; the adapter does not disable it.
- Registry and business-source verification still use V4's separate DNS-pinned HTTP fetcher and robots policy. Search-engine browser requests use the fixed provider adapter and do not inherit that fetcher's robots rules; review the selected engine's access terms before enabling.

Only recognised organic result links and titles are returned, marked `discovered_unverified`. Search snippets, adaptive selector guesses and CAPTCHA pages never establish directors or phone ownership. Browser discovery does not render blocked business websites for verification; insufficient source evidence still means `needs_review`.

The Python dependency environment and offline adapter tests were checked here. The Chromium download timed out in this environment, so browser launch, live engine layouts and challenge success remain unvalidated. After installation on the target server, first try one known-business query, inspect source errors and check the returned company manually before increasing volume. No claim of 20,000 reliable free searches is made.

## Existing V3 headless-browser/CAPTCHA option

V3 already includes Crawl4AI, Playwright/Chromium and ScrapeGraphAI. The actual `fetch_html_with_crawl4ai()` helper starts a headless browser. Its configuration is plain `BrowserConfig(headless=True, verbose=False)` plus a page timeout; no explicit CAPTCHA-solving service or solver is configured in the repository examined.

A concrete wiring bug was found: `_search_with_crawl4ai()` claimed to render search pages in Chromium but called `fetch_html_from_url()`, the static HTTP helper. That call is now corrected to the actual renderer in **both** `EnrichmentV3.sh` and `main.py`. A regression test verifies the search function invokes the renderer for its Bing/DDG fallback. The path is still selected only when static search attempts fail and `quality_mode` is `high`. This repairs the existing free browser option; it does not add a CAPTCHA solver or prove success against live challenges. The V4 service does not load the legacy GPU/browser dependency stack.

A browser can render JavaScript, which may resolve some failures of plain HTTP. Interactive CAPTCHA and source IP restrictions are separate problems; Google explicitly identifies automated search programs and search scrapers as potential causes of unusual-traffic challenges. [Google's explanation](https://support.google.com/websearch/answer/86640?hl=en). Headless browsers also need more runtime resources than ordinary page fetches, so use rendering where needed rather than for every request.

## Checked provider prices — 19 September 2026

Prices below are from each provider's own documentation, in USD; review current plan conditions before signing up. These providers are alternatives, not automatic V4 fallbacks.

| Option | Free allowance shown | Paid entry shown |
|---|---|---|
| [SerpApi](https://serpapi.com/pricing) | 250 searches/month | $25/month for 1,000 searches |
| [Serper](https://serper.dev/) | 2,500 free queries; do not assume monthly renewal | $50 top-up for 50,000 queries; credits valid six months; taxes extra |
| [Brave Search](https://brave.com/search/api/) | $5 monthly search credits, equivalent to 1,000 Search requests at the listed rate | $5 per 1,000 Search requests |
| [Tavily](https://docs.tavily.com/documentation/api-credits) | 1,000 credits/month, no card required; basic search costs one credit | $0.008 per pay-as-you-go credit |
| [SearXNG](https://docs.searxng.org/user/about.html) | Free self-hosted software; no purchased search credits with the supplied engine configuration | Existing-server resources and maintenance; upstream limits still apply |

No verified offer of 20,000 hosted web searches for $5 was found. Serper's starter unit rate would make 20,000 queries worth $20, but its listed starter purchase is $50. A search query is not a fully verified business: several queries and fetched pages may be needed per business. The cheapest useful improvement is avoiding repeated searches when a company website or regulator link is already known.

## Validation boundary

Offline tests exercise zero-provider-fee enrichment/discovery, authenticated CSV import, local search, budget enforcement, persistence/concurrency, SearXNG parsing/caching/cooldowns, Scrapling result parsing and process/network guards, and the corrected V3 browser invocation. The full existing verification suite remains in place. These tests do not measure live Google/Bing/DDG challenge success, deploy the SearXNG container, or establish real business-contact accuracy. Those require testing on the target Linux host and checking a labelled business sample.
