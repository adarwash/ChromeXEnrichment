#!/bin/bash

# ==============================================================================
# AI Data Enrichment System - Advanced Ubuntu Installer
# Features: GPU/VRAM Detection, Disk Space Checks, Smart Model Selection
# ==============================================================================

set -e # Exit on error

# Configuration
APP_DIR="/opt/ai-enrichment"
VENV_DIR="$APP_DIR/venv"
APP_USER="root"
PYTHON_VERSION="python3"
MIN_DISK_SPACE_GB=5
APP_PORT=8000

echo "------------------------------------------------"
echo " AI Enrichment System - Smart Installer"
echo "------------------------------------------------"

if [ -n "${COMPANIES_HOUSE_API_KEY:-}" ]; then
    echo "[*] Companies House API key detected. UK enrichment lookup will use API first."
else
    echo "[*] No Companies House API key set. UK enrichment lookup will crawl Companies House website."
    echo "[*] To use API mode, export COMPANIES_HOUSE_API_KEY before running this installer."
fi

# 1. System Update & Deps
echo "[*] Updating system and installing hardware utils..."
sudo apt-get update -y
sudo apt-get install -y curl git python3-pip python3-venv build-essential pciutils lsof

# Check for NVIDIA GPU and VRAM
GPU_COUNT=0
TOTAL_VRAM_MB=0
HAS_NVIDIA=false
OLLAMA_NUM_PARALLEL=1
OLLAMA_SCHED_SPREAD=0
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE="-1"
MULTI_GPU_MODE="disabled"

if command -v nvidia-smi &> /dev/null; then
    echo "[*] NVIDIA Driver detected. Querying hardware..."
    HAS_NVIDIA=true
    # Count GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    # Sum VRAM (in MiB)
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    echo "[+] Found $GPU_COUNT GPU(s). Total VRAM: ${TOTAL_VRAM_MB}MB."

    # Multi-GPU tuning for Ollama: spread scheduling and parallel workers.
    # Scale parallel inference slots based on available VRAM.
    if [ "$GPU_COUNT" -ge 2 ]; then
        OLLAMA_NUM_PARALLEL=$GPU_COUNT
        OLLAMA_SCHED_SPREAD=1
        OLLAMA_MAX_LOADED_MODELS=$GPU_COUNT
        MULTI_GPU_MODE="enabled (${GPU_COUNT} GPUs)"
        echo "[+] Multi-GPU mode enabled. Ollama will spread workloads across ${GPU_COUNT} GPUs."
    elif [ "$TOTAL_VRAM_MB" -gt 20000 ]; then
        OLLAMA_NUM_PARALLEL=3
        echo "[+] High-VRAM single GPU. Setting OLLAMA_NUM_PARALLEL=3 for faster concurrent inference."
    fi
else
    echo "[!] No NVIDIA drivers found. Mode: CPU Only."
fi

export OLLAMA_NUM_PARALLEL
export OLLAMA_SCHED_SPREAD
export OLLAMA_MAX_LOADED_MODELS
export OLLAMA_KEEP_ALIVE

# Check Disk Space
AVAIL_DISK_GB=$(df -BG /opt | awk 'NR==2 {print $4}' | sed 's/G//')
echo "[*] Available Disk Space: ${AVAIL_DISK_GB}GB"

if [ "$AVAIL_DISK_GB" -lt "$MIN_DISK_SPACE_GB" ]; then
    echo "[!] Warning: Low disk space. System will enforce the smallest model."
fi

# 2. Install Ollama
echo "[*] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server if not already running and wait until ready
ensure_ollama_running() {
    if ! pgrep -x "ollama" > /dev/null; then
        echo "[*] Starting Ollama server..."
        nohup ollama serve > /var/log/ollama.log 2>&1 &
    fi

    # Wait for Ollama API to be responsive
    for i in $(seq 1 20); do
        if curl -fsS http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "[!] Ollama failed to become ready within timeout. Check /var/log/ollama.log"
    return 1
}

configure_ollama_systemd_override() {
    if ! pidof systemd &> /dev/null; then
        return 0
    fi

    sudo mkdir -p /etc/systemd/system/ollama.service.d
    cat > /tmp/ollama-override.conf << OLLAMAOVR
[Service]
Environment="OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
Environment="OLLAMA_SCHED_SPREAD=$OLLAMA_SCHED_SPREAD"
Environment="OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"
Environment="OLLAMA_KEEP_ALIVE=-1"
OLLAMAOVR
    sudo mv /tmp/ollama-override.conf /etc/systemd/system/ollama.service.d/override.conf
}

ensure_ollama_running

# 3. Smart Model Selection Logic
# Model Sizes (Approximate):
# llama3.1:8b (4.7GB) - Best for decent GPUs
# llama3.2:1b (1.3GB) - Best for low VRAM/CPU/Low Disk
# phi3:mini (2.2GB)   - Alternative for constrained envs

SELECTED_MODEL=""
MODEL_NAME=""

# Logic Tree
if [ "$AVAIL_DISK_GB" -lt 6 ]; then
    # Critical low space
    SELECTED_MODEL="llama3.2:1b"
    echo "[!] Critical Disk Space. Selecting tiny model: $SELECTED_MODEL"
elif [ "$HAS_NVIDIA" = true ] && [ "$TOTAL_VRAM_MB" -gt 10000 ]; then
    # Good GPU (>=10GB VRAM)
    SELECTED_MODEL="llama3.1:8b"
    echo "[+] High Performance GPU detected. Selecting model: $SELECTED_MODEL"
elif [ "$HAS_NVIDIA" = true ] && [ "$TOTAL_VRAM_MB" -gt 4000 ]; then
    # Entry GPU (4-8GB VRAM)
    SELECTED_MODEL="llama3.2:1b" # 8B might be tight on 4GB, using 1b for speed
    echo "[+] Entry Level GPU detected. Selecting optimized model: $SELECTED_MODEL"
else
    # CPU Only
    SELECTED_MODEL="llama3.2:1b"
    echo "[+] CPU Mode. Selecting lightweight model: $SELECTED_MODEL"
fi

echo "[*] Pulling AI Model: $SELECTED_MODEL (This may take time...)"
ollama pull $SELECTED_MODEL

# Set environment variable for the app to know which model is in use
export AI_MODEL_NAME=$SELECTED_MODEL

# 4. Create Directory Structure
echo "[*] Setting up directories..."
sudo mkdir -p $APP_DIR/app
cd $APP_DIR

# 5. Python Environment
echo "[*] Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

# 6. Requirements
echo "[*] Installing Python dependencies..."
cat > $APP_DIR/app/requirements.txt << 'REQEOF'
fastapi==0.111.0
uvicorn[standard]==0.30.1
httpx==0.27.2
pydantic==2.7.4
phonenumbers==8.13.40
email-validator==2.2.0
trafilatura==1.8.0
ollama==0.2.1
dnspython==2.6.1
requests==2.32.3
beautifulsoup4==4.12.3
REQEOF

pip install --upgrade pip
pip install -r $APP_DIR/app/requirements.txt

# 7. FastAPI Application Code
echo "[*] Writing Application Code..."
cat <<'PYEOF' > $APP_DIR/app/main.py
import logging
import os
import asyncio
import re
import smtplib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs, unquote, urljoin
from pydantic import BaseModel, EmailStr, Field, model_validator
from fastapi import FastAPI, HTTPException
import requests
import trafilatura
import phonenumbers
import ollama
import json
from bs4 import BeautifulSoup

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get Model Name from Environment (set by installer)
AI_MODEL = os.getenv("AI_MODEL_NAME", "llama3.2:1b")
COMPANIES_HOUSE_API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY", "").strip()

app = FastAPI(
    title="Chrome X AI Enrichment API",
    description="""Self-hosted B2B company enrichment API powered by local AI (Ollama + llama3.1).

Features:
- **Zero-cost crawling** — no paid APIs required (optional Companies House API key for UK)
- **AI-powered extraction** — company name, industry, directors, address, description, tags
- **Companies House integration** — UK company status, directors, registered office, company number
- **Phone verification** — libphonenumber validation + cross-referencing across multiple sources
- **Flexible inputs** — accepts domain, company_name, linkedin_url, or free-text query
- **Discovery mode** — search by query + location, discover and enrich multiple businesses
- **Overall summary** — cross-referenced B2B profile with Companies House priority, verified phones, domains_scanned
- **Parallel processing** — concurrent AI inference, Companies House lookups, and page crawling
""",
    version="2.5.0"
)

# --- Pydantic Models ---

class EnrichRequest(BaseModel):
    domain: Optional[str] = Field(default=None, description="Company domain or website URL")
    company_name: Optional[str] = None
    query: Optional[str] = Field(default=None, description="Optional discovery query. If provided, /enrich behaves like crawl-businesses.")
    phone_number: Optional[str] = Field(default=None, description="Optional phone hint. May be inaccurate; use only as a soft clue.")
    location: Optional[str] = None
    max_results: int = Field(default=5, ge=1, le=50)
    country: Optional[str] = None
    industry_hint: Optional[str] = None
    linkedin_url: Optional[str] = None
    additional_context: Optional[str] = None

    @model_validator(mode="after")
    def validate_identifiers(self):
        if self.query:
            return self
        if not any([self.domain, self.company_name, self.linkedin_url]):
            raise ValueError("Provide at least one of: domain, company_name, linkedin_url, query")
        return self

class BatchEnrichRequest(BaseModel):
    domains: List[str]

class VerifyEmailRequest(BaseModel):
    email: EmailStr

class VerifyPhoneRequest(BaseModel):
    phone_number: str
    country_code: str = "US"

class VerifyBusinessAddressRequest(BaseModel):
    company_name: Optional[str] = Field(default=None, description="Legal or trading company name")
    domain: Optional[str] = Field(default=None, description="Official company domain if known")
    location: Optional[str] = Field(default=None, description="City, state, region, or country hint")
    country: Optional[str] = Field(default=None, description="Optional country hint")
    max_sources: int = Field(default=8, ge=3, le=20)

    @model_validator(mode="after")
    def validate_lookup_input(self):
        if not (self.company_name or self.domain):
            raise ValueError("Provide at least one of: company_name, domain")
        return self

class BatchVerifyBusinessAddressRequest(BaseModel):
    items: List[VerifyBusinessAddressRequest] = Field(..., min_length=1, max_length=100)

class CrawlRequest(BaseModel):
    url: str

class CrawlBusinessesRequest(BaseModel):
    query: str = Field(..., description="Business type or keyword, e.g. 'HVAC companies'")
    location: Optional[str] = Field(default=None, description="Optional city/region/country")
    max_results: int = Field(default=10, ge=1, le=50)

class SystemStatus(BaseModel):
    status: str
    ai_model_loaded: str
    hardware_mode: str

# --- Core Logic: AI & Helpers ---

def normalize_domain(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def fetch_html_from_url(url: str) -> Optional[str]:
    try:
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
        )
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("content-type", "").lower():
            return None
        return resp.text
    except Exception:
        return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def extract_contact_signals(text_content: str) -> Dict[str, Any]:
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_content)
    phones = re.findall(r"\+?\d[\d\s().-]{7,}\d", text_content)
    return {
        "email_count": len(set(e.lower() for e in emails)),
        "phone_count": len(set(phones)),
    }


def extract_contact_details(text_content: str, country_hint: Optional[str] = None) -> Dict[str, List[str]]:
    if not text_content:
        return {"emails": [], "phones": []}

    raw_emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_content)
    raw_phones = re.findall(r"\+?\d[\d\s().-]{7,}\d", text_content)

    emails = []
    seen_emails = set()
    for email in raw_emails:
        item = email.strip().lower()
        if item and item not in seen_emails:
            seen_emails.add(item)
            emails.append(item)
        if len(emails) >= 5:
            break

    phones = []
    seen_phones = set()
    for phone in raw_phones:
        cleaned = re.sub(r"\s+", " ", phone).strip()
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) < 8 or len(digits) > 15:
            continue

        # Filter obvious date-like values often present on registry pages.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned) or re.fullmatch(r"\d{2}-\d{2}-\d{4}", cleaned):
            continue
        if re.fullmatch(r"\d{8}", digits) and (digits.startswith("19") or digits.startswith("20")):
            continue
        # Catch date substrings even when extra chars are appended (e.g. "2010-03-09 16").
        if re.search(r"(?:19|20)\d{2}-\d{2}-\d{2}", cleaned):
            continue
        # Reject range/category indicators like "0-2 - 3-5 - 6-20 - 21-50 - 51".
        # Real phone numbers have at least one run of 3+ consecutive digits.
        if not re.search(r"\d{3}", cleaned):
            continue

        # Prefer values that look like real phone formats.
        if len(digits) < 10 and not cleaned.startswith("+"):
            continue

        key = digits[-12:]
        if key in seen_phones:
            continue
        seen_phones.add(key)
        phones.append(cleaned)
        if len(phones) >= 5:
            break

    # Verify each phone with libphonenumber and keep only valid ones.
    verified_phones = []
    for raw_phone in phones:
        result = verify_phone_offline(raw_phone, country_hint)
        if result and result.get("is_valid"):
            verified_phones.append(result["international"])
        elif result and result.get("is_possible"):
            verified_phones.append(result["international"])

    return {
        "emails": emails,
        "phones": verified_phones,
    }


def verify_phone_offline(raw_phone: str, country_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Validate a phone string using libphonenumber. Returns None if unparseable."""
    region = (country_hint or "GB").strip().upper()
    # Map common country names to ISO codes.
    region_map = {
        "UK": "GB", "UNITED KINGDOM": "GB", "ENGLAND": "GB",
        "SCOTLAND": "GB", "WALES": "GB", "NORTHERN IRELAND": "GB",
        "US": "US", "USA": "US", "UNITED STATES": "US",
    }
    region = region_map.get(region, region)
    if len(region) != 2:
        region = "GB"

    try:
        parsed = phonenumbers.parse(raw_phone, region)
        is_valid = phonenumbers.is_valid_number(parsed)
        is_possible = phonenumbers.is_possible_number(parsed)
        return {
            "raw": raw_phone,
            "is_valid": is_valid,
            "is_possible": is_possible,
            "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
            "number_type": phonenumbers.number_type(parsed),
        }
    except Exception:
        return None


def format_headquarters_location(location_data: Any) -> Optional[str]:
    if not location_data:
        return None
    if isinstance(location_data, str):
        value = normalize_address_text(location_data)
        return value if value else None
    if not isinstance(location_data, dict):
        return None

    ordered_keys = [
        "street_address",
        "street",
        "address",
        "line1",
        "line2",
        "city",
        "state",
        "region",
        "county",
        "postcode",
        "post_code",
        "postal_code",
        "zip",
        "country",
    ]
    seen = set()
    parts = []
    for key in ordered_keys:
        value = str(location_data.get(key) or "").strip()
        if not value:
            continue
        low = value.lower()
        if low in seen:
            continue
        # Avoid duplicates like "..., Crowborough" + city="Crowborough".
        if any(low in existing.lower() or existing.lower() in low for existing in parts):
            continue
        seen.add(low)
        parts.append(value)
    return ", ".join(parts) if parts else None


def format_registered_office_address(address_data: Any) -> Optional[str]:
    if not address_data:
        return None
    if isinstance(address_data, str):
        value = normalize_address_text(address_data)
        return value if value else None
    if not isinstance(address_data, dict):
        return None

    ordered_keys = [
        "address_line_1",
        "address_line_2",
        "locality",
        "region",
        "postal_code",
        "country",
    ]
    seen = set()
    parts = []
    for key in ordered_keys:
        value = str(address_data.get(key) or "").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            parts.append(value)
    return ", ".join(parts) if parts else None


def build_overall_b2b_summary(query: str, location: Optional[str], results: List[Dict[str, Any]], domains_scanned: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    def names_look_related(a: str, b: str) -> bool:
        a_tokens = set(company_name_tokens(a or ""))
        b_tokens = set(company_name_tokens(b or ""))
        if not a_tokens or not b_tokens:
            return slugify_text(a or "") == slugify_text(b or "")
        overlap = len(a_tokens.intersection(b_tokens))
        return overlap >= min(2, len(a_tokens), len(b_tokens))

    def choose_best_text(values: List[str]) -> Optional[str]:
        ranked: Dict[str, Dict[str, Any]] = {}
        for idx, raw in enumerate(values):
            value = normalize_address_text(str(raw or ""))
            if not value:
                continue
            key = slugify_text(value)
            if not key:
                continue
            if key not in ranked:
                ranked[key] = {"value": value, "count": 0, "first_index": idx}
            ranked[key]["count"] += 1
            if idx < ranked[key]["first_index"]:
                ranked[key]["first_index"] = idx

        if not ranked:
            return None

        ordered = sorted(
            ranked.values(),
            key=lambda item: (int(item["count"]), -int(item["first_index"]), len(str(item["value"]))),
            reverse=True,
        )
        return ordered[0]["value"]

    def dedupe_ranked(values: List[str], limit: int = 5) -> List[str]:
        counters: Dict[str, Dict[str, Any]] = {}
        for idx, raw in enumerate(values):
            value = normalize_address_text(str(raw or ""))
            if not value:
                continue
            key = slugify_text(value)
            if not key:
                continue
            if key not in counters:
                counters[key] = {"value": value, "count": 0, "first_index": idx}
            counters[key]["count"] += 1
            if idx < counters[key]["first_index"]:
                counters[key]["first_index"] = idx

        ordered = sorted(
            counters.values(),
            key=lambda item: (int(item["count"]), -int(item["first_index"])),
            reverse=True,
        )
        return [str(item["value"]) for item in ordered[:limit]]

    def unique_directors(director_rows: List[Any], limit: int = 10) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen = set()
        for row in director_rows:
            if isinstance(row, dict):
                name = str(row.get("name") or "").strip()
                title = str(row.get("title") or "Director").strip()
                if not name:
                    continue
                key = (name.lower(), title.lower())
                if key in seen:
                    continue
                seen.add(key)
                normalized = dict(row)
                normalized["name"] = name
                normalized["title"] = title
                out.append(normalized)
            elif isinstance(row, str):
                name = row.strip()
                if not name:
                    continue
                key = (name.lower(), "director")
                if key in seen:
                    continue
                seen.add(key)
                out.append({"name": name, "title": "Director"})

            if len(out) >= limit:
                break
        return out

    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return {
            "query": query,
            "location": location,
            "company_name": None,
            "website": None,
            "linkedin_url": None,
            "address": None,
            "phones": [],
            "emails": [],
            "directors": [],
            "companies_house_url": None,
            "confidence_band": "low",
            "reason": "No successful enrichment result",
        }

    successful_sorted = sorted(
        successful,
        key=lambda item: (
            int((item.get("confidence") or {}).get("overall") or 0),
            len(((item.get("enrichment") or {}).get("directors") or [])),
        ),
        reverse=True,
    )

    companies_house_hits = [
        item for item in successful_sorted
        if isinstance(item.get("companies_house"), dict)
        and (
            item["companies_house"].get("company_number")
            or item["companies_house"].get("matched_company_name")
            or item["companies_house"].get("profile_url")
        )
    ]

    all_name_candidates: List[str] = []
    for item in companies_house_hits:
        ch = item.get("companies_house") or {}
        matched = str(ch.get("matched_company_name") or "").strip()
        if matched:
            all_name_candidates.append(matched)
    for item in successful_sorted:
        enr = item.get("enrichment") or {}
        ai_name = str(enr.get("company_name") or "").strip()
        if ai_name:
            all_name_candidates.append(ai_name)
        title_name = str(item.get("title") or "").strip()
        if title_name:
            all_name_candidates.append(title_name)

    company_name = choose_best_text(all_name_candidates)

    # Trust only Companies House records that match the target company name/query.
    trusted_companies_house_hits = []
    for item in companies_house_hits:
        ch = item.get("companies_house") or {}
        matched = str(ch.get("matched_company_name") or "").strip()
        if matched and (names_look_related(matched, company_name or "") or names_look_related(matched, query or "")):
            trusted_companies_house_hits.append(item)

    primary = trusted_companies_house_hits[0] if trusted_companies_house_hits else (companies_house_hits[0] if companies_house_hits else successful_sorted[0])

    companies_house_url = None
    if isinstance(primary.get("companies_house"), dict):
        companies_house_url = primary["companies_house"].get("profile_url")

    linkedin_candidates = [
        str((item.get("enrichment") or {}).get("linkedin_url") or "").strip()
        for item in successful_sorted
        if str((item.get("enrichment") or {}).get("linkedin_url") or "").strip()
    ]
    linkedin_url = choose_best_text(linkedin_candidates)

    # Prefer a non-aggregator site as the representative website.
    website_candidates = [
        str(item.get("url") or "").strip()
        for item in successful_sorted
        if str(item.get("url") or "").strip() and not item.get("is_aggregator")
    ]
    website_url = choose_best_text(website_candidates)
    if not website_url:
        website_url = companies_house_url or primary.get("url")

    # Rank address candidates by source trust before length.
    # non-aggregator website > trusted Companies House/aggregator AI.
    address_candidates: List[Dict[str, Any]] = []
    for item in (trusted_companies_house_hits or companies_house_hits):
        ch = item.get("companies_house") or {}
        formatted = format_registered_office_address(ch.get("registered_office_address"))
        if formatted:
            address_candidates.append({"address": formatted, "priority": 4})
    for item in successful_sorted:
        enr = item.get("enrichment") or {}
        formatted = format_headquarters_location(enr.get("headquarters_location"))
        if formatted:
            priority = 3 if not item.get("is_aggregator") else 1
            address_candidates.append({"address": formatted, "priority": priority})

    # Prefer the longest address that contains a postcode-like pattern.
    uk_postcode_re = re.compile(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", re.IGNORECASE)
    us_zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    def has_postcode(addr: str) -> bool:
        return bool(uk_postcode_re.search(addr) or us_zip_re.search(addr))

    with_postcode = [c for c in address_candidates if has_postcode(str(c.get("address") or ""))]
    if with_postcode:
        # Prefer higher priority source, then longest detailed address.
        best = max(with_postcode, key=lambda c: (int(c.get("priority") or 0), len(str(c.get("address") or ""))))
        address = str(best.get("address") or "")
    else:
        # Fall back to highest priority then longest.
        if address_candidates:
            best = max(address_candidates, key=lambda c: (int(c.get("priority") or 0), len(str(c.get("address") or ""))))
            address = str(best.get("address") or "")
        else:
            address = None

    # --- Phone cross-referencing ---
    # Collect phones tagged with the domain they came from.
    phone_by_source: Dict[str, List[str]] = {}  # e164 -> list of source domains
    phone_display: Dict[str, str] = {}  # e164 -> international display format
    for item in successful_sorted:
        source_domain = item.get("domain") or ""
        contact = item.get("contact_details") or {}
        for phone in (contact.get("phones") or []):
            # Normalize to E.164 for dedup / cross-ref.
            parsed_phone = verify_phone_offline(phone, location)
            if not parsed_phone:
                continue
            key = parsed_phone.get("e164") or phone
            if key not in phone_by_source:
                phone_by_source[key] = []
                phone_display[key] = parsed_phone.get("international") or phone
            if source_domain and source_domain not in phone_by_source[key]:
                phone_by_source[key].append(source_domain)

    # Determine the company's official website domain (non-aggregator, non-directory).
    official_domain = None
    for item in successful_sorted:
        if item.get("is_aggregator"):
            continue
        candidate_domain = item.get("domain") or ""
        if candidate_domain:
            official_domain = candidate_domain
            break

    # If we have an official website, try crawling its /contact page to find phones
    # that we can cross-reference against.
    if official_domain:
        contact_urls = [
            f"https://{official_domain}/contact",
            f"https://{official_domain}/contact-us",
        ]
        for contact_url in contact_urls:
            try:
                contact_text = clean_text_from_url(contact_url)
                if not contact_text:
                    continue
                extra = extract_contact_details(contact_text, country_hint=location)
                for phone in (extra.get("phones") or []):
                    parsed_phone = verify_phone_offline(phone, location)
                    if not parsed_phone:
                        continue
                    key = parsed_phone.get("e164") or phone
                    if key not in phone_by_source:
                        phone_by_source[key] = []
                        phone_display[key] = parsed_phone.get("international") or phone
                    if official_domain not in phone_by_source[key]:
                        phone_by_source[key].append(official_domain)
            except Exception:
                pass

    # Keep phones that appear on 2+ independent sources OR on the official website.
    verified_phones: List[str] = []
    for key, sources in phone_by_source.items():
        on_official = official_domain and official_domain in sources
        multi_source = len(sources) >= 2
        if on_official or multi_source:
            verified_phones.append(phone_display[key])

    email_candidates: List[str] = []
    for item in successful_sorted:
        contact = item.get("contact_details") or {}
        email_candidates.extend(contact.get("emails") or [])

    directors_input: List[Any] = []
    for item in (trusted_companies_house_hits or companies_house_hits):
        ch = item.get("companies_house") or {}
        if isinstance(ch.get("directors"), list):
            directors_input.extend(ch.get("directors") or [])
    if not directors_input:
        for item in successful_sorted:
            enr = item.get("enrichment") or {}
            if isinstance(enr.get("directors"), list):
                directors_input.extend(enr.get("directors") or [])
    directors = unique_directors(directors_input)

    return {
        "query": query,
        "location": location,
        "company_name": company_name or primary.get("title"),
        "website": website_url,
        "linkedin_url": linkedin_url,
        "address": address,
        "phones": verified_phones,
        "emails": dedupe_ranked(email_candidates),
        "directors": directors,
        "companies_house_url": companies_house_url,
        "company_number": (primary.get("companies_house") or {}).get("company_number"),
        "company_status": (primary.get("companies_house") or {}).get("company_status"),
        "confidence_band": (primary.get("confidence") or {}).get("band"),
        "source": primary.get("source"),
        "top_domains": [item.get("domain") for item in successful_sorted[:3] if item.get("domain")],
        "domains_scanned": domains_scanned or [],
        "verified_address": None,
        "verified_address_fields": None,
    }


def score_enrichment(ai_data: Dict[str, Any], hints: Dict[str, str], text_content: str) -> Dict[str, Any]:
    score = 0
    reasons = []

    if ai_data.get("company_name"):
        score += 20
        reasons.append("company_name_found")
    if ai_data.get("industry"):
        score += 15
        reasons.append("industry_found")
    if ai_data.get("description"):
        score += 15
        reasons.append("description_found")
    if ai_data.get("tags") and isinstance(ai_data.get("tags"), list) and len(ai_data.get("tags")) > 0:
        score += 10
        reasons.append("tags_found")
    if ai_data.get("directors") and isinstance(ai_data.get("directors"), list) and len(ai_data.get("directors")) > 0:
        score += 10
        reasons.append("directors_found")

    signals = extract_contact_signals(text_content)
    if signals["email_count"] > 0:
        score += 15
        reasons.append("email_signal")
    if signals["phone_count"] > 0:
        score += 10
        reasons.append("phone_signal")

    hint_name = (hints.get("company_name") or "").strip().lower()
    extracted_name = str(ai_data.get("company_name") or "").strip().lower()
    if hint_name and extracted_name:
        if hint_name in extracted_name or extracted_name in hint_name:
            score += 15
            reasons.append("name_hint_match")

    overall = min(100, max(0, score))
    band = "high" if overall >= 75 else "medium" if overall >= 45 else "low"
    return {
        "overall": overall,
        "band": band,
        "reasons": reasons,
        "signals": signals,
    }


def build_ai_prompt(text_content: str, hints: Dict[str, str]) -> str:
    hint_lines = []
    for k, v in hints.items():
        if v:
            hint_lines.append(f"- {k}: {v}")

    hint_block = "\n".join(hint_lines) if hint_lines else "- none"

    return f"""
    You are a B2B data extraction specialist.

    Known hints about target business:
    {hint_block}

    Analyze the website text and return ONLY valid JSON with keys:
    company_name, industry, description, tags, headquarters_location, linkedin_url, directors

    Rules:
    - Use null for unknown values.
    - tags must be an array of short strings.
    - directors must be an array of objects with keys: name, title.
    - if no directors are found, return an empty array for directors.
    - Treat hints as guidance only, not ground truth.
    - The phone_number hint may be wrong. Do not rely on it unless the website text clearly corroborates it.
    - Prefer data that aligns with provided hints only when the website text supports it.

    Text:
    {text_content[:5000]}
    """


def extract_director_candidates(text_content: str) -> List[Dict[str, str]]:
    patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[-,:]\s*(Managing Director|Director|CEO|Chief Executive Officer|Founder)",
        r"(Managing Director|Director|CEO|Chief Executive Officer|Founder)\s*[-,:]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    ]

    found = []
    seen = set()
    for pattern in patterns:
        for m in re.finditer(pattern, text_content):
            g1 = (m.group(1) or "").strip()
            g2 = (m.group(2) or "").strip()

            # Pattern order differs, normalize to name/title
            if any(t in g1.lower() for t in ["director", "ceo", "founder", "chief executive"]):
                title, name = g1, g2
            else:
                name, title = g1, g2

            key = (name.lower(), title.lower())
            if name and key not in seen:
                seen.add(key)
                found.append({"name": name, "title": title})

            if len(found) >= 10:
                return found

    return found


def normalize_directors(ai_data: Dict[str, Any], text_content: str) -> Dict[str, Any]:
    directors = ai_data.get("directors")
    normalized: List[Dict[str, str]] = []

    if isinstance(directors, list):
        for item in directors:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                title = str(item.get("title") or "Director").strip()
                if name:
                    normalized.append({"name": name, "title": title})
            elif isinstance(item, str):
                raw = item.strip()
                if raw:
                    normalized.append({"name": raw, "title": "Director"})

    if not normalized:
        normalized = extract_director_candidates(text_content)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for d in normalized:
        key = (d["name"].lower(), d["title"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    ai_data["directors"] = uniq
    return ai_data


async def run_ai_extraction(text_content: str, hints: Optional[Dict[str, str]] = None):
    """
    Asynchronous wrapper for Ollama calls.
    Prevents blocking the event loop during heavy AI inference.
    """
    hints = hints or {}
    prompt = build_ai_prompt(text_content, hints)
    
    loop = asyncio.get_event_loop()
    try:
        # Run synchronous ollama call in a thread executor
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(model=AI_MODEL, messages=[{'role': 'user', 'content': prompt}], format='json')
        )
        content = response['message']['content']
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        return normalize_directors(parsed, text_content)
    except Exception as e:
        logger.error(f"AI Extraction Error: {e}")
        return None

def get_mx_records(domain):
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, 'MX')
        return [rdata.exchange.to_text() for rdata in answers]
    except Exception:
        return []

def check_smtp_connection(email: str):
    domain = email.split('@')[1]
    try:
        mx_records = get_mx_records(domain)
        if not mx_records: return False, "No MX records"
        
        mx_host = str(mx_records[0])
        server = smtplib.SMTP(timeout=5)
        server.connect(mx_host, 25)
        server.helo(server.local_hostname)
        server.mail('test@local.ai')
        code, message = server.rcpt(email)
        server.quit()
        
        return (True, "Valid") if code == 250 else (False, f"Rejected {code}")
    except Exception as e:
        return None, f"Undetermined ({type(e).__name__})"

def clean_text_from_url(url: str):
    html = fetch_html_from_url(url)
    if not html:
        return None

    try:
        extracted = trafilatura.extract(html, include_comments=False)
        if extracted:
            return extracted
    except Exception:
        pass

    return html_to_text(html)


def parse_ddg_result_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("http"):
        return href
    if "uddg=" in href:
        parsed = urlparse(href)
        q = parse_qs(parsed.query)
        if "uddg" in q and len(q["uddg"]) > 0:
            return unquote(q["uddg"][0])
    return ""


def discover_business_urls(query: str, location: Optional[str], max_results: int) -> List[Dict[str, str]]:
    search_query = query.strip()
    if location:
        search_query = f"{search_query} {location.strip()}"

    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": search_query},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    candidates = []
    seen_domains = set()

    for link in soup.select("a.result__a"):
        href = link.get("href", "")
        real_url = parse_ddg_result_url(href)
        if not real_url:
            continue
        domain = normalize_domain(real_url)
        if not domain or domain in seen_domains:
            continue
        seen_domains.add(domain)
        candidates.append({
            "title": link.get_text(" ", strip=True),
            "url": real_url,
            "domain": domain,
        })
        if len(candidates) >= max_results:
            break

    return candidates


def search_public_results(query: str, max_results: int) -> List[Dict[str, str]]:
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()
    for link in soup.select("a.result__a"):
        href = link.get("href", "")
        real_url = parse_ddg_result_url(href)
        if not real_url:
            continue
        key = real_url.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "title": link.get_text(" ", strip=True),
            "url": real_url,
            "domain": normalize_domain(real_url),
        })
        if len(results) >= max_results:
            break

    return results


def normalize_domain_hint(raw_domain: Optional[str]) -> str:
    if not raw_domain:
        return ""
    candidate = raw_domain.strip()
    if not candidate:
        return ""
    if not candidate.startswith("http://") and not candidate.startswith("https://"):
        candidate = f"https://{candidate}"
    return normalize_domain(candidate)


def slugify_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def has_postcode_pattern(text: str) -> bool:
    """Return True if text contains a UK postcode or US ZIP code pattern."""
    if not text:
        return False
    if re.search(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", text, re.IGNORECASE):
        return True
    if re.search(r"\b\d{5}(?:-\d{4})?\b", text):
        return True
    return False


def company_name_tokens(company_name: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9]+", company_name.lower()):
        if len(token) > 2 and token not in {"inc", "llc", "ltd", "the", "and", "group", "company", "co"}:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def likely_company_match(company_name: str, title: str, url: str) -> bool:
    tokens = company_name_tokens(company_name)
    haystack = slugify_text(f"{title} {url}")
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token in haystack)
    return matched >= min(2, len(tokens)) or company_name.lower() in haystack


def extract_relevant_site_links(base_url: str, html: str, max_links: int = 8) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_domain = normalize_domain(base_url)
    keywords = [
        "contact",
        "about",
        "location",
        "office",
        "find-us",
        "find us",
        "visit",
        "showroom",
        "headquarter",
        "registered-office",
        "registered office",
    ]

    urls = []
    seen = set()
    for link in soup.select("a[href]"):
        href = (link.get("href") or "").strip()
        if not href or href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
            continue
        text = f"{link.get_text(' ', strip=True)} {href}".lower()
        if not any(keyword in text for keyword in keywords):
            continue
        full_url = urljoin(base_url, href)
        if normalize_domain(full_url) != base_domain:
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        urls.append(full_url)
        if len(urls) >= max_links:
            break

    return urls


def build_official_site_sources(domain: str) -> List[Dict[str, str]]:
    if not domain:
        return []

    base_url = f"https://{domain}"
    sources = [{
        "title": "Official site /",
        "url": base_url,
        "domain": domain,
        "source_type": "official_website",
    }]

    html = fetch_html_from_url(base_url)
    discovered_urls = extract_relevant_site_links(base_url, html, max_links=8) if html else []
    for url in discovered_urls:
        sources.append({
            "title": f"Official site {urlparse(url).path or '/'}",
            "url": url,
            "domain": domain,
            "source_type": "official_website",
        })

    common_paths = ["/contact", "/contact-us", "/about", "/locations", "/find-us", "/visit-us", "/offices"]
    known_urls = {item["url"] for item in sources}
    for path in common_paths:
        url = f"{base_url}{path}"
        if url not in known_urls:
            sources.append({
                "title": f"Official site {path}",
                "url": url,
                "domain": domain,
                "source_type": "official_website",
            })
            known_urls.add(url)

    return sources


def discover_address_sources(company_name: str, domain: Optional[str], location: Optional[str], max_sources: int) -> List[Dict[str, str]]:
    normalized_domain = normalize_domain_hint(domain)
    sources = []
    seen_urls = set()
    official_limit = min(4, max(2, max_sources // 2)) if normalized_domain else 0

    for item in build_official_site_sources(normalized_domain)[:official_limit]:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            sources.append(item)

    queries = [
        f'"{company_name}" address',
        f'"{company_name}" contact',
        f'site:linkedin.com/company "{company_name}"',
        f'site:yelp.com "{company_name}" address',
        f'site:google.com/maps "{company_name}" address',
        f'site:facebook.com "{company_name}" address',
        f'"{company_name}" linkedin yelp google address',
    ]
    if location:
        queries.insert(0, f'"{company_name}" {location} address')
    if normalized_domain:
        queries.append(f'site:{normalized_domain} address')

    preferred_domains = [
        normalized_domain,
        "linkedin.com",
        "yelp.com",
        "google.com",
        "g.page",
        "maps.apple.com",
        "facebook.com",
        "instagram.com",
        "find-and-update.company-information.service.gov.uk",
        "yellowpages.com",
        "crunchbase.com",
        "mapquest.com",
        "bing.com",
    ]

    for query in queries:
        try:
            for item in search_public_results(query, max_sources):
                url = item.get("url", "")
                result_domain = item.get("domain", "")
                if not url or url in seen_urls:
                    continue
                if preferred_domains and not any(d and d in result_domain for d in preferred_domains) and not likely_company_match(company_name, item.get("title", ""), url):
                    continue
                seen_urls.add(url)
                source_type = "directory"
                if normalized_domain and result_domain == normalized_domain:
                    source_type = "official_website"
                elif "linkedin.com" in result_domain:
                    source_type = "linkedin"
                elif "yelp.com" in result_domain:
                    source_type = "yelp"
                elif "google.com" in result_domain or "g.page" in result_domain:
                    source_type = "google"
                elif "company-information.service.gov.uk" in result_domain:
                    source_type = "companies_house"

                sources.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "domain": result_domain,
                    "source_type": source_type,
                })
                if len(sources) >= max_sources:
                    return sources[:max_sources]
        except Exception as e:
            logger.warning(f"Address source discovery failed for query '{query}': {e}")

    if len(sources) <= official_limit:
        try:
            fallback_results = discover_business_urls(company_name, location, max_sources)
            for item in fallback_results:
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "domain": item.get("domain", ""),
                    "source_type": "directory",
                })
                if len(sources) >= max_sources:
                    return sources[:max_sources]
        except Exception as e:
            logger.warning(f"Generic address discovery fallback failed: {e}")

    for item in build_official_site_sources(normalized_domain)[official_limit:]:
        if len(sources) >= max_sources:
            break
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            sources.append(item)

    return sources[:max_sources]


def normalize_address_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip(" ,;|-")
    return text


def extract_address_candidates(text_content: str) -> List[str]:
    if not text_content:
        return []

    street_token = r"(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|way|court|ct|place|pl|parkway|pkwy|highway|hwy|terrace|ter|circle|cir|close|crescent|cres|mews|square|sq|business park|industrial estate)"
    unit_token = r"(?:suite|ste|unit|floor|fl|building|bldg|room|rm|office|dept|department|level)"
    us_zip_token = r"\d{5}(?:-\d{4})?"
    uk_postcode_token = r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}"
    postcode_token = rf"(?:{uk_postcode_token}|{us_zip_token})"
    us_city_state_pattern = re.compile(rf"\b[A-Za-z][A-Za-z .'-]{{1,40}},\s*[A-Z]{{2}}\s+{us_zip_token}\b")
    uk_city_postcode_pattern = re.compile(rf"\b[A-Za-z][A-Za-z .'-]{{1,50}}\s+{uk_postcode_token}\b", flags=re.IGNORECASE)
    street_pattern = re.compile(
        rf"\b\d{{1,6}}(?:[A-Za-z-])?\s+[A-Za-z0-9][A-Za-z0-9.,'#/& -]{{2,110}}\s+{street_token}\b(?:[^\n]{{0,90}})?",
        flags=re.IGNORECASE,
    )
    label_pattern = re.compile(r"(?:address|registered office|registered address|head office|headquarters|visit us|our office|contact us|location)", flags=re.IGNORECASE)
    unit_pattern = re.compile(rf"\b{unit_token}\b", flags=re.IGNORECASE)

    candidates = []
    seen = set()

    def add_candidate(value: str):
        normalized = normalize_address_text(value)
        if len(normalized) < 12 or len(normalized) > 220:
            return
        key = slugify_text(normalized)
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    lines = [normalize_address_text(line) for line in text_content.splitlines()]
    filtered_lines = [line for line in lines if line and len(line) <= 120]

    for idx, line in enumerate(filtered_lines):
        next_lines = filtered_lines[idx + 1: idx + 4]
        block = normalize_address_text(" ".join([line] + next_lines))
        has_street = bool(street_pattern.search(line) or street_pattern.search(block))
        has_postcode = bool(re.search(postcode_token, line, flags=re.IGNORECASE) or us_city_state_pattern.search(block) or uk_city_postcode_pattern.search(block))
        has_label = bool(label_pattern.search(line))
        if has_street and has_postcode:
            add_candidate(block)
            add_candidate(line)
        elif has_label and (street_pattern.search(block) or unit_pattern.search(block)) and has_postcode:
            add_candidate(block)

    flat_text = normalize_address_text(text_content)
    for match in street_pattern.finditer(flat_text):
        snippet = flat_text[max(0, match.start() - 20): min(len(flat_text), match.end() + 110)]
        snippet = normalize_address_text(snippet)
        if us_city_state_pattern.search(snippet) or uk_city_postcode_pattern.search(snippet) or re.search(postcode_token, snippet, flags=re.IGNORECASE):
            add_candidate(snippet)

    return candidates[:15]


def address_source_weight(source: Dict[str, str], official_domain: str) -> int:
    domain = source.get("domain", "")
    source_type = source.get("source_type", "directory")
    if official_domain and domain == official_domain:
        return 6
    if source_type == "companies_house":
        return 5
    if source_type in {"linkedin", "yelp", "google"}:
        return 3
    if source_type == "directory":
        return 2
    return 1


def location_match_bonus(address: str, location: Optional[str], country: Optional[str]) -> int:
    haystack = (address or "").lower()
    bonus = 0
    for token_group in [location, country]:
        if not token_group:
            continue
        parts = [p.strip().lower() for p in re.split(r"[,/|]", token_group) if p.strip()]
        if any(part in haystack for part in parts):
            bonus += 1
    return bonus


def build_address_verification_prompt(company_name: str, location: Optional[str], country: Optional[str], evidence: List[Dict[str, Any]]) -> str:
    evidence_lines = []
    for item in evidence[:12]:
        evidence_lines.append(
            f"- source_type: {item.get('source_type')} | domain: {item.get('domain')} | url: {item.get('url')} | address: {item.get('address')} | weight: {item.get('weight')}"
        )

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- none"

    return f"""
    You are validating a business address from public web evidence.

    Company name: {company_name}
    Location hint: {location or 'unknown'}
    Country hint: {country or 'unknown'}

    Evidence:
    {evidence_block}

    Return ONLY valid JSON with keys:
    verified, best_address, confidence_band, reason

    Rules:
    - verified must be true only if the address appears credible and supported by at least one strong source or multiple public sources.
    - best_address must be null if evidence is too weak.
    - confidence_band must be one of: high, medium, low.
    - Keep reason short.
    """


async def run_ai_address_verification(company_name: str, location: Optional[str], country: Optional[str], evidence: List[Dict[str, Any]]):
    if not evidence:
        return None

    prompt = build_address_verification_prompt(company_name, location, country, evidence)
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(model=AI_MODEL, messages=[{'role': 'user', 'content': prompt}], format='json')
        )
        content = response['message']['content']
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logger.warning(f"AI address verification failed: {e}")
        return None


def verify_business_address_sources(company_name: str, domain: Optional[str], location: Optional[str], country: Optional[str], max_sources: int) -> Dict[str, Any]:
    official_domain = normalize_domain_hint(domain)
    sources = discover_address_sources(company_name, official_domain, location, max_sources)
    scored_candidates: Dict[str, Dict[str, Any]] = {}
    evidence = []

    def process_source(source: Dict[str, str]) -> Dict[str, Any]:
        text_content = clean_text_from_url(source["url"])
        if not text_content:
            return {"source": source, "candidates": []}
        return {
            "source": source,
            "candidates": extract_address_candidates(text_content),
        }

    with ThreadPoolExecutor(max_workers=min(10, max(1, len(sources)))) as executor:
        futures = [executor.submit(process_source, source) for source in sources]
        for future in as_completed(futures):
            result = future.result()
            source = result["source"]
            candidates = result["candidates"]
            if not candidates:
                continue

            weight = address_source_weight(source, official_domain)
            for candidate in candidates:
                key = re.sub(r"[^a-z0-9]+", " ", candidate.lower()).strip()
                if not key:
                    continue
                total_weight = weight + location_match_bonus(candidate, location, country)
                if key not in scored_candidates:
                    scored_candidates[key] = {
                        "address": candidate,
                        "score": 0,
                        "sources": [],
                    }
                scored_candidates[key]["score"] += total_weight
                scored_candidates[key]["sources"].append({
                    "url": source["url"],
                    "domain": source["domain"],
                    "source_type": source["source_type"],
                    "weight": total_weight,
                })
                evidence.append({
                    "address": candidate,
                    "url": source["url"],
                    "domain": source["domain"],
                    "source_type": source["source_type"],
                    "weight": total_weight,
                })

    ranked = sorted(scored_candidates.values(), key=lambda item: (item["score"], len(item["sources"])), reverse=True)
    best = ranked[0] if ranked else None
    heuristics = {
        "verified": bool(best and (best["score"] >= 6 or len(best["sources"]) >= 2)),
        "best_address": best["address"] if best else None,
        "confidence_band": "high" if best and best["score"] >= 8 else "medium" if best and best["score"] >= 5 else "low",
        "reason": "official site or repeated public-source match" if best else "no credible address extracted",
    }

    return {
        "sources_checked": sources,
        "candidates": ranked[:5],
        "evidence": evidence[:20],
        "heuristics": heuristics,
    }


def normalize_best_address_text(best_address: Any) -> Optional[str]:
    if not best_address:
        return None

    if isinstance(best_address, str):
        text = normalize_address_text(best_address)
        return text if text else None

    if isinstance(best_address, dict):
        ordered_keys = [
            "line1", "street", "address",
            "line2",
            "city",
            "state", "state_region", "county", "province",
            "postcode", "postal_code", "zip",
            "country",
        ]
        values = []
        seen = set()
        for key in ordered_keys:
            value = str(best_address.get(key) or "").strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                values.append(value)
        if values:
            return ", ".join(values)

    return None


def parse_address_fields(best_address_text: Optional[str], country_hint: Optional[str]) -> Dict[str, Optional[str]]:
    fields = {
        "line1": None,
        "line2": None,
        "city": None,
        "state_region": None,
        "postcode": None,
        "country": None,
    }

    if not best_address_text:
        if country_hint:
            fields["country"] = country_hint.strip()
        return fields

    text = normalize_address_text(best_address_text)
    lowered = text.lower()
    marker = "registered office address"
    if marker in lowered:
        text = normalize_address_text(text[lowered.index(marker) + len(marker):])
        lowered = text.lower()
    elif "registered address" in lowered:
        marker = "registered address"
        text = normalize_address_text(text[lowered.index(marker) + len(marker):])
        lowered = text.lower()
    if not text:
        if country_hint:
            fields["country"] = country_hint.strip()
        return fields

    postcode_regex = r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|\d{5}(?:-\d{4})?)\b"
    postcode_match = re.search(postcode_regex, text, flags=re.IGNORECASE)
    if postcode_match:
        fields["postcode"] = postcode_match.group(1).upper().strip()

    parts = [p.strip() for p in text.split(",") if p.strip()]
    non_postcode_parts = [p for p in parts if not re.search(postcode_regex, p, flags=re.IGNORECASE)]

    known_countries = {
        "uk": "UK",
        "united kingdom": "UK",
        "gb": "UK",
        "great britain": "UK",
        "england": "UK",
        "scotland": "UK",
        "wales": "UK",
        "northern ireland": "UK",
        "usa": "US",
        "us": "US",
        "united states": "US",
        "united states of america": "US",
    }

    if non_postcode_parts:
        tail = non_postcode_parts[-1].strip().lower()
        if tail in known_countries:
            fields["country"] = known_countries[tail]
            non_postcode_parts = non_postcode_parts[:-1]

    if not fields["country"] and country_hint:
        mapped = known_countries.get(country_hint.strip().lower())
        fields["country"] = mapped or country_hint.strip()

    if non_postcode_parts:
        fields["line1"] = non_postcode_parts[0]

    locality_parts = non_postcode_parts[1:] if len(non_postcode_parts) > 1 else []
    if len(locality_parts) >= 3:
        fields["line2"] = locality_parts[0]
        fields["city"] = locality_parts[1]
        fields["state_region"] = locality_parts[2]
    elif len(locality_parts) == 2:
        fields["city"] = locality_parts[0]
        fields["state_region"] = locality_parts[1]
    elif len(locality_parts) == 1:
        city_state_part = locality_parts[0]
        us_city_state_match = re.search(r"^\s*([A-Za-z][A-Za-z .'-]{1,40})\s+([A-Z]{2})\s*$", city_state_part)
        if us_city_state_match:
            fields["city"] = us_city_state_match.group(1).strip()
            fields["state_region"] = us_city_state_match.group(2).strip()
        else:
            fields["city"] = city_state_part.strip()

    return fields


async def evaluate_business_address_request(request: VerifyBusinessAddressRequest) -> Dict[str, Any]:
    lookup_name = request.company_name or request.domain or "unknown business"
    loop = asyncio.get_event_loop()
    findings = await loop.run_in_executor(
        None,
        verify_business_address_sources,
        lookup_name,
        request.domain,
        request.location,
        request.country,
        request.max_sources,
    )

    ai_assessment = await run_ai_address_verification(
        lookup_name,
        request.location,
        request.country,
        findings.get("evidence", []),
    )

    selected_best_address = (ai_assessment or {}).get("best_address") or findings["heuristics"]["best_address"]
    best_address_text = normalize_best_address_text(selected_best_address)
    best_address_fields = parse_address_fields(best_address_text, request.country)

    # If AI returns a partial location only, prefer the top crawled candidate that includes street detail.
    if (
        best_address_fields.get("line1")
        and not re.search(r"\d", best_address_fields["line1"])
        and findings.get("candidates")
        and isinstance(findings["candidates"], list)
        and len(findings["candidates"]) > 0
    ):
        fallback_address = findings["candidates"][0].get("address")
        fallback_text = normalize_best_address_text(fallback_address)
        fallback_fields = parse_address_fields(fallback_text, request.country)
        if fallback_fields.get("line1") and re.search(r"\d", fallback_fields["line1"]):
            best_address_text = fallback_text
            best_address_fields = fallback_fields

    cleaned_best_address = None
    if best_address_fields.get("line1"):
        cleaned_parts = [
            best_address_fields.get("line1"),
            best_address_fields.get("line2"),
            best_address_fields.get("city"),
            best_address_fields.get("state_region"),
            best_address_fields.get("postcode"),
            best_address_fields.get("country"),
        ]
        cleaned_best_address = ", ".join([part for part in cleaned_parts if part])

    return {
        "company_name": request.company_name or lookup_name,
        "domain": request.domain,
        "location": request.location,
        "country": request.country,
        "verified": bool((ai_assessment or {}).get("verified", findings["heuristics"]["verified"])),
        "best_address": cleaned_best_address or best_address_text,
        "best_address_fields": best_address_fields,
        "confidence_band": (ai_assessment or {}).get("confidence_band") or findings["heuristics"]["confidence_band"],
        "reason": (ai_assessment or {}).get("reason") or findings["heuristics"]["reason"],
        "heuristics": findings["heuristics"],
        "top_candidates": findings["candidates"],
        "sources_checked": findings["sources_checked"],
        "ai_assessment": ai_assessment,
    }


def is_uk_location(location: Optional[str]) -> bool:
    if not location:
        return False
    text = location.strip().lower()
    uk_tokens = ["uk", "u.k", "united kingdom", "great britain", "gb", "england", "scotland", "wales", "northern ireland"]
    return any(token in text for token in uk_tokens)


def discover_companies_house_api(query: str, max_results: int) -> List[Dict[str, str]]:
    if not COMPANIES_HOUSE_API_KEY:
        return []

    try:
        resp = requests.get(
            "https://api.company-information.service.gov.uk/search/companies",
            params={"q": query, "items_per_page": max_results},
            auth=(COMPANIES_HOUSE_API_KEY, ""),
            timeout=20,
            headers={"User-Agent": "B2BEnricher/1.0"},
        )
        if resp.status_code != 200:
            logger.warning(f"Companies House API returned status {resp.status_code}")
            return []

        payload = resp.json()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        results = []
        for item in items[:max_results]:
            title = str(item.get("title") or "").strip()
            company_number = str(item.get("company_number") or "").strip()
            if not title:
                continue
            profile_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}" if company_number else ""
            results.append({
                "title": title,
                "url": profile_url,
                "domain": "find-and-update.company-information.service.gov.uk",
                "company_number": company_number,
                "source": "companies_house_api",
            })
        return results
    except Exception as e:
        logger.warning(f"Companies House API lookup failed: {e}")
        return []


def discover_companies_house_web(query: str, max_results: int) -> List[Dict[str, str]]:
    try:
        resp = requests.get(
            "https://find-and-update.company-information.service.gov.uk/search/companies",
            params={"q": query},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for link in soup.select("a[href^='/company/']"):
            href = link.get("href", "")
            title = link.get_text(" ", strip=True)
            if not href or not title:
                continue

            company_number = href.replace("/company/", "").split("/")[0]
            profile_url = f"https://find-and-update.company-information.service.gov.uk{href}"
            results.append({
                "title": title,
                "url": profile_url,
                "domain": "find-and-update.company-information.service.gov.uk",
                "company_number": company_number,
                "source": "companies_house_web",
            })
            if len(results) >= max_results:
                break

        # De-duplicate by company number/title
        uniq = []
        seen = set()
        for r in results:
            key = (r.get("company_number", ""), r.get("title", "").lower())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        return uniq[:max_results]
    except Exception as e:
        logger.warning(f"Companies House web lookup failed: {e}")
        return []


def companies_house_lookup_by_name(company_name: str) -> Optional[Dict[str, Any]]:
    """
    Lookup a company on Companies House and enrich with active status + directors.
    API is preferred when key exists, else website fallback is used.
    """
    if not company_name:
        return None

    def is_inactive_officer_text(text: str) -> bool:
        low = (text or "").lower()
        inactive_markers = ["role resigned", "resigned on", "resigned director", "ceased", "terminated"]
        return any(m in low for m in inactive_markers)

    def extract_officer_dates(text: str) -> Dict[str, Optional[str]]:
        if not text:
            return {"appointed_on": None, "date_of_birth_month_year": None}

        appointed = None
        dob = None

        m_app = re.search(
            r"appointed on\s+(.+?)(?=\s+(?:date of birth|nationality|country of residence|role|resigned on|$))",
            text,
            flags=re.IGNORECASE,
        )
        if m_app:
            appointed = m_app.group(1).strip()

        m_dob = re.search(
            r"date of birth\s+(.+?)(?=\s+(?:appointed on|nationality|country of residence|role|resigned on|$))",
            text,
            flags=re.IGNORECASE,
        )
        if m_dob:
            dob = m_dob.group(1).strip()

        return {"appointed_on": appointed, "date_of_birth_month_year": dob}

    search_results = discover_companies_house_api(company_name, 1)
    source = "companies_house_api"
    if not search_results:
        search_results = discover_companies_house_web(company_name, 1)
        source = "companies_house_web"

    if not search_results:
        return None

    base = search_results[0]
    company_number = base.get("company_number")
    profile_url = base.get("url")

    if not company_number:
        return {
            "source": source,
            "matched_company_name": base.get("title"),
            "company_number": None,
            "company_status": None,
            "directors": [],
            "registered_office_address": None,
            "incorporation_date": None,
        }

    # API path with richer structured fields
    if source == "companies_house_api" and COMPANIES_HOUSE_API_KEY:
        try:
            profile = requests.get(
                f"https://api.company-information.service.gov.uk/company/{company_number}",
                auth=(COMPANIES_HOUSE_API_KEY, ""),
                timeout=20,
                headers={"User-Agent": "B2BEnricher/1.0"},
            )
            officers = requests.get(
                f"https://api.company-information.service.gov.uk/company/{company_number}/officers",
                params={"items_per_page": 100},
                auth=(COMPANIES_HOUSE_API_KEY, ""),
                timeout=20,
                headers={"User-Agent": "B2BEnricher/1.0"},
            )

            p_json = profile.json() if profile.status_code == 200 else {}
            o_json = officers.json() if officers.status_code == 200 else {}

            director_rows = []
            for item in (o_json.get("items") or []):
                role = str(item.get("officer_role") or "")
                resigned_on = item.get("resigned_on")
                ceased_on = item.get("ceased_on")
                if "director" in role.lower() and not resigned_on and not ceased_on:
                    dob_obj = item.get("date_of_birth") if isinstance(item.get("date_of_birth"), dict) else {}
                    dob_month = dob_obj.get("month")
                    dob_year = dob_obj.get("year")
                    dob_text = None
                    if dob_month and dob_year:
                        dob_text = f"{dob_month}/{dob_year}"
                    director_rows.append({
                        "name": str(item.get("name") or "").strip(),
                        "title": "Director",
                        "appointed_on": item.get("appointed_on"),
                        "date_of_birth_month_year": dob_text,
                    })

            address = p_json.get("registered_office_address") if isinstance(p_json, dict) else None
            return {
                "source": source,
                "matched_company_name": p_json.get("company_name") or base.get("title"),
                "company_number": company_number,
                "company_status": p_json.get("company_status"),
                "directors": [d for d in director_rows if d.get("name")],
                "registered_office_address": address,
                "incorporation_date": p_json.get("date_of_creation"),
                "profile_url": profile_url,
            }
        except Exception as e:
            logger.warning(f"Companies House API profile lookup failed: {e}")

    # Website fallback path
    try:
        prof_resp = requests.get(profile_url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"})
        prof_resp.raise_for_status()
        soup = BeautifulSoup(prof_resp.text, "html.parser")

        status = None
        inc_date = None
        registered_office_address = None
        for dt in soup.select("dt"):
            key = dt.get_text(" ", strip=True).lower()
            dd = dt.find_next_sibling("dd")
            value = dd.get_text(" ", strip=True) if dd else None
            if key == "company status":
                status = value
            if key == "incorporated on":
                inc_date = value
            if key == "registered office address":
                registered_office_address = value

        if not registered_office_address:
            office_node = soup.select_one("#company-registered-office-address, .registered-office-address, [data-id='company-registered-office-address']")
            if office_node:
                registered_office_address = office_node.get_text(" ", strip=True)

        # Prefer the explicit People/Officers page if available from profile.
        officers_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}/officers"
        people_link = soup.select_one(f"a[href*='/company/{company_number}/officers']")
        if people_link and people_link.get("href"):
            href = people_link.get("href")
            if href.startswith("http"):
                officers_url = href
            else:
                officers_url = f"https://find-and-update.company-information.service.gov.uk{href}"

        off_resp = requests.get(
            officers_url,
            params={"items_per_page": 100},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
        )
        director_rows = []
        if off_resp.status_code == 200:
            off_soup = BeautifulSoup(off_resp.text, "html.parser")
            cards = off_soup.select("li.appointments-list__item, article.appointments-list__item, .appointments-list__item")
            for card in cards:
                text = card.get_text(" ", strip=True)
                low_text = text.lower()
                if "director" in low_text and not is_inactive_officer_text(low_text):
                    name_el = card.select_one("h2, h3, a, span")
                    name = name_el.get_text(" ", strip=True) if name_el else ""
                    if name:
                        date_parts = extract_officer_dates(text)
                        director_rows.append({
                            "name": name,
                            "title": "Director",
                            "appointed_on": date_parts["appointed_on"],
                            "date_of_birth_month_year": date_parts["date_of_birth_month_year"],
                        })

            # Fallback parse for people links if card structure differs.
            if not director_rows:
                for link in off_soup.select("a[href*='/officers/']"):
                    container = link.find_parent("div", class_=re.compile(r"appointment-"))
                    row_text = container.get_text(" ", strip=True).lower() if container else ""
                    is_active_director = (
                        "role active director" in row_text
                        or ("director" in row_text and "role active" in row_text)
                    )
                    if is_active_director and not is_inactive_officer_text(row_text):
                        name = link.get_text(" ", strip=True)
                        if name:
                            date_parts = extract_officer_dates(row_text)
                            director_rows.append({
                                "name": name,
                                "title": "Director",
                                "appointed_on": date_parts["appointed_on"],
                                "date_of_birth_month_year": date_parts["date_of_birth_month_year"],
                            })

            # Some Companies House layouts only show officer links; fetch each officer page
            # and confirm role there.
            if not director_rows:
                for link in off_soup.select("a[href*='/officers/']"):
                    href = link.get("href", "")
                    if not href:
                        continue
                    name = link.get_text(" ", strip=True)
                    if not name:
                        continue

                    detail_url = href if href.startswith("http") else f"https://find-and-update.company-information.service.gov.uk{href}"
                    try:
                        detail_resp = requests.get(
                            detail_url,
                            timeout=20,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
                        )
                        if detail_resp.status_code != 200:
                            continue
                        detail_text = BeautifulSoup(detail_resp.text, "html.parser").get_text(" ", strip=True).lower()
                        if "director" in detail_text and not is_inactive_officer_text(detail_text):
                            date_parts = extract_officer_dates(detail_text)
                            director_rows.append({
                                "name": name,
                                "title": "Director",
                                "appointed_on": date_parts["appointed_on"],
                                "date_of_birth_month_year": date_parts["date_of_birth_month_year"],
                            })
                    except Exception:
                        continue

        return {
            "source": source,
            "matched_company_name": base.get("title"),
            "company_number": company_number,
            "company_status": status,
            "directors": director_rows,
            "registered_office_address": registered_office_address,
            "incorporation_date": inc_date,
            "profile_url": profile_url,
        }
    except Exception as e:
        logger.warning(f"Companies House web profile lookup failed: {e}")
        return {
            "source": source,
            "matched_company_name": base.get("title"),
            "company_number": company_number,
            "company_status": None,
            "directors": [],
            "registered_office_address": None,
            "incorporation_date": None,
            "profile_url": profile_url,
        }

# --- API Endpoints ---

@app.get("/health", tags=["System"])
async def health_check():
    """Returns API status and active AI model name."""
    return {"status": "healthy", "model": AI_MODEL}

@app.get("/system-info", tags=["System"])
async def system_info():
    """Returns active AI model, loaded Ollama models, and operational status."""
    # Basic check if Ollama is responsive
    try:
        models = ollama.list()
        return {
            "active_model": AI_MODEL,
            "ollama_models": [m['model'] for m in models.get('models', [])],
            "status": "Operational"
        }
    except Exception as e:
        return {"status": "Error", "detail": str(e)}

@app.post("/enrich", tags=["Enrichment"])
async def enrich_single(request: EnrichRequest):
    """
    Enrich a single business by domain, company_name, linkedin_url, or free-text query.

    - **domain** — crawls the website directly for AI extraction.
    - **company_name** (no domain) — discovers up to `max_results` URLs via search, enriches all, and returns an `overall_summary`.
    - **query** — delegates to /crawl-businesses for multi-result discovery + enrichment.
    - Phones are verified via libphonenumber and cross-referenced across sources.
    - UK companies are automatically looked up on Companies House for directors, status, and registered address.
    """
    if request.query:
        return await crawl_businesses(
            CrawlBusinessesRequest(
                query=request.query,
                location=request.location,
                max_results=request.max_results,
            )
        )

    target_url = None

    if request.domain:
        target_url = request.domain if request.domain.startswith("http") else f"https://{request.domain}"
    elif request.linkedin_url:
        target_url = request.linkedin_url

    # 1. Async Crawl
    # We run crawl in executor too as requests/trafilatura is sync
    loop = asyncio.get_event_loop()

    # Fallback discovery when domain is not provided — search max_results domains.
    if not target_url and request.company_name:
        discovery_location = request.location or request.country
        candidates = await loop.run_in_executor(
            None,
            discover_business_urls,
            request.company_name,
            discovery_location,
            request.max_results,
        )
        if not candidates:
            raise HTTPException(status_code=400, detail="Could not discover any URLs for the given company_name.")

        # Multiple candidates discovered — enrich all of them like crawl-businesses.
        if len(candidates) > 1:
            return await crawl_businesses(
                CrawlBusinessesRequest(
                    query=request.company_name,
                    location=discovery_location,
                    max_results=request.max_results,
                )
            )

        target_url = candidates[0].get("url")

    if not target_url:
        raise HTTPException(status_code=400, detail="Could not resolve a target URL. Provide domain/linkedin_url or a company_name that can be discovered.")

    text_content = await loop.run_in_executor(None, clean_text_from_url, target_url)
    
    if not text_content:
        raise HTTPException(status_code=400, detail="Could not extract content from resolved URL")

    # 2. Async AI with hints
    hints = {
        "company_name": request.company_name or "",
        "phone_number": request.phone_number or "",
        "location": request.location or "",
        "country": request.country or "",
        "industry_hint": request.industry_hint or "",
        "linkedin_url": request.linkedin_url or "",
        "additional_context": request.additional_context or "",
    }
    ai_data = await run_ai_extraction(text_content, hints=hints)
    
    if not ai_data:
        ai_data = {"error": "AI processing failed"}
        confidence = {"overall": 0, "band": "low", "reasons": ["ai_failed"], "signals": {"email_count": 0, "phone_count": 0}}
    else:
        confidence = score_enrichment(ai_data, hints, text_content)

    return {
        "domain": request.domain,
        "resolved_url": target_url,
        "enrichment": ai_data,
        "confidence": confidence,
        "processing_model": AI_MODEL
    }

@app.post("/batch-enrich", tags=["Enrichment"])
async def batch_enrich(request: BatchEnrichRequest):
    """
    Enrich multiple domains in parallel using asyncio.gather.

    Accepts a list of domains and returns AI-extracted data + confidence scores for each.
    """
    async def process_one(domain: str):
        url = domain if domain.startswith("http") else f"https://{domain}"
        text = await asyncio.get_event_loop().run_in_executor(None, clean_text_from_url, url)
        if text:
            data = await run_ai_extraction(text, hints={})
            confidence = score_enrichment(data or {}, {}, text) if data else {"overall": 0, "band": "low", "reasons": ["ai_failed"], "signals": {"email_count": 0, "phone_count": 0}}
            return {"domain": domain, "data": data, "confidence": confidence, "status": "success"}
        return {"domain": domain, "data": None, "status": "crawl_failed"}

    # Create tasks for all domains
    tasks = [process_one(d) for d in request.domains]
    
    # Run in parallel
    results = await asyncio.gather(*tasks)
    
    return {
        "total": len(request.domains),
        "processed": len(results),
        "results": results
    }

@app.post("/verify-email", tags=["Verification"])
async def verify_email_endpoint(request: VerifyEmailRequest):
    """Validate an email address via format check, DNS MX lookup, and SMTP handshake."""
    email = request.email
    domain = email.split('@')[1]
    
    loop = asyncio.get_event_loop()
    
    # Run DNS/SMTP checks in thread to not block
    mx_records = await loop.run_in_executor(None, get_mx_records, domain)
    dns_valid = len(mx_records) > 0
    
    smtp_valid, smtp_msg = await loop.run_in_executor(None, check_smtp_connection, email)
    
    status = "valid" if dns_valid else "invalid"
    if smtp_valid: status = "verified"
    
    return {
        "email": email,
        "dns_valid": dns_valid,
        "mx_records": mx_records[:3], # Limit output
        "smtp_status": smtp_msg
    }

@app.post("/verify-phone", tags=["Verification"])
async def verify_phone_endpoint(request: VerifyPhoneRequest):
    """Validate a phone number offline using libphonenumber. Returns validity, type, E.164, and international format."""
    try:
        parsed = phonenumbers.parse(request.phone_number, request.country_code)
        return {
            "is_valid": phonenumbers.is_valid_number(parsed),
            "type": phonenumbers.number_type(parsed),
            "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/verify-business-address", tags=["Verification"])
async def verify_business_address_endpoint(request: VerifyBusinessAddressRequest):
    """
    Verify a business address by crawling the official website, public directories, and registries.

    Extracts address candidates, scores them by source authority, and uses AI to select the best match.
    Returns structured address fields (line1, city, postcode, country) plus source evidence.
    """
    return await evaluate_business_address_request(request)


@app.post("/batch-verify-business-address", tags=["Verification"])
async def batch_verify_business_address_endpoint(request: BatchVerifyBusinessAddressRequest):
    """
    Verify addresses for multiple companies in parallel.

    Each item is independently crawled and verified. Returns per-item results plus a verified count.
    """
    tasks = [evaluate_business_address_request(item) for item in request.items]
    results = await asyncio.gather(*tasks)
    verified_count = sum(1 for item in results if item.get("verified"))

    return {
        "total": len(request.items),
        "verified": verified_count,
        "results": results,
    }


@app.post("/crawl-businesses", tags=["Discovery"])
async def crawl_businesses(request: CrawlBusinessesRequest):
    """
    Discover and enrich multiple businesses from a search query.

    1. Searches DuckDuckGo for candidate websites matching `query` + `location`.
    2. Crawls each candidate and runs AI extraction + Companies House lookup in parallel.
    3. Phones are verified via libphonenumber and cross-referenced (official site or 2+ sources required).
    4. Returns per-result enrichment, `domains_scanned` list, and a cross-referenced `overall_summary`.
    """
    try:
        candidates = await asyncio.get_event_loop().run_in_executor(
            None,
            discover_business_urls,
            request.query,
            request.location,
            request.max_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {e}")

    async def enrich_candidate(item: Dict[str, str]):
        text = await asyncio.get_event_loop().run_in_executor(None, clean_text_from_url, item["url"])
        if not text:
            return {
                "title": item["title"],
                "url": item["url"],
                "domain": item["domain"],
                "status": "crawl_failed",
            }

        hints = {
            "discovery_query": request.query,
            "location": request.location or "",
        }
        # Run AI extraction and Companies House lookup in parallel for speed.
        ai_task = run_ai_extraction(text, hints=hints)
        ch_task = None
        if is_uk_location(request.location):
            # Use query/company intent, not result title, to avoid mismatching
            # similarly named companies from directory pages.
            lookup_name = request.query
            ch_task = asyncio.get_event_loop().run_in_executor(
                None,
                companies_house_lookup_by_name,
                lookup_name,
            )

        ai_data = await ai_task
        companies_house = (await ch_task) if ch_task else None

        if companies_house and isinstance(companies_house, dict):
            # Merge Companies House directors into enrichment if model missed them.
            if isinstance(ai_data, dict) and isinstance(companies_house.get("directors"), list):
                if not ai_data.get("directors"):
                    ai_data["directors"] = companies_house.get("directors")

        # Only extract phones/emails from the company's own website.
        # Directory/aggregator pages list many businesses so their contact
        # details would pollute the results with unrelated numbers.
        aggregator_domains = {
            "efinder.uk", "opengovuk.com", "checkcompany.co.uk",
            "find-and-update.company-information.service.gov.uk",
            "endole.co.uk", "companieslist.co.uk", "companycheck.co.uk",
            "dnb.com", "duedil.com", "opencorporates.com",
            "companiesintheuk.co.uk", "ukdata.com", "192.com",
            "cylex-uk.co.uk", "brownbook.net", "hotfrog.co.uk",
            "scoot.co.uk", "thomsonlocal.com", "yell.com",
            "yelp.com", "yelp.co.uk", "trustpilot.com",
            "glassdoor.co.uk", "glassdoor.com", "indeed.co.uk",
            "linkedin.com", "facebook.com", "twitter.com",
            "instagram.com", "tiktok.com", "pinterest.com",
            "crunchbase.com", "zoominfo.com",
        }
        candidate_domain = item.get("domain", "")
        is_aggregator = any(candidate_domain == agg or candidate_domain.endswith("." + agg) for agg in aggregator_domains)

        # Avoid confidence inflation from directory pages that list many phone/email snippets.
        confidence_text = "" if is_aggregator else text
        confidence = score_enrichment(ai_data or {}, hints, confidence_text) if ai_data else {"overall": 0, "band": "low", "reasons": ["ai_failed"], "signals": {"email_count": 0, "phone_count": 0}}

        contact_details = extract_contact_details(text, country_hint=request.location) if not is_aggregator else {"emails": [], "phones": []}

        return {
            "title": item["title"],
            "url": item["url"],
            "domain": item["domain"],
            "source": item.get("source", "web"),
            "is_aggregator": is_aggregator,
            "status": "success" if ai_data else "ai_failed",
            "enrichment": ai_data,
            "companies_house": companies_house,
            "contact_details": contact_details,
            "confidence": confidence,
        }

    tasks = [enrich_candidate(item) for item in candidates]
    results = await asyncio.gather(*tasks)
    domains_scanned = [
        {"domain": r.get("domain"), "url": r.get("url"), "title": r.get("title"), "status": r.get("status")}
        for r in results
    ]
    overall_summary = build_overall_b2b_summary(request.query, request.location, results, domains_scanned=domains_scanned)

    # Run address verification to get a proper verified address with postcode.
    summary_company = overall_summary.get("company_name") or request.query
    summary_domain = None
    for item in results:
        if item.get("status") == "success" and not item.get("is_aggregator"):
            summary_domain = item.get("domain")
            break
    try:
        addr_result = await evaluate_business_address_request(
            VerifyBusinessAddressRequest(
                company_name=summary_company,
                domain=summary_domain,
                location=request.location,
            )
        )
        if addr_result and addr_result.get("verified"):
            candidate_verified_addr = addr_result.get("best_address") or ""
            candidate_verified_fields = dict(addr_result.get("best_address_fields") or {})
            current_addr = overall_summary.get("address") or ""
            current_addr_fields = parse_address_fields(current_addr, request.location) if current_addr else {}
            same_postcode = False
            if current_addr_fields and candidate_verified_fields:
                same_postcode = (
                    (candidate_verified_fields.get("postcode") or "").strip().upper()
                    and (current_addr_fields.get("postcode") or "").strip().upper()
                    and (candidate_verified_fields.get("postcode") or "").strip().upper() == (current_addr_fields.get("postcode") or "").strip().upper()
                )

            # If verifier returns a partial address (e.g. only line1+postcode),
            # enrich missing locality fields from the richer cross-referenced address.
            if candidate_verified_fields and current_addr:
                fallback_fields = current_addr_fields
                if same_postcode:
                    for key in ["city", "state_region", "country"]:
                        if not candidate_verified_fields.get(key) and fallback_fields.get(key):
                            candidate_verified_fields[key] = fallback_fields.get(key)

                    rebuilt_parts = [
                        candidate_verified_fields.get("line1"),
                        candidate_verified_fields.get("line2"),
                        candidate_verified_fields.get("city"),
                        candidate_verified_fields.get("state_region"),
                        candidate_verified_fields.get("postcode"),
                        candidate_verified_fields.get("country"),
                    ]
                    rebuilt_addr = ", ".join([part for part in rebuilt_parts if part])
                    if rebuilt_addr:
                        candidate_verified_addr = rebuilt_addr

            # If the verifier points to a different postcode than the current summary
            # address, do not surface it as a verified address. This avoids replacing
            # a stronger authoritative address (for example Companies House) with a
            # different office/location found on the public website.
            if current_addr and has_postcode_pattern(current_addr) and candidate_verified_addr and not same_postcode:
                candidate_verified_addr = ""
                candidate_verified_fields = {}

            has_field_value = any(bool(v) for v in candidate_verified_fields.values()) if candidate_verified_fields else False
            if candidate_verified_addr or has_field_value:
                overall_summary["verified_address"] = candidate_verified_addr or None
                overall_summary["verified_address_fields"] = candidate_verified_fields if has_field_value else None
            # Only upgrade the summary address when there is no meaningful current address,
            # or when both addresses point to the same postcode.
            verified_addr = candidate_verified_addr or ""
            if (not current_addr and verified_addr) or (not has_postcode_pattern(current_addr) and has_postcode_pattern(verified_addr)) or same_postcode:
                overall_summary["address"] = verified_addr
    except Exception as e:
        logger.warning(f"Address verification in enrichment failed: {e}")

    return {
        "query": request.query,
        "location": request.location,
        "discovered": len(candidates),
        "domains_scanned": domains_scanned,
        "results": results,
        "processing_model": AI_MODEL,
        "overall_summary": overall_summary,
    }

PYEOF

# 8. Service Configuration
# Check if systemd is available
if pidof systemd &> /dev/null; then
    echo "[*] Configuring Systemd Service..."
    configure_ollama_systemd_override

    # Free target port if already in use
    PORT_PIDS=$(sudo lsof -t -iTCP:${APP_PORT} -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "${PORT_PIDS}" ]; then
        echo "[!] Port ${APP_PORT} is in use. Stopping PID(s): ${PORT_PIDS}"
        echo "${PORT_PIDS}" | xargs -r sudo kill -9
    fi

    cat > /etc/systemd/system/ai-enrichment.service << SVCEOF
[Unit]
Description=AI Data Enrichment FastAPI Service
After=network.target ollama.service

[Service]
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR/app
Environment="PATH=$VENV_DIR/bin"
Environment="AI_MODEL_NAME=$SELECTED_MODEL"
Environment="COMPANIES_HOUSE_API_KEY=${COMPANIES_HOUSE_API_KEY:-}"
Environment="OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
Environment="OLLAMA_KEEP_ALIVE=-1"
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port $APP_PORT --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

    # 9. Start Service
    echo "[*] Starting Service..."
    # Ensure Ollama service is enabled/running when systemd is available
    sudo systemctl enable ollama 2>/dev/null || true
    sudo systemctl restart ollama 2>/dev/null || true
    sudo systemctl daemon-reload
    sudo systemctl enable ai-enrichment.service
    sudo systemctl restart ai-enrichment.service
else
    echo "[!] Systemd not available. Starting service directly..."
    ensure_ollama_running
    # Free target port if already in use
    PORT_PIDS=$(sudo lsof -t -iTCP:${APP_PORT} -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "${PORT_PIDS}" ]; then
        echo "[!] Port ${APP_PORT} is in use. Stopping PID(s): ${PORT_PIDS}"
        echo "${PORT_PIDS}" | xargs -r sudo kill -9
    fi
    # Kill any existing instance
    pkill -f "uvicorn main:app" 2>/dev/null || true
    cd $APP_DIR/app
    export AI_MODEL_NAME=$SELECTED_MODEL
    export COMPANIES_HOUSE_API_KEY=${COMPANIES_HOUSE_API_KEY:-}
    export OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL
    export OLLAMA_SCHED_SPREAD=$OLLAMA_SCHED_SPREAD
    export OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS
    nohup $VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port $APP_PORT --workers 1 > /var/log/ai-enrichment.log 2>&1 &
    echo "[+] Service started (PID: $!). Log: /var/log/ai-enrichment.log"
fi

echo "------------------------------------------------"
echo " Installation Complete!"
echo "------------------------------------------------"
echo "Hardware Detected:"
echo "  GPUs: $GPU_COUNT"
echo "  VRAM: ${TOTAL_VRAM_MB}MB"
echo "  Multi-GPU: $MULTI_GPU_MODE"
echo "  Disk: ${AVAIL_DISK_GB}GB"
echo ""
echo "Selected AI Model: $SELECTED_MODEL"
echo ""
echo "API Endpoint: http://$(hostname -I | awk '{print $1}'):${APP_PORT}"
echo "Docs:        http://$(hostname -I | awk '{print $1}'):${APP_PORT}/docs"
echo "------------------------------------------------"
