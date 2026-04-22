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
# mistral-small:24b (14GB) - Better quality for high-end multi-GPU hosts
# llama3.2:1b (1.3GB) - Best for low VRAM/CPU/Low Disk
# phi3:mini (2.2GB)   - Alternative for constrained envs

SELECTED_MODEL=""
MODEL_NAME=""

# Logic Tree
if [ "$AVAIL_DISK_GB" -lt 6 ]; then
    # Critical low space
    SELECTED_MODEL="llama3.2:1b"
    echo "[!] Critical Disk Space. Selecting tiny model: $SELECTED_MODEL"
elif [ "$HAS_NVIDIA" = true ] && [ "$TOTAL_VRAM_MB" -gt 45000 ] && [ "$AVAIL_DISK_GB" -gt 20 ]; then
    # High-end multi-GPU box with enough disk for a larger model.
    SELECTED_MODEL="mistral-small:24b"
    echo "[+] High-end GPU host detected. Selecting larger model: $SELECTED_MODEL"
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
import html as html_lib
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
AI_MODEL = os.getenv("AI_MODEL_NAME", "mistral-small:24b")
# Task-specific local models. Default to AI_MODEL when not overridden so the
# system works out of the box, but operators can point each specialist task at
# a smaller/faster model (e.g. AI_MODEL_MATCHER=llama3.1:8b for entity match,
# AI_MODEL_SUMMARIZER=mistral-small:24b for narrative output).
AI_MODEL_MATCHER = os.getenv("AI_MODEL_MATCHER", AI_MODEL)        # entity / name matching
AI_MODEL_VALIDATOR = os.getenv("AI_MODEL_VALIDATOR", AI_MODEL)    # website-to-company validation
AI_MODEL_ADDRESS = os.getenv("AI_MODEL_ADDRESS", AI_MODEL)        # address comparison / verification
AI_MODEL_CLASSIFIER = os.getenv("AI_MODEL_CLASSIFIER", AI_MODEL)  # industry / sector classification
AI_MODEL_SUMMARIZER = os.getenv("AI_MODEL_SUMMARIZER", AI_MODEL)  # final business activity summary
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
    version="2.7.0"
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


def fetch_html_from_url(url: str, allow_non_html: bool = False) -> Optional[str]:
    try:
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; B2BEnricher/1.0)"},
        )
        resp.raise_for_status()
        if not allow_non_html and "text/html" not in resp.headers.get("content-type", "").lower():
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
    companies_house_address_candidates: List[str] = []
    for item in (trusted_companies_house_hits or companies_house_hits):
        ch = item.get("companies_house") or {}
        formatted = format_registered_office_address(ch.get("registered_office_address"))
        if formatted:
            companies_house_address_candidates.append(formatted)
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

    verified_address = None
    verified_address_fields = None
    if address and companies_house_address_candidates:
        selected_fields = parse_address_fields(address, location)
        selected_postcode = (selected_fields.get("postcode") or "").strip().upper()
        selected_line1 = slugify_text(selected_fields.get("line1") or "")
        for candidate in companies_house_address_candidates:
            candidate_fields = parse_address_fields(candidate, location)
            candidate_postcode = (candidate_fields.get("postcode") or "").strip().upper()
            candidate_line1 = slugify_text(candidate_fields.get("line1") or "")
            same_postcode = bool(selected_postcode and candidate_postcode and selected_postcode == candidate_postcode)
            same_line1 = bool(selected_line1 and candidate_line1 and selected_line1 == candidate_line1)
            if same_postcode or same_line1 or slugify_text(candidate) == slugify_text(address):
                verified_address = address
                verified_address_fields = selected_fields if any(bool(v) for v in selected_fields.values()) else None
                break

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
        "verified_address": verified_address,
        "verified_address_fields": verified_address_fields,
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

    blocked_domains = {
        "bing.com", "www.bing.com", "duckduckgo.com", "www.duckduckgo.com",
        "search.yahoo.com", "yahoo.com", "www.yahoo.com",
    }
    candidates: List[Dict[str, str]] = []
    seen_domains: set = set()

    # Two-pass query strategy:
    # 1) original query (can include phone)
    # 2) cleaned company-name query (phone removed) to avoid reverse-phone/ad spam
    query_variants: List[str] = [search_query]
    cleaned = strip_phones_from_text(search_query)
    if cleaned and cleaned.lower() != search_query.lower():
        query_variants.append(cleaned)

    for q in query_variants:
        for item in search_public_results(q, max_results * 3):
            domain = (item.get("domain") or "").lower()
            url = item.get("url") or ""
            if not domain or not url:
                continue
            if domain in blocked_domains or domain.endswith(".bing.com") or domain.endswith(".yahoo.com"):
                continue
            if "/aclick" in url.lower() or "trafficguard.ai" in url.lower():
                continue
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            candidates.append(item)
            if len(candidates) >= max_results:
                return candidates
    return candidates


def search_public_results(query: str, max_results: int) -> List[Dict[str, str]]:
    """Search public web results, falling back across providers when one is
    rate-limited or returns nothing. Currently: DuckDuckGo HTML → Yahoo HTML
    redirects → Bing HTML.
    """
    results = _search_duckduckgo(query, max_results)
    if not results:
        results = _search_yahoo(query, max_results)
    if not results:
        results = _search_bing(query, max_results)
    return results


def _search_duckduckgo(query: str, max_results: int) -> List[Dict[str, str]]:
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query[:60]}': {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []
    seen: set = set()
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


def _search_bing(query: str, max_results: int) -> List[Dict[str, str]]:
    """Bing HTML fallback. Used when DDG is rate-limited or empty."""
    try:
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max(10, max_results)},
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Bing search failed for '{query[:60]}': {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []
    seen: set = set()
    blocked_domains = {"bing.com", "www.bing.com", "r.bing.com", "cn.bing.com"}
    # Bing wraps each organic hit in <li class="b_algo"> with an <h2><a href=...>.
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a") or li.select_one("a")
        if not a:
            continue
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        # Exclude Bing ad-click and tracking URLs (aclick/ck/a) that are not
        # organic destination pages.
        href_lower = href.lower()
        if "bing.com/aclick" in href_lower or "bing.com/ck/a" in href_lower:
            continue
        # Bing sometimes wraps URLs in a redirect; the visible href is usually direct.
        key = href.lower().strip()
        if key in seen:
            continue
        domain = normalize_domain(href)
        if not domain or domain in blocked_domains or domain.endswith(".bing.com"):
            continue
        seen.add(key)
        results.append({
            "title": a.get_text(" ", strip=True),
            "url": href,
            "domain": domain,
        })
        if len(results) >= max_results:
            break
    return results


def _parse_yahoo_result_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    match = re.search(r"/RU=([^/]+)/", raw_url)
    if match:
        return unquote(match.group(1))
    return raw_url if raw_url.startswith("http") else ""


def _search_yahoo(query: str, max_results: int) -> List[Dict[str, str]]:
    """Yahoo HTML fallback.

    Yahoo exposes organic results as redirect URLs containing `/RU=<encoded>`.
    That works reliably on hosts where DDG/Bing strip all extractable anchors.
    """
    try:
        resp = requests.get(
            "https://search.yahoo.com/search",
            params={"p": query},
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Yahoo search failed for '{query[:60]}': {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []
    seen: set = set()
    blocked_domains = {
        "yahoo.com", "us.mail.yahoo.com", "finance.yahoo.com", "sports.yahoo.com",
        "shopping.yahoo.com", "guce.yahoo.com", "help.yahoo.com",
        "advertising.yahoo.com", "search.yahoo.com",
    }

    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if "r.search.yahoo.com" not in href:
            continue
        real_url = _parse_yahoo_result_url(href)
        if not real_url:
            continue
        domain = normalize_domain(real_url)
        if not domain or domain in blocked_domains or domain.endswith(".yahoo.com"):
            continue
        key = real_url.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "title": a.get_text(" ", strip=True),
            "url": real_url,
            "domain": domain,
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
    # Strip "Companies House default address" placeholder segments. Some CH
    # records include "12345678 - COMPANIES HOUSE DEFAULT ADDRESS" as a comma
    # part which would otherwise be treated as the city/region.
    def _is_default_address_segment(seg: str) -> bool:
        s = seg.strip().lower()
        if "default address" in s or "companies house default" in s:
            return True
        if re.match(r"^\d{6,}\s*[-–]\s*", s):
            return True
        return False
    parts = [p for p in parts if not _is_default_address_segment(p)]
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


# =============================================================================
# Verified B2B Record Builder
#
# Produces an explainable, CRM-ready enrichment record with per-field
# provenance, confidence, mismatch warnings, and a final business summary.
# Companies House is treated as the authoritative source where available;
# website-derived data must pass a name/address/contact match check before
# fields are accepted.
# =============================================================================

# Per-source confidence weights (0-100). Higher = more authoritative.
SOURCE_CONFIDENCE = {
    "companies_house": 95,
    "official_website": 80,
    "user_query+libphonenumber": 80,  # phone supplied by user, validated offline
    "cross_referenced": 78,    # phone/address seen on 2+ independent sites
    "linkedin": 60,
    "yelp": 55,
    "google": 55,
    "directory": 45,
    "ai_inferred": 35,
    "discovery": 30,
    "unknown": 20,
}


def field_record(value: Any,
                 source: str,
                 confidence: int,
                 alternatives: Optional[List[Any]] = None,
                 notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Wrap a value with provenance metadata for the final enrichment record."""
    return {
        "value": value,
        "source": source,
        "confidence": int(max(0, min(100, confidence))),
        "alternatives": alternatives or [],
        "notes": notes or [],
    }


def normalized_postcode(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").upper())


def address_match_signals(ch_address: Optional[str],
                          candidate_address: Optional[str],
                          country_hint: Optional[str]) -> Dict[str, Any]:
    """Compare a Companies House address against another address string.

    Returns match flags (postcode/line1/locality) plus a 0-100 score.
    Score thresholds: >=80 strong match, 50-79 partial, <50 weak/conflict.
    """
    if not ch_address or not candidate_address:
        return {"score": 0, "same_postcode": False, "same_line1": False,
                "same_city": False, "conflict": False}

    ch_fields = parse_address_fields(ch_address, country_hint)
    cand_fields = parse_address_fields(candidate_address, country_hint)

    ch_pc = normalized_postcode(ch_fields.get("postcode") or "")
    cand_pc = normalized_postcode(cand_fields.get("postcode") or "")
    same_postcode = bool(ch_pc and cand_pc and ch_pc == cand_pc)
    pc_conflict = bool(ch_pc and cand_pc and ch_pc != cand_pc)

    ch_line1 = slugify_text(ch_fields.get("line1") or "")
    cand_line1 = slugify_text(cand_fields.get("line1") or "")
    same_line1 = bool(ch_line1 and cand_line1 and (ch_line1 == cand_line1
                                                    or ch_line1 in cand_line1
                                                    or cand_line1 in ch_line1))

    ch_city = (ch_fields.get("city") or "").strip().lower()
    cand_city = (cand_fields.get("city") or "").strip().lower()
    same_city = bool(ch_city and cand_city and ch_city == cand_city)

    score = 0
    if same_postcode:
        score += 60
    if same_line1:
        score += 25
    if same_city:
        score += 15
    if not (same_postcode or same_line1) and same_city:
        score = max(score, 20)

    return {
        "score": min(100, score),
        "same_postcode": same_postcode,
        "same_line1": same_line1,
        "same_city": same_city,
        "conflict": pc_conflict and not same_line1,
    }


def website_company_match_score(ch_record: Optional[Dict[str, Any]],
                                website_text: str,
                                website_domain: str,
                                ai_data: Optional[Dict[str, Any]],
                                contact_details: Optional[Dict[str, Any]],
                                country_hint: Optional[str],
                                query_phones: Optional[List[Dict[str, Any]]] = None,
                                registered_office_is_proxy: bool = False) -> Dict[str, Any]:
    """Score how strongly a website appears to belong to a Companies House record.

    Combines: domain↔name token overlap, address presence on page, postcode
    co-occurrence, director surname mentions, company-number / "Registered in"
    markers, and (when supplied) the presence of the user-supplied phone
    number on the page. Returns score 0-100.

    When ``registered_office_is_proxy`` is True the CH address is a shared
    registered-office / PO Box (e.g. a formations agent), so postcode and
    line-1 matches are NOT rewarded — those signals would otherwise promote
    every business that uses the same registered office.
    """
    if not website_domain or not website_text:
        return {"score": 0, "matches": [], "mismatches": ["no_website_text"], "signals": {}}

    text_lower = website_text.lower()
    domain_slug = slugify_text(website_domain.split(".")[0])
    matches: List[str] = []
    mismatches: List[str] = []
    signals: Dict[str, Any] = {}
    score = 0

    ch = ch_record or {}
    ch_name = str(ch.get("matched_company_name") or "").strip()
    name_tokens = company_name_tokens(ch_name) if ch_name else []
    primary_token = name_tokens[0] if name_tokens else ""

    # 1) Domain ↔ company name overlap
    domain_token_hits = [t for t in name_tokens if t and t in domain_slug]
    if domain_token_hits:
        score += 30
        matches.append(f"domain_contains_name_token({','.join(domain_token_hits)})")
        signals["domain_token_hits"] = domain_token_hits
    elif primary_token:
        mismatches.append("domain_missing_company_name")

    # 2) Page mentions company name
    if ch_name and ch_name.lower() in text_lower:
        score += 15
        matches.append("page_mentions_legal_name")
    elif name_tokens and sum(1 for t in name_tokens if t in text_lower) >= max(1, len(name_tokens) // 2):
        score += 8
        matches.append("page_mentions_name_tokens")

    # 3) Address signals
    ch_addr = format_registered_office_address(ch.get("registered_office_address"))
    if ch_addr:
        ch_fields = parse_address_fields(ch_addr, country_hint)
        ch_pc = normalized_postcode(ch_fields.get("postcode") or "")
        ch_line1_slug = slugify_text(ch_fields.get("line1") or "")

        # Find any postcode-shaped tokens on the page.
        page_postcodes = set()
        for m in re.finditer(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", website_text, flags=re.IGNORECASE):
            page_postcodes.add(normalized_postcode(m.group(0)))
        for m in re.finditer(r"\b\d{5}(?:-\d{4})?\b", website_text):
            page_postcodes.add(normalized_postcode(m.group(0)))
        signals["page_postcodes"] = sorted(page_postcodes)[:8]

        if registered_office_is_proxy:
            signals["registered_office_is_proxy"] = True
            mismatches.append("ch_address_is_proxy_skipped_for_scoring")
        else:
            if ch_pc and ch_pc in page_postcodes:
                score += 25
                matches.append("postcode_matches_companies_house")
            elif ch_pc and page_postcodes:
                mismatches.append(f"page_postcode_differs(ch={ch_pc},page={','.join(sorted(page_postcodes))[:60]})")

            if ch_line1_slug:
                slug_text = slugify_text(website_text[:20000])
                if ch_line1_slug in slug_text:
                    score += 15
                    matches.append("address_line1_on_page")

    # 3.5) Query-phone presence — extremely strong signal when the user gave a
    #      phone number and the page literally lists it.
    if query_phones:
        page_digits = re.sub(r"\D", "", website_text)
        for variant in _phone_digit_variants(query_phones):
            if variant and variant in page_digits:
                score += 35
                matches.append("query_phone_on_page")
                signals["query_phone_match"] = variant
                break

    # 4) Director surname mentions
    director_hits: List[str] = []
    for d in (ch.get("directors") or []):
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        if not name:
            continue
        # Take the surname (last token) for a less noisy match.
        surname = name.split()[-1].lower() if name.split() else ""
        if len(surname) >= 4 and surname in text_lower:
            director_hits.append(name)
    if director_hits:
        score += min(10, 4 * len(director_hits))
        matches.append(f"director_surname_on_page({len(director_hits)})")
        signals["director_hits"] = director_hits[:5]

    # 5) Registry markers
    company_number = str(ch.get("company_number") or "").strip()
    if company_number and company_number.lower() in text_lower:
        score += 20
        matches.append("company_number_on_page")
    elif re.search(r"registered (?:in|office)\s+(?:england|wales|scotland|northern ireland|uk)", text_lower):
        score += 5
        matches.append("registered_in_uk_marker")

    # 6) Phone area-code coherence (rough)
    if contact_details and isinstance(contact_details, dict):
        for phone in (contact_details.get("phones") or []):
            if "+44" in phone or phone.strip().startswith("0"):
                signals.setdefault("uk_phone_present", True)
                break

    # 7) Sector / activity hint from AI extraction
    industry = str((ai_data or {}).get("industry") or "").strip()
    if industry:
        signals["ai_industry"] = industry

    return {
        "score": min(100, score),
        "matches": matches,
        "mismatches": mismatches,
        "signals": signals,
    }


# Domains that are never the company's official website. Used to filter
# discovery results before scoring so junk pages cannot win by default.
JUNK_WEBSITE_DOMAINS = {
    "419scam.org", "joewein.net", "scamadviser.com", "scamwatch.gov.au",
    "trustpilot.com", "scamguard.com",
    "reverse-phone-lookup.com", "spokeo.com", "whitepages.com",
    "searchyellowdirectory.com", "searchpeopledirectory.com",
    "sync.me", "truecaller.com", "whocallsme.com", "whocalled.us",
    "callercenter.com", "shouldianswer.com", "tellows.co.uk",
    "mirror.co.uk", "thesun.co.uk", "dailymail.co.uk",
    "bbc.co.uk", "bbc.com", "wikipedia.org",
    "endole.co.uk", "checkcompany.co.uk", "companycheck.co.uk",
    "opencorporates.com", "duedil.com", "dnb.com",
    "find-and-update.company-information.service.gov.uk",
    "yell.com", "thomsonlocal.com", "yelp.com", "yelp.co.uk",
    "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
    "tiktok.com", "youtube.com", "google.com", "g.page", "maps.apple.com",
}

# A website candidate must beat this validator score (0-100) before we will
# treat it as the company's likely official site. Anything weaker is reported
# as "no validated website" rather than crowning a junk domain.
WEBSITE_MIN_MATCH_SCORE = 30


# Companies that provide registered-office / virtual-office / formations
# services. They are real businesses, but they are NEVER the website of the
# client company that registered through them. They will frequently match
# (their site lists their PO Box / postcode and they hold mail for thousands
# of CH companies), so they need their own filter.
FORMATIONS_AGENT_DOMAINS = {
    "identeco.co.uk",
    "1stformations.co.uk",
    "yourcompanyformations.co.uk",
    "rapidformations.co.uk",
    "duport.co.uk",
    "your-virtual-office.co.uk",
    "yourvirtualofficelondon.co.uk",
    "thehoxtonmix.com",
    "hoxton-mix.com",
    "regus.com",
    "icompanyformation.co.uk",
    "madesimplegroup.com",
    "companieshelp.co.uk",
    "formationsdirect.com",
    "quickformations.co.uk",
    "newincorporations.co.uk",
    "thecompanywarehouse.co.uk",
    "yourcompanysetup.com",
    "creativecompanyformations.co.uk",
    "uniwide.co.uk",
    "uniwidemail.co.uk",
    "ukpostbox.com",
    "mailboxesetc.co.uk",
    "ukvirtualoffices.com",
    "officeworld.co.uk",
    "swiftformations.co.uk",
    "formationswise.com",
}


def is_junk_website_domain(domain: str) -> bool:
    if not domain:
        return True
    domain = domain.lower()
    for junk in JUNK_WEBSITE_DOMAINS:
        if domain == junk or domain.endswith("." + junk):
            return True
    # Heuristic junk markers in the host string.
    if any(token in domain for token in ("scam", "419", "phonelookup", "reverse-phone")):
        return True
    return False


def is_formations_agent_domain(domain: str) -> bool:
    if not domain:
        return False
    domain = domain.lower()
    for d in FORMATIONS_AGENT_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return True
    return False


def is_companies_house_default_address(text: Optional[str]) -> bool:
    """Detect Companies House 'default'/proxy registered-office strings such as
    '12345678 - COMPANIES HOUSE DEFAULT ADDRESS' or 'PO Box 4385'. Such an
    address is NOT a unique business location, so postcode/line1 matches on a
    third-party site should not be rewarded.
    """
    if not text:
        return False
    t = str(text).lower()
    if "default address" in t or "companies house default" in t:
        return True
    if re.search(r"\bpo\s*box\s*\d+", t):
        return True
    return False


def extract_query_phones(query: str, country_hint: Optional[str]) -> List[Dict[str, Any]]:
    """Pull phone numbers out of a free-text query (e.g. when the user pastes a
    company name + phone). Uses libphonenumber's matcher so we accept the
    user's exact format and emit normalized variants.
    """
    if not query:
        return []
    region = (country_hint or "GB").strip().upper()
    region_map = {"UK": "GB", "UNITED KINGDOM": "GB", "ENGLAND": "GB",
                  "SCOTLAND": "GB", "WALES": "GB", "NORTHERN IRELAND": "GB",
                  "USA": "US", "US": "US"}
    region = region_map.get(region, region)
    if len(region) != 2:
        region = "GB"

    found: List[Dict[str, Any]] = []
    seen: set = set()
    try:
        for m in phonenumbers.PhoneNumberMatcher(query, region):
            num = m.number
            if not phonenumbers.is_possible_number(num):
                continue
            e164 = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
            if e164 in seen:
                continue
            seen.add(e164)
            found.append({
                "e164": e164,
                "international": phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                "national": phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.NATIONAL),
                "is_valid": phonenumbers.is_valid_number(num),
            })
    except Exception as e:
        logger.warning(f"extract_query_phones failed: {e}")
    return found


def strip_phones_from_text(text: str) -> str:
    """Remove phone-like substrings from a free-text query so a name-only
    search doesn't get polluted by the digits."""
    if not text:
        return ""
    cleaned = re.sub(r"\+?\d[\d\s().\-]{6,}\d", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _phone_digit_variants(query_phones: List[Dict[str, Any]]) -> List[str]:
    """Produce digit-only forms (with and without country code) used to detect
    a query phone inside arbitrary HTML."""
    out: List[str] = []
    for ph in query_phones or []:
        for key in ("e164", "national", "international"):
            digits = re.sub(r"\D", "", str(ph.get(key) or ""))
            if digits and len(digits) >= 9 and digits not in out:
                out.append(digits)
        # GB national without leading 0 (in case the page strips it)
        nat = re.sub(r"\D", "", str(ph.get("national") or ""))
        if nat.startswith("0") and len(nat) >= 10:
            trimmed = nat[1:]
            if trimmed not in out:
                out.append(trimmed)
    return out


def pick_likely_website(results: List[Dict[str, Any]],
                        ch_record: Optional[Dict[str, Any]],
                        country_hint: Optional[str],
                        query_phones: Optional[List[Dict[str, Any]]] = None,
                        registered_office_is_proxy: bool = False) -> Dict[str, Any]:
    """Select the most likely live company website from enrichment results.

    Skips aggregator/directory pages, scores each non-aggregator candidate by
    `website_company_match_score`, and returns the winner with its match
    breakdown. Falls back to the highest-confidence non-aggregator result when
    no Companies House record is available.
    """
    candidates: List[Dict[str, Any]] = []
    for item in results:
        if item.get("status") != "success":
            continue
        if item.get("is_aggregator"):
            continue
        domain = item.get("domain") or ""
        if not domain or is_junk_website_domain(domain) or is_formations_agent_domain(domain):
            continue
        # Re-fetch the page text once to score address/name presence.
        text = clean_text_from_url(item.get("url") or f"https://{domain}") or ""
        match = website_company_match_score(
            ch_record,
            text,
            domain,
            item.get("enrichment") or {},
            item.get("contact_details") or {},
            country_hint,
            query_phones=query_phones,
            registered_office_is_proxy=registered_office_is_proxy,
        )
        ai_conf = int(((item.get("confidence") or {}).get("overall")) or 0)
        # Composite: weight the validator score heavily, with AI confidence as tiebreaker.
        composite = int(match["score"] * 0.75 + ai_conf * 0.25) if ch_record else int(ai_conf * 0.6 + match["score"] * 0.4)
        candidates.append({
            "domain": domain,
            "url": item.get("url"),
            "title": item.get("title"),
            "match_score": match["score"],
            "composite_score": composite,
            "matches": match["matches"],
            "mismatches": match["mismatches"],
            "signals": match["signals"],
            "ai_confidence": ai_conf,
        })

    if not candidates:
        return {"selected": None, "candidates": []}

    candidates.sort(key=lambda c: (c["composite_score"], c["match_score"], c["ai_confidence"]), reverse=True)
    best = candidates[0]
    if int(best.get("match_score") or 0) < WEBSITE_MIN_MATCH_SCORE:
        # No candidate is convincing. Return the ranked list for transparency
        # but do not crown a winner.
        return {"selected": None, "candidates": candidates}
    return {"selected": best, "candidates": candidates}


def find_official_website(ch_record: Dict[str, Any],
                          country_hint: Optional[str],
                          max_candidates: int = 8,
                          query_phones: Optional[List[Dict[str, Any]]] = None,
                          registered_office_is_proxy: bool = False) -> Dict[str, Any]:
    """Run a clean, name-only DuckDuckGo search to find the company's official
    website once we have a Companies House legal name. This avoids contamination
    from noisy original queries (e.g. those including phone numbers).
    """
    if not ch_record:
        return {"selected": None, "candidates": []}

    legal_name = str(ch_record.get("matched_company_name") or "").strip()
    if not legal_name:
        return {"selected": None, "candidates": []}

    # Strip the corporate suffix to broaden the search a little.
    short_name = re.sub(r"\b(limited|ltd|plc|llp|llc|inc)\.?\b", "", legal_name, flags=re.IGNORECASE).strip()

    queries = [
        f'"{legal_name}" official site',
        f'"{legal_name}" contact',
        f'"{legal_name}"',
    ]
    if short_name and short_name.lower() != legal_name.lower():
        queries.append(f'"{short_name}" official site')
    if country_hint:
        queries.insert(0, f'"{legal_name}" {country_hint}')

    seen: Dict[str, Dict[str, str]] = {}
    for q in queries:
        try:
            for hit in search_public_results(q, max_candidates):
                domain = hit.get("domain") or ""
                if not domain or domain in seen:
                    continue
                if is_junk_website_domain(domain) or is_formations_agent_domain(domain):
                    continue
                # Skip well-known aggregator domains using the same list as the crawler.
                aggregator_markers = ("linkedin.com", "facebook.com", "twitter.com",
                                      "yelp.com", "yell.com", "google.com",
                                      "find-and-update.company-information.service.gov.uk")
                if any(m in domain for m in aggregator_markers):
                    continue
                seen[domain] = hit
                if len(seen) >= max_candidates:
                    break
        except Exception as e:
            logger.warning(f"Official-site search failed for '{q}': {e}")
        if len(seen) >= max_candidates:
            break

    candidates: List[Dict[str, Any]] = []
    for domain, hit in seen.items():
        url = hit.get("url") or f"https://{domain}"
        text = clean_text_from_url(url) or ""
        match = website_company_match_score(
            ch_record, text, domain, None, None, country_hint,
            query_phones=query_phones,
            registered_office_is_proxy=registered_office_is_proxy,
        )
        candidates.append({
            "domain": domain,
            "url": url,
            "title": hit.get("title"),
            "match_score": match["score"],
            "composite_score": match["score"],
            "matches": match["matches"],
            "mismatches": match["mismatches"],
            "signals": match["signals"],
            "ai_confidence": 0,
        })

    if not candidates:
        return {"selected": None, "candidates": []}

    candidates.sort(key=lambda c: (c["match_score"],), reverse=True)
    best = candidates[0]
    if int(best.get("match_score") or 0) < WEBSITE_MIN_MATCH_SCORE:
        return {"selected": None, "candidates": candidates}
    return {"selected": best, "candidates": candidates}


def find_website_by_phone(query_phones: List[Dict[str, Any]],
                          ch_record: Optional[Dict[str, Any]],
                          country_hint: Optional[str],
                          max_candidates: int = 10,
                          registered_office_is_proxy: bool = False) -> Dict[str, Any]:
    """Use the user-supplied phone number as a discovery anchor.

    A site that publishes the company's own phone number is overwhelmingly
    likely to BE that company's site. This bypasses both the noisy original
    query and the often-shared Companies House registered office.
    """
    if not query_phones:
        return {"selected": None, "candidates": []}

    legal_name = str((ch_record or {}).get("matched_company_name") or "").strip()

    queries: List[str] = []
    for ph in query_phones:
        nat = ph.get("national")
        intl = ph.get("international")
        if nat:
            # Combined name + phone is the strongest disambiguator when we
            # have a CH legal name; quoted phone alone is the fallback.
            if legal_name:
                queries.append(f'"{legal_name}" "{nat}"')
                queries.append(f'"{nat}" contact')
            queries.append(f'"{nat}"')
        if intl:
            if legal_name:
                queries.append(f'"{legal_name}" "{intl}"')
            queries.append(f'"{intl}"')

    # Deduplicate while preserving order.
    seen_q: set = set()
    queries = [q for q in queries if not (q in seen_q or seen_q.add(q))]

    aggregator_markers = (
        "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
        "yelp.com", "yell.com", "google.com", "g.page",
        "find-and-update.company-information.service.gov.uk",
        "scamadviser", "reverse-phone", "phonelookup", "whocalled",
        "tellows", "shouldianswer", "spokeo", "whitepages",
        "searchyellowdirectory", "searchpeopledirectory",
        "419scam", "qrius.com", "bing.com", "msn.com",
    )
    # E-commerce / shopping domains accidentally match phone-shaped digit
    # strings (SKUs, product IDs). Reject so they cannot dominate.
    shopping_markers = (
        "amazon.", "walmart.com", "ebay.", "etsy.com", "aliexpress.",
        "yami.com", "jomashop.com", "lyko.com", "dermstore.com",
        "elementvapor.com", "nin-nin-game.com", "wayfair.", "target.com",
        "homedepot.com", "bestbuy.com", "argos.co.uk", "currys.co.uk",
        "shopify.com",
    )

    seen: Dict[str, Dict[str, str]] = {}
    # Cap per-query results so one noisy query can't flood the pool.
    per_query_cap = max(3, max_candidates // 2)
    for q in queries[:8]:
        try:
            for hit in search_public_results(q, per_query_cap):
                domain = (hit.get("domain") or "").lower()
                if not domain or domain in seen:
                    continue
                if is_junk_website_domain(domain) or is_formations_agent_domain(domain):
                    continue
                if any(m in domain for m in aggregator_markers):
                    continue
                if any(m in domain for m in shopping_markers):
                    continue
                seen[domain] = hit
                if len(seen) >= max_candidates:
                    break
        except Exception as e:
            logger.warning(f"Phone-based search failed for '{q}': {e}")
        if len(seen) >= max_candidates:
            break

    candidates: List[Dict[str, Any]] = []
    digit_variants = _phone_digit_variants(query_phones)
    for domain, hit in seen.items():
        url = hit.get("url") or f"https://{domain}"
        text = clean_text_from_url(url) or ""
        # Also pull the raw HTML stripped of tags so footer phone numbers
        # (which trafilatura strips out) are still visible to the scorer.
        try:
            raw_html = fetch_html_from_url(url) or ""
            raw_text = html_to_text(raw_html) if raw_html else ""
            if raw_text and raw_text not in text:
                text = (text + "\n" + raw_text)[:80000]
        except Exception:
            pass
        # If the chosen page doesn't already contain the phone digits, also
        # peek at common contact pages — companies often list their phone on
        # /contact rather than the URL DuckDuckGo surfaced.
        page_digits = re.sub(r"\D", "", text)
        has_phone = any(v and v in page_digits for v in digit_variants)
        if not has_phone:
            for path in ("/contact", "/contact-us", "/about", "/about-us"):
                try:
                    extra_url = f"https://{domain}{path}"
                    extra_html = fetch_html_from_url(extra_url) or ""
                    if not extra_html:
                        continue
                    extra_text = html_to_text(extra_html)
                    if extra_text:
                        text = (text + "\n" + extra_text)[:80000]
                        if any(v and v in re.sub(r"\D", "", extra_text) for v in digit_variants):
                            has_phone = True
                            break
                except Exception:
                    continue
        match = website_company_match_score(
            ch_record, text, domain, None, None, country_hint,
            query_phones=query_phones,
            registered_office_is_proxy=registered_office_is_proxy,
        )
        candidates.append({
            "domain": domain,
            "url": url,
            "title": hit.get("title"),
            "match_score": match["score"],
            "composite_score": match["score"],
            "matches": match["matches"],
            "mismatches": match["mismatches"],
            "signals": match["signals"],
            "ai_confidence": 0,
        })

    if not candidates:
        return {"selected": None, "candidates": []}

    # Heavily prefer pages that literally contain the phone number.
    candidates.sort(
        key=lambda c: (
            1 if "query_phone_on_page" in (c.get("matches") or []) else 0,
            c.get("match_score") or 0,
        ),
        reverse=True,
    )
    best = candidates[0]
    if int(best.get("match_score") or 0) < WEBSITE_MIN_MATCH_SCORE:
        return {"selected": None, "candidates": candidates}
    return {"selected": best, "candidates": candidates}



def build_ai_business_summary_prompt(record: Dict[str, Any]) -> str:
    facts = []
    for key in ["matched_company", "company_number", "company_status",
                "registered_address", "verified_address", "likely_website",
                "industry", "phones", "emails", "directors"]:
        item = record.get(key)
        if not item:
            continue
        if isinstance(item, dict) and "value" in item:
            facts.append(f"- {key}: {item.get('value')} (source={item.get('source')}, confidence={item.get('confidence')})")
        else:
            facts.append(f"- {key}: {item}")
    fact_block = "\n".join(facts) if facts else "- (no facts)"
    return f"""
    You are summarising a verified B2B company profile for a CRM.

    Verified facts:
    {fact_block}

    Write a single short paragraph (max 60 words) that:
    - states the company name, status, sector, and where it operates
    - mentions the official website and registered address only if present
    - does NOT invent anything not in the facts
    - is plain prose, no bullet points, no headings

    Return ONLY valid JSON with a single key 'summary' whose value is the paragraph string.
    """


async def run_ai_business_summary(record: Dict[str, Any]) -> Optional[str]:
    prompt = build_ai_business_summary_prompt(record)
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(
                model=AI_MODEL_SUMMARIZER,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
            ),
        )
        parsed = json.loads(response['message']['content'])
        if isinstance(parsed, dict):
            value = parsed.get("summary")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception as e:
        logger.warning(f"Business summary generation failed: {e}")
    return None


# =============================================================================
# Active email discovery
#
# Crawls common contact/about/privacy/legal pages on the chosen company domain
# plus a handful of public sources, de-obfuscates encoded addresses, filters
# noise, classifies each email by role, and validates MX records so the caller
# can attach per-email provenance and confidence.
# =============================================================================

EMAIL_NOISE_LOCAL_PARTS = {
    "example", "test", "user", "username", "your", "yourname", "email",
    "name", "firstname", "lastname", "someone", "sample", "noreply",
    "no-reply", "do-not-reply", "donotreply", "wordpress", "sentry",
}

EMAIL_NOISE_DOMAINS = {
    "example.com", "example.org", "example.net", "test.com", "domain.com",
    "yourdomain.com", "email.com", "sentry.io", "sentry-cdn.com",
    "wordpress.com", "wixpress.com", "squarespace.com", "cloudflare.com",
    "googleusercontent.com", "gstatic.com", "w3.org", "schema.org",
    "placeholder.com",
}

ROLE_EMAIL_PREFIXES = {
    "info", "contact", "enquiries", "enquiry", "hello", "admin",
    "office", "accounts", "billing", "finance", "sales", "support",
    "help", "service", "careers", "jobs", "hr", "recruitment",
    "press", "media", "marketing", "privacy", "legal", "dpo",
    "compliance", "orders", "bookings",
}

EMAIL_SOURCE_PATHS = [
    "/", "/contact", "/contact-us", "/contact/", "/get-in-touch",
    "/about", "/about-us", "/about/", "/team", "/our-team", "/staff",
    "/people", "/leadership", "/management",
    "/privacy", "/privacy-policy", "/legal", "/impressum", "/terms",
    "/help", "/support", "/customer-service",
]


def classify_email(email: str, site_domain: str) -> Dict[str, Any]:
    """Return {role, is_role_account, is_personal, is_official_domain} for an email."""
    email = email.strip().lower()
    local = email.split("@", 1)[0] if "@" in email else ""
    domain = email.split("@", 1)[1] if "@" in email else ""
    is_role = local in ROLE_EMAIL_PREFIXES
    role = local if is_role else None
    is_official = bool(site_domain and (domain == site_domain or domain.endswith("." + site_domain)))
    return {
        "role": role,
        "is_role_account": is_role,
        "is_personal": not is_role and "@" in email,
        "is_official_domain": is_official,
        "domain": domain,
    }


def _clean_email_candidate(raw: str) -> Optional[str]:
    """Normalise an email; return None if obviously invalid/noise."""
    if not raw:
        return None
    email = raw.strip().lower().strip(".,;:<>()[]{}\"'`")
    if "@" not in email or email.count("@") != 1:
        return None
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        return None
    if any(ch.isspace() for ch in email):
        return None
    # Strip common noise / placeholder patterns.
    if local in EMAIL_NOISE_LOCAL_PARTS:
        return None
    if domain in EMAIL_NOISE_DOMAINS:
        return None
    # Reject file-like locals (image hashes, css sprites mislabelled as emails).
    if re.fullmatch(r"[a-f0-9]{16,}", local):
        return None
    # Reject values that look like filenames (e.g. logo-v2@2x.png was parsed as email).
    if any(domain.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
        return None
    if len(email) > 120:
        return None
    return email


def _decode_obfuscated_emails(html: str) -> List[str]:
    """Extract emails from common obfuscation patterns on websites."""
    if not html:
        return []
    found: List[str] = []
    decoded_html = html_lib.unescape(html)
    # 1) mailto: links (may be HTML-entity encoded).
    for m in re.finditer(r"mailto:([^\"'?>\s]+)", decoded_html, flags=re.IGNORECASE):
        raw = m.group(1)
        # Decode HTML entities and URL encoding.
        raw = raw.replace("&#64;", "@").replace("%40", "@")
        raw = re.sub(r"&#(\d+);", lambda mm: chr(int(mm.group(1))), raw)
        found.append(raw)
    # 2) Plain-text emails.
    for m in re.finditer(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", decoded_html):
        found.append(m.group(0))
    # 3) "name [at] domain [dot] com" style.
    patt = re.compile(
        r"([A-Za-z0-9._%+\-]+)\s*(?:\[at\]|\(at\)|\{at\}|\s+at\s+)\s*([A-Za-z0-9.\-]+)\s*(?:\[dot\]|\(dot\)|\{dot\}|\s+dot\s+)\s*([A-Za-z]{2,})",
        flags=re.IGNORECASE,
    )
    for m in patt.finditer(decoded_html):
        found.append(f"{m.group(1)}@{m.group(2)}.{m.group(3)}")
    # 4) JS concatenation style, e.g. 'info' + '@' + 'example.com'.
    js_cat = re.compile(
        r"['\"]([A-Za-z0-9._%+\-]{1,64})['\"]\s*\+\s*['\"]@['\"]\s*\+\s*['\"]([A-Za-z0-9.\-]+\.[A-Za-z]{2,})['\"]",
        flags=re.IGNORECASE,
    )
    for m in js_cat.finditer(decoded_html):
        found.append(f"{m.group(1)}@{m.group(2)}")
    return found


def discover_company_emails(site_domain: str,
                            extra_urls: Optional[List[str]] = None,
                            country_hint: Optional[str] = None,
                            max_pages: int = 10) -> Dict[str, Any]:
    """Crawl likely contact pages on the company's own domain and de-obfuscate
    emails found there. Returns structured per-email metadata plus source pages
    and an MX validation flag for the domain.
    """
    if not site_domain:
        return {"emails": [], "pages_crawled": [], "mx_valid": False}

    site_domain = site_domain.lower().strip()
    base_url = f"https://{site_domain}"
    urls: List[str] = [f"{base_url}{path}" for path in EMAIL_SOURCE_PATHS]
    # Also follow in-site links from the homepage (contact/about/legal).
    home_html = fetch_html_from_url(base_url)
    if home_html:
        for link in extract_relevant_site_links(base_url, home_html, max_links=12):
            if link not in urls:
                urls.append(link)
    if extra_urls:
        for u in extra_urls:
            if not u:
                continue
            # Keep only in-domain URLs; ignore trackers / foreign domains.
            if normalize_domain(u) != site_domain:
                continue
            if u not in urls:
                urls.insert(0, u)

    urls = urls[:max_pages * 2]  # hard cap
    found_emails: Dict[str, Dict[str, Any]] = {}
    pages_crawled: List[str] = []

    def process_url(url: str) -> Dict[str, Any]:
        html = fetch_html_from_url(url)
        if not html:
            return {"url": url, "emails": []}
        raw_emails = _decode_obfuscated_emails(html)
        # Also scan first-party script bundles for hardcoded addresses.
        try:
            soup = BeautifulSoup(html, "html.parser")
            script_links: List[str] = []
            for script in soup.select("script[src]"):
                src = script.get("src") or ""
                if not src:
                    continue
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = urljoin(url, src)
                elif not src.startswith("http"):
                    continue
                if normalize_domain(src) != site_domain:
                    continue
                if src not in script_links:
                    script_links.append(src)
                if len(script_links) >= 8:
                    break
            for src in script_links:
                js = fetch_html_from_url(src, allow_non_html=True)
                if not js:
                    continue
                raw_emails.extend(_decode_obfuscated_emails(js))
        except Exception:
            pass
        return {"url": url, "emails": raw_emails}

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(urls)))) as executor:
        futures = [executor.submit(process_url, u) for u in urls]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                continue
            # Record successfully fetched pages even if no emails were found.
            if r.get("url") and r["url"] not in pages_crawled:
                pages_crawled.append(r["url"])
            if not r.get("emails"):
                continue
            for raw in r["emails"]:
                email = _clean_email_candidate(raw)
                if not email:
                    continue
                if email not in found_emails:
                    meta = classify_email(email, site_domain)
                    meta["email"] = email
                    meta["sources"] = []
                    found_emails[email] = meta
                if r["url"] not in found_emails[email]["sources"]:
                    found_emails[email]["sources"].append(r["url"])
            if len(pages_crawled) >= max_pages:
                break

    mx_valid = bool(get_mx_records(site_domain))

    # Rank: official-domain + role accounts first, then personal on-domain,
    # then anything else. Within a tier, more source pages wins.
    def sort_key(meta: Dict[str, Any]):
        tier = 0
        if meta["is_official_domain"] and meta["is_role_account"]:
            tier = 3
        elif meta["is_official_domain"]:
            tier = 2
        elif meta["is_role_account"]:
            tier = 1
        return (tier, len(meta.get("sources", [])))

    ranked = sorted(found_emails.values(), key=sort_key, reverse=True)

    return {
        "emails": ranked,
        "pages_crawled": pages_crawled[:max_pages],
        "mx_valid": mx_valid,
        "domain": site_domain,
    }


async def build_verified_b2b_record(query: str,
                                    location: Optional[str],
                                    results: List[Dict[str, Any]],
                                    overall_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Produce the final structured CRM-ready record.

    Layers:
      1. Companies House record is authoritative for legal identity + registered address.
      2. Each non-aggregator website is scored against the CH record; the best
         match becomes the 'likely_website'.
      3. Verified address combines CH address with cross-referenced web evidence.
      4. Phones/emails are accepted only when verified (multi-source or on the
         official site) — provenance recorded per field.
      5. A specialist summarizer model produces a short business activity narrative.

    Returns a dict containing per-field provenance, validation_notes,
    mismatch_warnings, and a final 'final_enrichment_summary' string.
    """
    successful = [r for r in results if r.get("status") == "success"]
    record: Dict[str, Any] = {
        "query": query,
        "location": location,
        "matched_company": None,
        "company_number": None,
        "company_status": None,
        "registered_address": None,
        "registered_address_fields": None,
        "verified_address": None,
        "verified_address_fields": None,
        "directors": None,
        "likely_website": None,
        "trading_name": None,
        "industry": None,
        "phones": None,
        "emails": None,
        "email_details": None,
        "social_links": None,
        "field_confidence": {},
        "field_sources": {},
        "validation_notes": [],
        "mismatch_warnings": [],
        "final_enrichment_summary": None,
    }

    def set_field(name: str, fr: Optional[Dict[str, Any]]) -> None:
        record[name] = fr
        if fr is None:
            return
        record["field_confidence"][name] = fr.get("confidence", 0)
        record["field_sources"][name] = fr.get("source", "unknown")

    # ---- 1) Pick best Companies House hit (already filtered by overall_summary) ----
    companies_house_hits = [
        item for item in successful
        if isinstance(item.get("companies_house"), dict)
        and (item["companies_house"].get("company_number") or item["companies_house"].get("matched_company_name"))
    ]
    ch_primary = None
    for item in companies_house_hits:
        ch = item["companies_house"]
        matched = str(ch.get("matched_company_name") or "")
        if not matched:
            continue
        # Reuse the same trust check as overall_summary: name overlap with query/summary.
        summary_name = str(overall_summary.get("company_name") or "")
        toks_a = set(company_name_tokens(matched))
        toks_b = set(company_name_tokens(summary_name)) | set(company_name_tokens(query or ""))
        if toks_a and toks_b and toks_a.intersection(toks_b):
            ch_primary = ch
            break
    if not ch_primary and companies_house_hits:
        ch_primary = companies_house_hits[0]["companies_house"]
    if not ch_primary:
        # When web discovery is noisy/empty, still try a direct CH lookup using
        # a cleaned query string (remove phones and location noise).
        query_for_ch = strip_phones_from_text(query or "")
        query_for_ch = re.sub(r"\b(united kingdom|uk|england|scotland|wales|northern ireland)\b", " ", query_for_ch, flags=re.IGNORECASE)
        query_for_ch = re.sub(r"\s+", " ", query_for_ch).strip()
        if query_for_ch:
            try:
                direct_ch = await asyncio.get_event_loop().run_in_executor(
                    None, companies_house_lookup_by_name, query_for_ch,
                )
                if isinstance(direct_ch, dict) and (direct_ch.get("matched_company_name") or direct_ch.get("company_number")):
                    ch_primary = direct_ch
            except Exception as e:
                logger.warning(f"Direct Companies House fallback failed: {e}")

    # ---- 2) Authoritative identity fields from Companies House ----
    if ch_primary:
        legal_name = ch_primary.get("matched_company_name")
        company_number = ch_primary.get("company_number")
        company_status = ch_primary.get("company_status")
        if legal_name:
            set_field("matched_company", field_record(
                legal_name, "companies_house",
                SOURCE_CONFIDENCE["companies_house"],
                notes=["from Companies House register"]))
        if company_number:
            set_field("company_number", field_record(
                company_number, "companies_house",
                SOURCE_CONFIDENCE["companies_house"]))
        if company_status:
            set_field("company_status", field_record(
                company_status, "companies_house",
                SOURCE_CONFIDENCE["companies_house"]))

        ch_addr = format_registered_office_address(ch_primary.get("registered_office_address"))
        if ch_addr:
            set_field("registered_address", field_record(
                ch_addr, "companies_house",
                SOURCE_CONFIDENCE["companies_house"]))
            ch_addr_fields = parse_address_fields(ch_addr, location)
            if any(bool(v) for v in ch_addr_fields.values()):
                record["registered_address_fields"] = ch_addr_fields

        directors = ch_primary.get("directors") or []
        if directors:
            set_field("directors", field_record(
                directors, "companies_house",
                SOURCE_CONFIDENCE["companies_house"],
                notes=[f"{len(directors)} active directors from Companies House"]))

    # Fallback identity from overall_summary when no CH record was found.
    if not record["matched_company"] and overall_summary.get("company_name"):
        set_field("matched_company", field_record(
            overall_summary["company_name"], "ai_inferred",
            SOURCE_CONFIDENCE["ai_inferred"],
            notes=["no Companies House match; derived from website/AI"]))

    # ---- 3) Likely website via dedicated validator ----
    # Pull any phone numbers out of the user's free-text query first; these are
    # one of the strongest discovery signals we have because the company's own
    # site will publish its phone, but a registered-office agent / scam list
    # generally won't.
    query_phones = extract_query_phones(query, location)

    # Detect "default" / shared registered office addresses so we don't reward
    # third parties (formations agents, virtual offices) that happen to use
    # the same postcode.
    ch_addr_text_for_proxy = (record.get("registered_address") or {}).get("value") or ""
    if not ch_addr_text_for_proxy and ch_primary:
        ch_addr_text_for_proxy = format_registered_office_address(
            ch_primary.get("registered_office_address")) or ""
    registered_office_is_proxy = is_companies_house_default_address(ch_addr_text_for_proxy)
    if registered_office_is_proxy:
        record["validation_notes"].append(
            "Companies House registered office is a shared/default address — "
            "postcode/line-1 matches will not score websites")

    # Pass 1: score the URLs that the original discovery already crawled.
    website_choice = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pick_likely_website(
            results, ch_primary, location,
            query_phones=query_phones,
            registered_office_is_proxy=registered_office_is_proxy,
        ),
    )
    selected_site = website_choice.get("selected")

    # Pass 2: if Companies House gave us a clean legal name, run a focused
    # search using just that name. Original queries can include phone numbers,
    # SIC keywords, etc., which often surface phone-lookup or scam-list pages.
    fresh_choice: Dict[str, Any] = {"selected": None, "candidates": []}
    if ch_primary:
        fresh_choice = await asyncio.get_event_loop().run_in_executor(
            None, lambda: find_official_website(
                ch_primary, location, 8,
                query_phones=query_phones,
                registered_office_is_proxy=registered_office_is_proxy,
            ),
        )
        fresh_best = fresh_choice.get("selected")
        if fresh_best and (
            not selected_site
            or int(fresh_best.get("match_score") or 0) > int((selected_site or {}).get("match_score") or 0)
        ):
            selected_site = fresh_best
            website_choice = fresh_choice

    # Pass 3: if the user gave us a phone number, search for that exact phone.
    # Whichever site lists it is overwhelmingly likely to be the real one.
    phone_choice: Dict[str, Any] = {"selected": None, "candidates": []}
    if query_phones:
        phone_choice = await asyncio.get_event_loop().run_in_executor(
            None, lambda: find_website_by_phone(
                query_phones, ch_primary, location, 10,
                registered_office_is_proxy=registered_office_is_proxy,
            ),
        )
        phone_best = phone_choice.get("selected")
        if phone_best:
            phone_score = int(phone_best.get("match_score") or 0)
            current_score = int((selected_site or {}).get("match_score") or 0)
            phone_confirmed = "query_phone_on_page" in (phone_best.get("matches") or [])
            # A phone-confirmed page wins ties and beats anything within 20 points.
            if not selected_site or phone_score > current_score or (phone_confirmed and phone_score + 20 >= current_score):
                selected_site = phone_best
                website_choice = phone_choice

    if selected_site:
        match_score = int(selected_site.get("match_score") or 0)
        phone_confirmed = "query_phone_on_page" in (selected_site.get("matches") or [])
        # Confidence: validator score blended with source authority (official_website).
        conf = int(SOURCE_CONFIDENCE["official_website"] * 0.5 + match_score * 0.5)
        if phone_confirmed:
            conf = min(99, conf + 10)
        notes = [f"validator score {match_score}/100"]
        if phone_confirmed:
            notes.append("query phone number found on page")
        if selected_site.get("matches"):
            notes.append("matches: " + ", ".join(selected_site["matches"][:5]))
        if selected_site.get("mismatches"):
            notes.append("mismatches: " + ", ".join(selected_site["mismatches"][:3]))
            for warn in selected_site["mismatches"]:
                record["mismatch_warnings"].append(f"website:{selected_site['domain']}: {warn}")
        alternatives = [
            {"domain": c["domain"], "match_score": c["match_score"], "composite_score": c.get("composite_score", c["match_score"])}
            for c in website_choice.get("candidates", [])[1:4]
        ]
        site_url = f"https://{selected_site['domain']}"
        # A phone-confirmed match always counts as the official website.
        is_official = phone_confirmed or match_score >= 50
        set_field("likely_website", field_record(
            site_url, "official_website" if is_official else "discovery",
            conf, alternatives=alternatives, notes=notes))
    else:
        # No website passed validation. Surface ranked candidates as alternatives
        # but do not crown a winner; this prevents junk pages from being picked.
        ranked = ((website_choice.get("candidates") or [])
                  + (fresh_choice.get("candidates") or [])
                  + (phone_choice.get("candidates") or []))
        alt = [
            {"domain": c["domain"], "match_score": c["match_score"]}
            for c in ranked[:4]
        ]
        record["validation_notes"].append(
            "No website passed minimum match score; not surfacing a likely_website")
        if alt:
            record["mismatch_warnings"].append(
                f"website: best candidate scored {alt[0]['match_score']}/100 (below threshold {WEBSITE_MIN_MATCH_SCORE})")

    # ---- 4) Verified address: cross-check CH against summary/web evidence ----
    summary_addr = overall_summary.get("verified_address") or overall_summary.get("address")
    if record["registered_address"] and summary_addr:
        ch_addr_value = record["registered_address"]["value"]
        signals = address_match_signals(ch_addr_value, summary_addr, location)
        if signals["same_postcode"] or signals["same_line1"]:
            # Companies House address is corroborated by web sources.
            set_field("verified_address", field_record(
                ch_addr_value,
                "companies_house+cross_referenced",
                min(99, SOURCE_CONFIDENCE["companies_house"] + 4),
                notes=[f"address match score {signals['score']}/100",
                       f"same_postcode={signals['same_postcode']}",
                       f"same_line1={signals['same_line1']}"]))
            verified_fields = parse_address_fields(ch_addr_value, location)
            if any(bool(v) for v in verified_fields.values()):
                record["verified_address_fields"] = verified_fields
        elif signals["conflict"]:
            # CH and web disagree on postcode — surface CH as authoritative but flag.
            set_field("verified_address", field_record(
                ch_addr_value, "companies_house",
                SOURCE_CONFIDENCE["companies_house"] - 10,
                notes=["conflicting address found on web"]))
            verified_fields = parse_address_fields(ch_addr_value, location)
            if any(bool(v) for v in verified_fields.values()):
                record["verified_address_fields"] = verified_fields
            record["mismatch_warnings"].append(
                f"address: Companies House '{ch_addr_value}' conflicts with web-derived '{summary_addr}'")
        else:
            set_field("verified_address", field_record(
                ch_addr_value, "companies_house",
                SOURCE_CONFIDENCE["companies_house"] - 5,
                notes=["web evidence neither corroborated nor conflicted"]))
            verified_fields = parse_address_fields(ch_addr_value, location)
            if any(bool(v) for v in verified_fields.values()):
                record["verified_address_fields"] = verified_fields
    elif record["registered_address"]:
        set_field("verified_address", field_record(
            record["registered_address"]["value"], "companies_house",
            SOURCE_CONFIDENCE["companies_house"] - 8,
            notes=["no independent web corroboration available"]))
        verified_fields = parse_address_fields(record["registered_address"]["value"], location)
        if any(bool(v) for v in verified_fields.values()):
            record["verified_address_fields"] = verified_fields
    elif summary_addr:
        # No CH record; rely on cross-referenced web evidence.
        set_field("verified_address", field_record(
            summary_addr, "cross_referenced",
            SOURCE_CONFIDENCE["cross_referenced"],
            notes=["no Companies House record; derived from web cross-reference"]))
        verified_fields = parse_address_fields(summary_addr, location)
        if any(bool(v) for v in verified_fields.values()):
            record["verified_address_fields"] = verified_fields

    # ---- 5) Phones: prefer the user-supplied phone (validated), then any
    #         cross-referenced phones from the discovery pass.
    user_phones: List[str] = []
    for ph in (query_phones or []):
        if ph.get("is_valid") and ph.get("international") and ph["international"] not in user_phones:
            user_phones.append(ph["international"])
    discovered_phones = list(overall_summary.get("phones") or [])
    if user_phones:
        # User phone is canonical; surface other discovered phones as alternatives.
        alts = [p for p in discovered_phones if p not in user_phones]
        notes = ["validated by libphonenumber from user query"]
        site_matches = (selected_site.get("matches") if selected_site else []) or []
        if "query_phone_on_page" in site_matches:
            notes.append("confirmed on the company's website")
            conf = SOURCE_CONFIDENCE["official_website"]
        else:
            conf = SOURCE_CONFIDENCE["cross_referenced"]
        set_field("phones", field_record(
            user_phones, "user_query+libphonenumber", conf,
            alternatives=alts, notes=notes))
    elif discovered_phones:
        set_field("phones", field_record(
            discovered_phones, "cross_referenced",
            SOURCE_CONFIDENCE["cross_referenced"],
            notes=["validated by libphonenumber and seen on official site or 2+ sources"]))

    # ---- 6) Emails: only run when we have a validated official website ----
    # Without a trusted website we cannot tell whether a scraped email belongs
    # to the company. Emitting random emails here led to nonsense results.
    likely = record.get("likely_website") or {}
    site_value = likely.get("value") or ""
    site_source = likely.get("source") or ""
    site_domain = normalize_domain(site_value) if site_value else ""

    if site_domain and site_source == "official_website" and not is_junk_website_domain(site_domain):
        try:
            seed_urls: List[str] = []
            if selected_site and selected_site.get("url"):
                seed_urls.append(str(selected_site.get("url")))
            if site_value:
                seed_urls.append(site_value)
            discovery = await asyncio.get_event_loop().run_in_executor(
                None, discover_company_emails, site_domain, seed_urls, location, 10,
            )
        except Exception as e:
            logger.warning(f"Email discovery failed for {site_domain}: {e}")
            discovery = {"emails": [], "pages_crawled": [], "mx_valid": False, "domain": site_domain}

        # Merge prior emails from the initial enrichment pass, but only keep
        # those that match the official domain. This drops random emails
        # scraped from directory / scam-list pages.
        merged: Dict[str, Dict[str, Any]] = {}
        for meta in discovery.get("emails") or []:
            if meta.get("is_official_domain"):
                merged[meta["email"]] = meta

        for raw in (overall_summary.get("emails") or []):
            cleaned = _clean_email_candidate(raw) if raw else None
            if not cleaned or cleaned in merged:
                continue
            meta = classify_email(cleaned, site_domain)
            if not meta.get("is_official_domain"):
                continue
            meta.update({"email": cleaned, "sources": ["initial_crawl"]})
            merged[cleaned] = meta

        email_list = list(merged.values())
        if email_list:
            primary = [m["email"] for m in email_list]
            base_conf = SOURCE_CONFIDENCE["official_website"]
            notes = [
                f"{len(email_list)} email(s) on domain {site_domain}",
                f"MX valid: {discovery.get('mx_valid')}",
            ]
            if discovery.get("pages_crawled"):
                notes.append(f"crawled: {', '.join(discovery['pages_crawled'][:4])}")
            if discovery.get("mx_valid"):
                base_conf = min(99, base_conf + 5)

            set_field("emails", field_record(
                primary, "official_website", base_conf,
                alternatives=[],
                notes=notes,
            ))
            record["email_details"] = {
                "domain": site_domain,
                "mx_valid": discovery.get("mx_valid"),
                "pages_crawled": discovery.get("pages_crawled"),
                "entries": [
                    {
                        "email": m["email"],
                        "role": m.get("role"),
                        "is_role_account": m.get("is_role_account"),
                        "is_personal": m.get("is_personal"),
                        "is_official_domain": m.get("is_official_domain"),
                        "sources": m.get("sources") or [],
                    }
                    for m in email_list
                ],
            }
        else:
            # Fallback: infer common role-based mailbox on the validated domain.
            # Some sites hide emails behind JS/apps or forms; this provides a
            # low-confidence candidate instead of returning none.
            inferred_candidates = [
                f"info@{site_domain}",
                f"enquiries@{site_domain}",
                f"hello@{site_domain}",
                f"contact@{site_domain}",
            ]
            if discovery.get("mx_valid"):
                primary = inferred_candidates[0]
                primary_meta = classify_email(primary, site_domain)
                primary_meta.update({
                    "email": primary,
                    "sources": ["heuristic:common_role_account"],
                    "inferred": True,
                })

                set_field("emails", field_record(
                    [primary], "ai_inferred", SOURCE_CONFIDENCE["ai_inferred"],
                    alternatives=inferred_candidates[1:],
                    notes=[
                        "No explicit email found on fetched pages; inferred common role mailbox",
                        f"domain MX valid: {discovery.get('mx_valid')}",
                    ],
                ))

                record["email_details"] = {
                    "domain": site_domain,
                    "mx_valid": discovery.get("mx_valid"),
                    "pages_crawled": discovery.get("pages_crawled"),
                    "entries": [
                        {
                            "email": primary_meta["email"],
                            "role": primary_meta.get("role"),
                            "is_role_account": primary_meta.get("is_role_account"),
                            "is_personal": primary_meta.get("is_personal"),
                            "is_official_domain": primary_meta.get("is_official_domain"),
                            "sources": primary_meta.get("sources") or [],
                            "inferred": True,
                        }
                    ],
                }
            else:
                record["email_details"] = {
                    "domain": site_domain,
                    "mx_valid": discovery.get("mx_valid"),
                    "pages_crawled": discovery.get("pages_crawled"),
                    "entries": [],
                }
                record["validation_notes"].append(
                    f"No emails found on official website {site_domain}")
    else:
        # No validated official website → don't surface emails at all.
        record["validation_notes"].append(
            "Email discovery skipped: no validated official website")

    # ---- 7) Industry / sector via AI on best result ----
    best_ai = None
    for item in successful:
        enr = item.get("enrichment") or {}
        if isinstance(enr, dict) and enr.get("industry"):
            best_ai = enr
            break
    if best_ai:
        set_field("industry", field_record(
            best_ai.get("industry"), "ai_inferred",
            SOURCE_CONFIDENCE["ai_inferred"] + 10,
            notes=["extracted from website text by local model"]))

    # ---- 8) Trading name vs registered name ----
    if record["matched_company"]:
        legal = str(record["matched_company"]["value"])
        ai_name = ""
        for item in successful:
            cand = str(((item.get("enrichment") or {}).get("company_name") or "")).strip()
            if cand:
                ai_name = cand
                break
        if ai_name and slugify_text(ai_name) != slugify_text(legal):
            # Genuinely different from the legal name → likely a trading name.
            if not any(suffix in ai_name.lower() for suffix in ["limited", "ltd", "plc", "llp"]):
                set_field("trading_name", field_record(
                    ai_name, "ai_inferred",
                    SOURCE_CONFIDENCE["ai_inferred"],
                    notes=["differs from registered legal name"]))

    # ---- 9) Social links ----
    social_links: Dict[str, str] = {}
    for item in successful:
        enr = item.get("enrichment") or {}
        ln = str(enr.get("linkedin_url") or "").strip()
        if ln and "linkedin" not in social_links:
            social_links["linkedin"] = ln
        for url_key in ("url",):
            url = str(item.get(url_key) or "")
            for platform in ("facebook.com", "twitter.com", "instagram.com"):
                if platform in url and platform.split(".")[0] not in social_links:
                    social_links[platform.split(".")[0]] = url
    if social_links:
        set_field("social_links", field_record(
            social_links, "discovery",
            SOURCE_CONFIDENCE["discovery"] + 10,
            notes=["collected from discovery results"]))

    # ---- 10) Validation notes summary ----
    if ch_primary:
        record["validation_notes"].append("Companies House record used as authoritative source")
    else:
        record["validation_notes"].append("No Companies House record matched; results derived from web only")
    if selected_site and selected_site.get("match_score", 0) >= 50:
        record["validation_notes"].append(
            f"Website {selected_site['domain']} validated against Companies House (score {selected_site['match_score']}/100)")
    elif selected_site:
        record["validation_notes"].append(
            f"Website {selected_site['domain']} chosen but failed strong validation (score {selected_site['match_score']}/100)")

    # ---- 11) AI business activity summary (specialist summarizer model) ----
    summary_text = await run_ai_business_summary(record)
    if summary_text:
        record["final_enrichment_summary"] = summary_text

    return record


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

    # ---- Build the explainable, CRM-ready verified B2B record ----
    try:
        verified_record = await build_verified_b2b_record(
            request.query, request.location, results, overall_summary,
        )
        overall_summary["verified_record"] = verified_record
    except Exception as e:
        logger.warning(f"Verified record build failed: {e}")
        verified_record = None

    return {
        "query": request.query,
        "location": request.location,
        "discovered": len(candidates),
        "domains_scanned": domains_scanned,
        "results": results,
        "processing_model": AI_MODEL,
        "models": {
            "default": AI_MODEL,
            "matcher": AI_MODEL_MATCHER,
            "validator": AI_MODEL_VALIDATOR,
            "address": AI_MODEL_ADDRESS,
            "classifier": AI_MODEL_CLASSIFIER,
            "summarizer": AI_MODEL_SUMMARIZER,
        },
        "overall_summary": overall_summary,
        "verified_record": verified_record,
    }


@app.post("/enrich-verified", tags=["Enrichment"])
async def enrich_verified_endpoint(request: CrawlBusinessesRequest):
    """
    Verification-first enrichment optimised for CRM import.

    Runs the full discovery + Companies House + website-validation pipeline and
    returns ONLY the explainable verified record:
      - matched_company, company_number, company_status
      - registered_address, verified_address (with cross-reference notes)
      - directors (Companies House, when available)
      - likely_website (validated against Companies House signals)
      - phones, emails, industry, social_links, trading_name
      - field_confidence and field_sources for every field
      - validation_notes and mismatch_warnings
      - final_enrichment_summary (short paragraph from local summarizer model)

    Companies House is the primary source where a record matches the query.
    Other fields are accepted only when corroborated; conflicts are surfaced as
    mismatch_warnings rather than silently overwriting authoritative data.
    """
    full = await crawl_businesses(request)
    if not isinstance(full, dict):
        raise HTTPException(status_code=500, detail="Enrichment pipeline returned unexpected payload")
    record = full.get("verified_record")
    if not record:
        raise HTTPException(status_code=404, detail="No verified record could be produced for this query")
    return {
        "query": request.query,
        "location": request.location,
        "models": full.get("models"),
        "domains_scanned": full.get("domains_scanned"),
        "verified_record": record,
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
Environment="AI_MODEL_MATCHER=${AI_MODEL_MATCHER:-$SELECTED_MODEL}"
Environment="AI_MODEL_VALIDATOR=${AI_MODEL_VALIDATOR:-$SELECTED_MODEL}"
Environment="AI_MODEL_ADDRESS=${AI_MODEL_ADDRESS:-$SELECTED_MODEL}"
Environment="AI_MODEL_CLASSIFIER=${AI_MODEL_CLASSIFIER:-$SELECTED_MODEL}"
Environment="AI_MODEL_SUMMARIZER=${AI_MODEL_SUMMARIZER:-$SELECTED_MODEL}"
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
    export AI_MODEL_MATCHER=${AI_MODEL_MATCHER:-$SELECTED_MODEL}
    export AI_MODEL_VALIDATOR=${AI_MODEL_VALIDATOR:-$SELECTED_MODEL}
    export AI_MODEL_ADDRESS=${AI_MODEL_ADDRESS:-$SELECTED_MODEL}
    export AI_MODEL_CLASSIFIER=${AI_MODEL_CLASSIFIER:-$SELECTED_MODEL}
    export AI_MODEL_SUMMARIZER=${AI_MODEL_SUMMARIZER:-$SELECTED_MODEL}
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
