"""Conservative deterministic extraction. Search snippets never enter this module."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup
import phonenumbers
from .http import publisher, canonical_url
from .models import company_number
from .providers import legal_name, norm, person_name

NUMBER = re.compile(r"\b(?:company\s+(?:registration\s+|registered\s+)?(?:number|no\.?)|registered\s+(?:company\s+)?(?:number|no\.?)|registration\s+(?:number|no\.?))\s*[:#.]?\s*([A-Z]{2}\s*\d{6}|\d{6,8})\b", re.I)
LINK_PATH = re.compile(r"contact|about|team|leadership|director|legal|privacy|terms", re.I)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PAST_ROLE = re.compile(r"\b(former|previous|resigned|retired|ex[- ]director|ceased|died|passed away)\b", re.I)
DIRECTOR = re.compile(r"\b(?:managing |executive |non.executive |finance |operations |sales |technical )?director\b", re.I)
DIRECTORY_HOSTS = {"company-information.service.gov.uk", "companycheck.co.uk", "endole.co.uk", "open.endole.co.uk", "companiesintheuk.co.uk", "find-and-update.company-information.service.gov.uk", "yell.com", "thomsonlocal.com", "cqc.org.uk", "nhs.uk", "linkedin.com", "facebook.com", "192.com", "dnb.com", "zoominfo.com"}


def valid_phone(value):
    try:
        parsed = phonenumbers.parse(value, "GB")
        if phonenumbers.is_valid_number(parsed) and parsed.country_code == 44 and not parsed.extension:
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        pass
    return None


@dataclass
class Page:
    url: str
    fetched_at: float
    sha256: str
    text: str
    numbers: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    links: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    legal_blocks: list[str] = field(default_factory=list)

    @property
    def publisher(self):
        return publisher(self.url)

    def evidence(self, quote="", kind="website"):
        return {"url": self.url, "publisher": self.publisher, "kind": kind, "fetched_at": self.fetched_at, "content_sha256": self.sha256, "excerpt": quote[:500]}


def parse_page(result):
    soup = BeautifulSoup(result["text"], "html.parser")
    for node in soup(["script", "style", "noscript", "template"]):
        node.decompose()
    text = soup.get_text(" ", strip=True)
    blocks = []
    for node in soup.find_all(["p", "li", "article", "section", "div", "footer"]):
        value = node.get_text(" ", strip=True)
        if 5 <= len(value) <= 600 and value not in blocks:
            blocks.append(value)
    # Also handle simple legal lines with no structural markup.
    legal_blocks = [b for b in blocks if NUMBER.search(b)]
    if not legal_blocks and len(text) <= 600 and NUMBER.search(text):
        legal_blocks = [text]
    numbers = set()
    for match in NUMBER.finditer(text):
        try:
            numbers.add(company_number(match.group(1)))
        except ValueError:
            pass
    phones = {phonenumbers.format_number(m.number, phonenumbers.PhoneNumberFormat.E164) for m in phonenumbers.PhoneNumberMatcher(text, "GB") if phonenumbers.is_valid_number(m.number) and m.number.country_code == 44 and not m.number.extension}
    links = []
    for a in soup.find_all("a", href=True):
        raw = a["href"].strip()
        if raw.startswith("tel:"):
            number = valid_phone(raw[4:])
            if number:
                phones.add(number)
        elif raw.startswith(("http://", "https://", "/")) or ":" not in raw:
            try:
                url = canonical_url(urljoin(result["url"], raw))
                if url not in links:
                    links.append(url)
            except Exception:
                pass
    return Page(result["url"], result["fetched_at"], result["sha256"], text, numbers, phones, set(EMAIL.findall(text)), links, blocks, legal_blocks)


def phrase_present(phrase, text):
    return bool(phrase and re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text))


def legal_binding(page, profile):
    """Only a single legal entity disclosed in context on a non-directory site."""
    number = profile["company_number"]
    if page.publisher in DIRECTORY_HOSTS or page.numbers != {number}:
        return False
    return any(phrase_present(legal_name(profile["company_name"]), legal_name(block)) and any(company_number(m.group(1)) == number for m in NUMBER.finditer(block)) for block in page.legal_blocks)


def related_page(page, profile, official_publisher):
    if page.numbers:
        return page.numbers == {profile["company_number"]} and phrase_present(legal_name(profile["company_name"]), legal_name(page.text))
    # Without a registry ID demand full legal name, postcode AND a link to the
    # confirmed official website. Shared postcodes or name similarity alone fail.
    postcode = profile.get("registered_office_address", {}).get("postal_code", "")
    return bool(postcode and phrase_present(legal_name(profile["company_name"]), legal_name(page.text)) and re.sub(r"\s", "", postcode).lower() in re.sub(r"\s", "", page.text).lower() and any(publisher(link) == official_publisher for link in page.links))


def director_support(page, registered_name):
    wanted = person_name(registered_name)
    # Full names only. An omitted middle name, initial or nickname goes to review.
    # News, dated stories and blog posts can describe historical appointments.
    if re.search(r"/(?:news|blog|press|archive|20\d{2})(?:/|$)", urlsplit(page.url).path, re.I):
        return [], []
    matches = [b for b in page.blocks if phrase_present(wanted, norm(b)) and DIRECTOR.search(b)]
    conflicts = [b for b in matches if PAST_ROLE.search(b) or re.search(r"\b(no longer|not (?:a |the )?director)\b", b, re.I)]
    # A name somewhere in the same paragraph as another person's title is not
    # enough. Require adjacent name/role (or an explicit "is/serves as" phrase).
    role = r"(?:(?:managing|executive|non executive|finance|operations|sales|technical) )?director"
    after = re.compile(r"\b" + re.escape(wanted) + r" (?:is (?:the |our )?|serves as (?:the |our )?)?" + role + r"\b")
    before = re.compile(r"\b" + role + r" " + re.escape(wanted) + r"\b")
    current = [b for b in matches if b not in conflicts and (after.search(norm(b)) or before.search(norm(b)))]
    return current, conflicts


def contact_quote(page, value):
    for block in page.blocks:
        if value in block or value in {valid_phone(m.raw_string) for m in phonenumbers.PhoneNumberMatcher(block, "GB")}:
            return block
    return value


def site_links(page):
    return [url for url in page.links if publisher(url) == page.publisher and LINK_PATH.search(urlsplit(url).path) and not urlsplit(url).query][:5]
