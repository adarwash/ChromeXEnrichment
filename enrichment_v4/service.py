from __future__ import annotations
import time
import hashlib
import uuid
from collections import defaultdict
from .evidence import parse_page, legal_binding, related_page, director_support, contact_quote, site_links, valid_phone, DIRECTORY_HOSTS
from .http import FetchError, canonical_url, publisher, DEADLINE
from .models import EnrichRequest
from .providers import CompaniesHouse, BraveSearch, legal_name, person_name, norm

# Explicit publisher families; different URLs/subdomains are not extra sources.
# Operators must review permission and common ownership before adding a source.
DEFAULT_SOURCE_GROUPS = {"cqc.org.uk": "cqc", "nhs.uk": "nhs", "yell.com": "yell", "thomsonlocal.com": "thomsonlocal"}
MAX_WEB_PAGES = 18
MAX_AGE = 86400


class EnrichmentService:
    def __init__(self, fetcher, store, ch_key="", search_key="", source_groups=None):
        self.fetcher, self.store = fetcher, store
        self.ch = CompaniesHouse(fetcher, ch_key)
        self.search = BraveSearch(fetcher, search_key)
        self.source_groups = dict(DEFAULT_SOURCE_GROUPS if source_groups is None else source_groups)

    def enrich(self, request: EnrichRequest):
        parent = DEADLINE.get()
        token = DEADLINE.set(min(parent, time.monotonic() + 180) if parent else time.monotonic() + 180)
        try:
            return self._enrich(request)
        finally:
            DEADLINE.reset(token)

    def _enrich(self, request: EnrichRequest):
        now = time.time()
        record_id = str(uuid.uuid4())
        result = {"schema_version": "4.0", "record_id": record_id, "checked_at": now, "expires_at": None,
                  "status": "needs_review", "call_ready": False, "blocking_reasons": [], "warnings": [],
                  "matched_company": None, "company_number": None, "company_status": None,
                  "registered_address": None, "sic_codes": [], "likely_website": None,
                  "registered_directors": [], "directors": [], "excluded_officers": [],
                  "phones": [], "emails": [], "call_contact": None,
                  "company_candidates": [], "website_candidates": [], "field_evidence": {}, "source_errors": [],
                  "checks": {"legal_entity": False, "current_officers_complete": False, "director_cross_reference": False, "phone_cross_reference": False, "fresh_sources": False},
                  "notes": ["call_ready is a data-quality decision, not proof of phone ownership, personal identity, purchasing authority or permission to market.", "Registered office may differ from trading premises. Search snippets and user hints never count as verification."]}
        pages = {}
        errors = result["source_errors"]
        reasons = result["blocking_reasons"]
        evidence = result["field_evidence"]

        def finish():
            result["blocking_reasons"] = list(dict.fromkeys(reasons))
            result["call_ready"] = not reasons and all(result["checks"].values())
            if result["call_ready"]:
                result["status"] = "ready"
            elif result["status"] != "blocked":
                result["status"] = "needs_review"
            if not result["call_ready"]:
                result["call_contact"] = None
            self.store.save(record_id, request.model_dump(), result)
            return result

        def page(url):
            try:
                url = canonical_url(url)
                if url in pages:
                    return pages[url]
                if len(pages) >= MAX_WEB_PAGES:
                    return None
                # Count failures too: no unbounded crawling on broken sites.
                pages[url] = None
                fetched = self.fetcher.get(url, refresh=request.refresh, ttl=21600)
                parsed = parse_page(fetched)
                pages[url] = parsed
                return parsed
            except FetchError as exc:
                errors.append(exc.as_dict())
                return None

        def search(query, count=10):
            try:
                rows, _ = self.search.search(query, count=count)
                return rows
            except FetchError as exc:
                errors.append(exc.as_dict())
                return []

        # Crawl a supplied domain before choosing a legal entity. A user-provided
        # number/domain is a hint, not proof that the business and entity match.
        domain_hint = None
        initial_site_pages = []
        if request.domain:
            root = page(request.domain)
            if root:
                domain_hint = root.publisher
                initial_site_pages.append(root)
                for link in site_links(root):
                    p = page(link)
                    if p:
                        initial_site_pages.append(p)
            else:
                reasons.append("supplied_website_unavailable")
        name = request.company_name or request.query
        number = request.company_number
        if not number and initial_site_pages:
            numbers = set().union(*(p.numbers for p in initial_site_pages))
            if len(numbers) == 1:
                number = next(iter(numbers))
            elif len(numbers) > 1:
                reasons.append("website_discloses_multiple_legal_entities")
        if not number and name:
            try:
                candidates, _ = self.ch.search(name, request.refresh)
                result["company_candidates"] = candidates
                exact = [r for r in candidates if legal_name(r["company_name"]) == legal_name(name)]
                if request.postcode:
                    clean = lambda x: str(x).replace(" ", "").upper()
                    exact = [r for r in exact if clean(r["address"].get("postal_code")) == clean(request.postcode)]
                if len(exact) == 1:
                    number = exact[0]["company_number"]
                else:
                    reasons.append("company_identity_ambiguous" if len(exact) > 1 else "no_exact_legal_name_match")
            except FetchError as exc:
                errors.append(exc.as_dict())
                reasons.append("company_search_unavailable")
        if not number:
            reasons.append("company_number_unresolved")
            return finish()
        try:
            profile, profile_ev = self.ch.profile(number, request.refresh)
        except FetchError as exc:
            errors.append(exc.as_dict())
            reasons.append("company_profile_unavailable")
            return finish()
        evidence["company"] = [profile_ev]
        result.update(matched_company=profile["company_name"], company_number=number,
                      company_status=profile["company_status"], registered_address=profile.get("registered_office_address"), sic_codes=profile.get("sic_codes", []))
        if name and legal_name(name) != legal_name(profile["company_name"]):
            reasons.append("requested_name_differs_from_legal_entity")
        if request.postcode and request.postcode.replace(" ", "").upper() != profile.get("registered_office_address", {}).get("postal_code", "").replace(" ", "").upper():
            reasons.append("requested_postcode_differs_from_registered_office")
        if profile["company_status"] != "active":
            reasons.append("company_not_active")
            result["status"] = "blocked"
            return finish()
        if profile.get("has_been_liquidated") or profile.get("has_insolvency_history") or profile.get("company_status_detail"):
            reasons.append("company_status_requires_review")
        try:
            registered, excluded, officer_evidence = self.ch.officers(number, request.refresh)
            result["registered_directors"], result["excluded_officers"] = registered, excluded
            evidence["registered_directors"] = officer_evidence
            result["checks"]["current_officers_complete"] = True
        except FetchError as exc:
            errors.append(exc.as_dict())
            reasons.append("officer_list_unavailable_or_incomplete")
            return finish()
        if not registered:
            reasons.append("no_current_natural_person_directors")

        # Discover the official website using a licensed search API. A result's
        # title or snippet is never evidence. All candidates must be fetched.
        seed_urls = [r["url"] for r in search(f'"{profile["company_name"]}" "{number}"')]
        if request.domain:
            seed_urls.insert(0, request.domain)
        candidates = list(initial_site_pages)
        visited_publishers = {p.publisher for p in candidates}
        # Reserve room for external checks and do not crawl obvious directories
        # as if they were the company's official website.
        if not any(legal_binding(p, profile) for p in candidates):
            for url in seed_urls:
                pub = publisher(url)
                if pub in visited_publishers or pub in DIRECTORY_HOSTS or pub in self.source_groups:
                    continue
                if len(visited_publishers) >= 3 or len(pages) >= 11:
                    break
                visited_publishers.add(pub)
                p = page(url)
                if not p:
                    continue
                candidates.append(p)
                for link in site_links(p)[:3]:
                    child = page(link)
                    if child:
                        candidates.append(child)
        bound = defaultdict(list)
        for p in candidates:
            if legal_binding(p, profile):
                bound[p.publisher].append(p)
        result["website_candidates"] = [{"publisher": pub, "evidence_urls": [p.url for p in ps]} for pub, ps in bound.items()]
        official = domain_hint if domain_hint in bound else next(iter(bound)) if len(bound) == 1 and not request.domain else None
        if not official:
            reasons.append("official_website_ambiguous_or_unconfirmed")
            return finish()
        official_pages = [p for p in candidates if p.publisher == official]
        # Any entity mismatch on a related page prevents mixing group subsidiaries.
        if any(p.numbers and p.numbers != {number} for p in official_pages):
            reasons.append("conflicting_company_numbers_on_website")
        result["checks"]["legal_entity"] = not any(r in reasons for r in ("requested_name_differs_from_legal_entity", "requested_postcode_differs_from_registered_office", "conflicting_company_numbers_on_website", "website_discloses_multiple_legal_entities"))
        result["likely_website"] = bound[official][0].url
        evidence["website"] = [p.evidence(next(iter(p.legal_blocks), ""), "official_website") for p in bound[official]]

        # Only names currently recorded as natural-person directors can pass.
        # A website's managing/technical director alone is not a legal director.
        director_evidence = []
        same_names = defaultdict(int)
        for d in registered:
            same_names[person_name(d["registered_name"])] += 1
        for director in registered:
            supports, conflicts = [], []
            for p in official_pages:
                current, past = director_support(p, director["registered_name"])
                supports.extend(p.evidence(q, "official_director_role") for q in current)
                conflicts.extend(p.evidence(q, "conflicting_director_role") for q in past)
            if conflicts:
                reasons.append("director_role_conflict")
                director_evidence.extend(conflicts)
            if supports and not conflicts and same_names[person_name(director["registered_name"])] == 1:
                result["directors"].append({**director, "verification": "current_registry_and_official_website", "evidence": supports})
                director_evidence.extend(supports)
        evidence["directors"] = director_evidence
        if request.director_name:
            result["directors"] = [d for d in result["directors"] if person_name(d["name"]) == person_name(request.director_name)]
            if not result["directors"]:
                reasons.append("requested_director_not_cross_verified")
        if result["directors"]:
            result["checks"]["director_cross_reference"] = True
        else:
            reasons.append("no_director_cross_verified_on_official_website")

        # For phone checks require official publication plus an approved external
        # publisher tied to this exact entity. Several pages on one website, or
        # identical content on several domains, do not multiply corroboration.
        official_phones = set().union(*(p.phones for p in official_pages))
        urls = list(request.corroboration_urls)
        urls.extend(seed_urls)
        if official_phones:
            query_phone = valid_phone(request.phone_number) if request.phone_number else sorted(official_phones)[0]
            for row in search(f'"{profile["company_name"]}" "{query_phone or ""}" contact'):
                urls.append(row["url"])
            # Target relevant approved publishers when broad results are noisy.
            if not any(publisher(u) in self.source_groups for u in urls):
                sites = " OR ".join("site:" + domain for domain in self.source_groups)
                if sites:
                    urls.extend(r["url"] for r in search(f'"{profile["company_name"]}" ({sites})', count=6))
        fingerprint = lambda p: hashlib.sha256(norm(p.text).encode()).hexdigest()
        external, seen_urls, seen_hashes = [], set(), {fingerprint(p) for p in official_pages}
        official_group = self.source_groups.get(official, "official:" + official)
        for url in urls:
            pub = publisher(url)
            if url in seen_urls or pub == official:
                continue
            seen_urls.add(url)
            if pub not in self.source_groups or self.source_groups[pub] in (official_group, "companies_house"):
                if url in request.corroboration_urls:
                    result["warnings"].append("Unapproved corroboration publisher: " + pub)
                continue
            p = page(url)
            if not p or p.publisher != pub or fingerprint(p) in seen_hashes:
                continue
            if related_page(p, profile, official):
                seen_hashes.add(fingerprint(p))
                external.append(p)
        evidence["external_company_matches"] = [p.evidence(profile["company_name"], "external_entity_match") for p in external]
        verified_phones = []
        phone_evidence = []
        for number_value in sorted(official_phones):
            corroborators = [p for p in external if number_value in p.phones]
            if not corroborators:
                continue
            primary = [p for p in official_pages if number_value in p.phones]
            ev = [p.evidence(contact_quote(p, number_value), "official_phone") for p in primary] + [p.evidence(contact_quote(p, number_value), "corroborating_phone") for p in corroborators]
            verified_phones.append(number_value)
            phone_evidence.extend(ev)
        # An external entity-matched source listing completely different numbers
        # is a conflict, not evidence that can be discarded to improve a score.
        if any(p.phones and not p.phones.intersection(official_phones) for p in external):
            reasons.append("conflicting_business_phones")
        result["phones"] = verified_phones
        evidence["phones"] = phone_evidence
        selected_phone = None
        if request.phone_number:
            supplied = valid_phone(request.phone_number)
            if supplied in verified_phones:
                selected_phone = supplied
            else:
                reasons.append("supplied_phone_not_cross_verified")
        elif len(verified_phones) == 1:
            selected_phone = verified_phones[0]
        elif len(verified_phones) > 1:
            reasons.append("multiple_phones_select_a_target")
        if selected_phone:
            result["checks"]["phone_cross_reference"] = True
        else:
            reasons.append("no_unambiguous_cross_verified_phone")
        # Emails are only published observations, never claimed deliverable or
        # assigned to a named person. MX/SMTP checks cannot prove mailbox ownership.
        email_values = set().union(*(p.emails for p in official_pages))
        result["emails"] = [{"value": v, "verification": "published_on_official_website", "deliverability": "unknown", "evidence": [p.evidence(v, "official_email") for p in official_pages if v in p.emails]} for v in sorted(email_values)]
        # All relied-upon sources must be recent; fetching old prose today is not
        # proof it was authored today, hence explicit conflict checks and review.
        timestamps = [profile_ev["fetched_at"]] + [e["fetched_at"] for e in officer_evidence]
        timestamps += [p.fetched_at for p in official_pages + external]
        result["expires_at"] = min(timestamps) + MAX_AGE
        result["checks"]["fresh_sources"] = all(0 <= time.time() - t <= MAX_AGE for t in timestamps)
        if not result["checks"]["fresh_sources"]:
            reasons.append("evidence_expired_or_clock_invalid")
        if selected_phone and result["directors"]:
            result["call_contact"] = {"company_number": result["company_number"], "legal_company_name": result["matched_company"], "business_phone": selected_phone,
                                      "verified_director_names": [d["name"] for d in result["directors"]], "phone_type": "business_number_not_personal_direct_line"}
        return finish()

    def discover(self, request):
        rows, source = self.search.search(request.query + " " + request.location, request.max_results, request.offset)
        unique, seen = [], set()
        for row in rows:
            pub = publisher(row["url"])
            if pub in seen or pub in DIRECTORY_HOSTS or pub in self.source_groups:
                continue
            seen.add(pub)
            candidate = {**row, "publisher": pub, "call_ready": False}
            if request.enrich:
                candidate["enrichment"] = self.enrich(EnrichRequest(domain=row["url"]))
                candidate["call_ready"] = candidate["enrichment"]["call_ready"]
            unique.append(candidate)
        return {"schema_version": "4.0", "query": request.query, "offset": request.offset, "results": unique, "search_source": source,
                "notes": ["Candidate discovery is not verification. Every business must pass /enrich-verified and /call-check before the calling queue."]}
