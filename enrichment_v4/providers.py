from __future__ import annotations
import re
import unicodedata
from datetime import date
from urllib.parse import urlencode
from .http import FetchError
from .models import company_number

CH = "https://api.company-information.service.gov.uk"
PUBLIC_CH = "https://find-and-update.company-information.service.gov.uk"


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower().replace("&", " and ")))


def legal_name(value):
    value = norm(value)
    return re.sub(r"\b(limited|ltd)\b", "ltd", value)


def person_name(value):
    if "," in value:
        surname, rest = value.split(",", 1)
        value = rest + " " + surname
    return re.sub(r"^(mr|mrs|ms|miss|dr|sir) ", "", norm(value))


def display_person(value):
    if "," in value:
        surname, rest = value.split(",", 1)
        value = rest.strip() + " " + surname.strip()
    return value.title()


class CompaniesHouse:
    def __init__(self, fetcher, token):
        self.fetcher, self.token = fetcher, token

    def _get(self, path, refresh=False):
        if not self.token:
            raise FetchError("companies_house_key_missing")
        return self.fetcher.json(CH + path, token=self.token, refresh=refresh)

    def search(self, name, refresh=False):
        payload, evidence = self._get("/search/companies?" + urlencode({"q": name, "items_per_page": 100}), refresh)
        if not isinstance(payload.get("items"), list):
            raise FetchError("company_search_incomplete")
        candidates = [{"company_number": r.get("company_number"), "company_name": r.get("title"), "company_status": r.get("company_status"), "address": r.get("address", {})} for r in payload["items"]]
        # Search ordering is not identity verification; these are candidates only.
        return candidates, evidence

    def profile(self, number, refresh=False):
        number = company_number(number)
        data, evidence = self._get("/company/" + number, refresh)
        if data.get("company_number") != number or not data.get("company_name") or not data.get("company_status"):
            raise FetchError("company_profile_incomplete", CH + "/company/" + number)
        return data, evidence

    def officers(self, number, refresh=False):
        number = company_number(number)
        all_items, pages, seen = [], [], set()
        expected, previous_etag = None, None
        start = 0
        for _ in range(50):
            payload, evidence = self._get(f"/company/{number}/officers?" + urlencode({"items_per_page": 100, "start_index": start}), refresh)
            items, total = payload.get("items"), payload.get("total_results")
            if not isinstance(items, list) or type(total) is not int or total < 0 or payload.get("start_index") != start:
                raise FetchError("officer_pagination_invalid")
            if expected is not None and total != expected:
                raise FetchError("officer_list_changed_during_fetch")
            if previous_etag and payload.get("etag") and previous_etag != payload["etag"]:
                raise FetchError("officer_list_changed_during_fetch")
            expected, previous_etag = total, payload.get("etag")
            for item in items:
                if not isinstance(item, dict):
                    raise FetchError("officer_record_incomplete")
                key = (item.get("links") or {}).get("self")
                if not key or key in seen or not item.get("name") or not item.get("officer_role"):
                    raise FetchError("officer_record_incomplete_or_duplicate")
                seen.add(key)
                all_items.append(item)
            pages.append(evidence)
            start += len(items)
            if start == total:
                break
            if not items or start > total:
                raise FetchError("officer_pagination_incomplete")
        else:
            raise FetchError("officer_page_limit")
        directors, excluded = [], []
        for item in all_items:
            # Exact natural-person role; corporate/nominee directors and LLP members
            # are never silently turned into human directors for sales calls.
            reason = None
            if item.get("officer_role") != "director":
                reason = "not_natural_person_director"
            elif item.get("resigned_on") or item.get("ceased_on"):
                reason = "resigned_or_ceased"
            elif item.get("appointed_on"):
                try:
                    if date.fromisoformat(item["appointed_on"]) > date.today():
                        reason = "future_appointment"
                except (ValueError, TypeError):
                    reason = "invalid_appointment_date"
            elif item.get("is_pre_1992_appointment") is not True:
                reason = "appointment_date_missing"
            row = {"name": display_person(item["name"]), "registered_name": item["name"], "title": "Director", "officer_role": item["officer_role"], "company_number": number,
                   "appointed_on": item.get("appointed_on"), "appointed_before": item.get("appointed_before"),
                   "appointment_id": item["links"]["self"], "source_url": PUBLIC_CH + f"/company/{number}/officers",
                   "identity_verified_on": (item.get("identity_verification_details") or {}).get("identity_verified_on")}
            if reason:
                excluded.append({"name": row["name"], "reason": reason})
            else:
                directors.append(row)
        return directors, excluded, pages


class BraveSearch:
    name = "brave"

    @property
    def configured(self):
        return bool(self.token)

    def __init__(self, fetcher, token):
        self.fetcher, self.token = fetcher, token

    def search(self, query, count=10, offset=0, refresh=False):
        if not self.token:
            raise FetchError("brave_search_key_missing")
        url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": min(count, 20), "offset": offset, "country": "GB", "search_lang": "en", "safesearch": "moderate"})
        data, evidence = self.fetcher.json(url, token=self.token, provider="brave", refresh=refresh, ttl=3600)
        rows = data.get("web", {}).get("results", [])
        if not isinstance(rows, list):
            raise FetchError("search_response_invalid")
        return [{"url": r.get("url"), "title": r.get("title", ""), "snippet": r.get("description", ""), "status": "discovered_unverified"} for r in rows if isinstance(r, dict) and r.get("url")], evidence
