from __future__ import annotations
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from . import __version__
from .http import Fetcher, FetchError
from .models import EnrichRequest, DiscoveryRequest, BatchRequest, CatalogueImport
from .service import EnrichmentService
from .store import Store


def create_app(service=None, api_key=None):
    key = api_key if api_key is not None else os.getenv("ENRICHMENT_API_KEY", "")
    slots = threading.BoundedSemaphore(4)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal service
        if len(key) < 24:
            raise RuntimeError("Set ENRICHMENT_API_KEY to at least 24 characters")
        if service is None:
            store = Store(os.getenv("ENRICHMENT_DB", "/var/lib/chromex-enrichment-v4/enrichment.sqlite3"))
            groups = json.loads(os.environ["SOURCE_GROUPS_JSON"]) if os.getenv("SOURCE_GROUPS_JSON") else None
            if groups is not None and (not isinstance(groups, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in groups.items())):
                raise RuntimeError("SOURCE_GROUPS_JSON must map publisher domains to ownership groups")
            service = EnrichmentService(Fetcher(store, user_agent=os.getenv("CRAWLER_USER_AGENT", "ChromeXEnrichment/4.0"), search_spacing=float(os.getenv("SEARCH_SPACING_SECONDS", "1.1")), search_monthly_limit=int(os.getenv("EXTERNAL_SEARCH_MONTHLY_LIMIT", "0"))), store,
                                        os.getenv("COMPANIES_HOUSE_API_KEY", ""), os.getenv("BRAVE_SEARCH_API_KEY", ""), groups, search_provider=os.getenv("SEARCH_PROVIDER", "local"), searxng_url=os.getenv("SEARXNG_URL", "http://127.0.0.1:8888"), searxng_daily_limit=int(os.getenv("SEARXNG_DAILY_LIMIT", "100")))
        service.store.prune()
        app.state.service = service
        yield

    app = FastAPI(title="ChromeX Enrichment V4", version=__version__, lifespan=lifespan,
                  description="Evidence-based UK company and director checks. Needs-review results are never calling records.")

    def authenticate(request: Request):
        provided = request.headers.get("X-API-Key", "")
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            provided = authorization[7:]
        if len(key) < 24 or not secrets.compare_digest(provided.encode(), key.encode()):
            raise HTTPException(401, "Valid API key required")

    protected = [Depends(authenticate)]

    @contextmanager
    def capacity():
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Enrichment capacity busy; retry later", headers={"Retry-After": "30"})
        try:
            # Pruning also happens during service use, rather than only at restart.
            service.store.prune()
            yield
        finally:
            slots.release()

    def envelope(record):
        return {"schema_version": "4.0", "status": record["status"], "call_ready": record["call_ready"], "degraded": False, "verified_record": record}

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/system-info", dependencies=protected)
    def system_info():
        return {"version": __version__, "companies_house_configured": bool(service.ch.token), "search_configured": service.search.configured, "search_provider": service.search.name, "external_search_monthly_limit": getattr(service.fetcher, "search_monthly_limit", 0), "catalogue": service.store.catalog_stats(),
                "source_groups": service.source_groups, "policy": "strict", "scope": "UK registered companies", "max_parallel_jobs": 4}

    @app.post("/catalogue/import", dependencies=protected)
    def catalogue_import(request: CatalogueImport):
        count = service.store.index_candidates([item.model_dump() for item in request.items], request.source)
        return {"indexed": count, "call_ready": False, "notes": ["Imported candidate URLs only. Every contact still requires verification."]}

    @app.get("/catalogue/status", dependencies=protected)
    def catalogue_status():
        return service.store.catalog_stats()

    @app.post("/enrich-verified", dependencies=protected)
    @app.post("/enrich", dependencies=protected)
    def enrich(request: EnrichRequest):
        with capacity():
            return envelope(service.enrich(request))

    @app.post("/batch-enrich", dependencies=protected)
    def batch(request: BatchRequest):
        with capacity():
            return {"results": [envelope(service.enrich(item)) for item in request.items]}

    @app.post("/discover-businesses", dependencies=protected)
    @app.post("/crawl-businesses", dependencies=protected)
    def discover(request: DiscoveryRequest):
        with capacity():
            try:
                return service.discover(request)
            except FetchError as exc:
                raise HTTPException(503, detail=exc.as_dict()) from None

    @app.get("/records/{record_id}", dependencies=protected)
    def record(record_id: str):
        stored = service.store.get(record_id)
        if not stored:
            raise HTTPException(404, "Record not found or expired retention")
        result = stored["result"]
        if result["call_ready"] and (result["expires_at"] or 0) <= time.time():
            result.update(call_ready=False, status="needs_review", call_contact=None)
            result["blocking_reasons"].append("evidence_expired")
        return envelope(result)

    @app.post("/call-check/{record_id}", dependencies=protected)
    def call_check(record_id: str):
        stored = service.store.get(record_id)
        if not stored:
            raise HTTPException(404, "Record not found or expired retention")
        with capacity():
            request = EnrichRequest.model_validate({**stored["request"], "refresh": True})
            result = service.enrich(request)
            previous_number = stored["result"].get("company_number")
            if previous_number and result.get("company_number") != previous_number:
                # Never allow a stored calling record to silently switch company.
                result.update(call_ready=False, status="needs_review", call_contact=None)
                result["blocking_reasons"].append("legal_entity_changed_since_previous_check")
                # The original audit remains immutable; persist the guarded result.
                with service.store.connect() as db:
                    db.execute("UPDATE records SET result=? WHERE id=?", (json.dumps(result), result["record_id"]))
            return envelope(result)

    return app


app = create_app()
