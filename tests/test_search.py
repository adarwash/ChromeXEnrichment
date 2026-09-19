import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import pytest
from fastapi.testclient import TestClient
from enrichment_v4.api import create_app
from enrichment_v4.catalogue import csv_candidates, send_batch
from enrichment_v4.http import Fetcher, FetchError
from enrichment_v4.models import EnrichRequest, DiscoveryRequest
from enrichment_v4.search import LocalSearch, SearXNGSearch, CandidateSearch, build_search
from enrichment_v4.service import EnrichmentService
from enrichment_v4.store import Store
from test_v4 import FakeFetch, EXTERNAL, request, OFFICIAL


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / 'state.db')


@pytest.fixture
def seeded(store):
    store.index_candidates([
        {'url': 'https://acme-care.co.uk/', 'title': 'Acme Care Limited', 'text': 'care home Hampshire PO15 7AD 01234567'},
        {'url': 'https://cqc.org.uk/location/acme', 'title': 'Acme Care Limited', 'text': 'care home Hampshire PO15 7AD 01234567'},
        {'url': 'https://wrong-business.co.uk/', 'title': 'Other Engineering Ltd', 'text': 'Engineering workshop Hampshire'},
    ])
    return store


def test_default_provider_ignores_existing_paid_key(store):
    fake = FakeFetch()
    service = EnrichmentService(fake, store, 'ch', 'already-configured-paid-key')
    assert service.search.name == 'local'
    rows, metadata = service.search.search('care homes')
    assert rows == [] and metadata['provider'] == 'local'
    assert not fake.calls


def test_local_fts_requires_sector_and_location(seeded):
    rows, evidence = LocalSearch(seeded).search('care homes Hampshire')
    assert len(rows) == 2
    assert all('acme' in r['url'] for r in rows)
    assert evidence['scope'] == 'imported_and_previously_fetched_urls_only'


def test_local_fts_operator_injection_safe(seeded):
    assert LocalSearch(seeded).search('" OR * : DROP TABLE records')[0] == []
    assert seeded.catalog_stats()['candidate_urls'] == 3


def test_enrichment_query_uses_name_without_needing_phone(seeded):
    rows, _ = LocalSearch(seeded).search('"Acme Care Limited" "+442079460958" contact')
    assert len(rows) == 2
    assert {r['status'] for r in rows} == {'discovered_unverified'}


def test_local_site_filters(seeded):
    rows, _ = LocalSearch(seeded).search('"Acme Care Limited" (site:cqc.org.uk OR site:nhs.uk)')
    assert [r['url'] for r in rows] == ['https://cqc.org.uk/location/acme']


def test_free_workflow_verifies_without_any_search_api(seeded):
    fake = FakeFetch()
    service = EnrichmentService(fake, seeded, 'ch')
    result = service.enrich(request())
    assert result['call_ready'], result['blocking_reasons']
    assert not any('api.search.brave.com' in url for url, _ in fake.calls)
    assert not result['source_errors']


def test_local_new_lead_discovery_and_enrichment(seeded):
    service = EnrichmentService(FakeFetch(), seeded, 'ch')
    result = service.discover(DiscoveryRequest(query='care homes', location='Hampshire', enrich=True))
    assert len(result['results']) == 1
    assert result['results'][0]['call_ready']
    assert result['results'][0]['enrichment']['company_number'] == '01234567'


def test_imports_never_bypass_current_director_checks(seeded):
    fake = FakeFetch()
    fake.officers[0]['resigned_on'] = '2026-01-01'
    seeded.index_candidates([{'url': 'https://acme-care.co.uk/', 'title': 'Acme Care Limited', 'text': 'John David Smith is VERIFIED current director 01234567'}])
    record = EnrichmentService(fake, seeded, 'ch').enrich(request())
    assert not record['call_ready']
    assert record['call_contact'] is None


def test_free_mode_without_external_evidence_still_fails_closed(store):
    result = EnrichmentService(FakeFetch(), store, 'ch').enrich(request())
    assert not result['call_ready']
    assert 'no_unambiguous_cross_verified_phone' in result['blocking_reasons']


def test_previous_verification_preserves_imported_sector_tags(seeded):
    service = EnrichmentService(FakeFetch(), seeded, 'ch')
    assert service.enrich(request())['call_ready']
    assert len(seeded.search_catalog('care homes Hampshire')) == 2


def test_catalogue_upsert_is_unique_and_survives_restart(seeded):
    seeded.index_candidates([{'url': 'https://acme-care.co.uk/', 'title': 'Acme Care Limited', 'text': 'dental Portsmouth'}])
    reopened = Store(seeded.path)
    assert reopened.catalog_stats()['candidate_urls'] == 3
    assert len(reopened.search_catalog('dental Portsmouth')) == 1


def test_catalogue_import_requires_auth_and_rejects_verification_fields(store):
    service = EnrichmentService(FakeFetch(), store, 'ch')
    with TestClient(create_app(service, 'k' * 32)) as client:
        payload = {'items': [{'url': 'https://example.co.uk', 'title': 'Acme'}]}
        assert client.post('/catalogue/import', json=payload).status_code == 401
        headers = {'X-API-Key': 'k' * 32}
        assert client.post('/catalogue/import', headers=headers, json=payload).json()['call_ready'] is False
        payload['items'][0]['call_ready'] = True
        assert client.post('/catalogue/import', headers=headers, json=payload).status_code == 422
        assert client.get('/catalogue/status', headers=headers).json()['candidate_urls'] == 1


def test_csv_column_mapping_and_bom(tmp_path):
    path = tmp_path / 'care.csv'
    path.write_text('\ufeffLocation Name,Location Website,County,CQC URL\nAcme Care,https://acme-care.co.uk,Hampshire,https://cqc.org.uk/location/acme\n', encoding='utf-8')
    rows = list(csv_candidates(path))
    assert len(rows) == 2
    assert rows[0]['title'] == 'Acme Care'
    assert 'Hampshire' in rows[0]['text']


def test_csv_requires_a_website_column(tmp_path):
    path = tmp_path / 'missing.csv'
    path.write_text('Name,Phone\nAcme,123\n')
    with pytest.raises(ValueError, match='--url-column'):
        list(csv_candidates(path))


def test_csv_import_never_sends_key_over_remote_http():
    with pytest.raises(ValueError, match='HTTPS'):
        send_batch('http://remote.example.com', 'secret', [], 'test')


@pytest.mark.parametrize('endpoint', ['http://localhost:8888', 'http://10.0.0.1:8888', 'http://169.254.169.254', 'https://public.example:8888', 'http://127.0.0.1:8888/redirect', 'http://user:pass@127.0.0.1:8888', 'http://127.0.0.1:8888?url=http://evil'])
def test_searxng_endpoint_is_operator_fixed_loopback(store, endpoint):
    with pytest.raises(ValueError):
        SearXNGSearch(store, endpoint)


def test_searxng_json_adapter_dedupes_and_caches(store):
    search = SearXNGSearch(store)
    payload = {'results': [{'url': 'https://example.co.uk', 'title': 'A'}, {'url': 'https://example.co.uk/', 'title': 'A'}, {'url': 'file:///etc/passwd'}]}
    with patch.object(search, '_request', return_value=(200, {}, json.dumps(payload).encode())) as wire:
        rows, source = search.search('business')
        assert len(rows) == 1 and rows[0]['status'] == 'discovered_unverified'
        assert source['provider'] == 'searxng'
        assert search.search('business')[0] == rows
        assert wire.call_count == 1


def test_searxng_does_not_follow_redirects(store):
    search = SearXNGSearch(store)
    with patch.object(search, '_request', return_value=(302, {'location': 'http://169.254.169.254/'}, b'')) as wire, pytest.raises(FetchError, match='searxng_http_302'):
        search.search('test')
    assert wire.call_count == 1


def test_searxng_failure_stops_retries_across_restart(store):
    search = SearXNGSearch(store)
    with patch.object(search, '_request', return_value=(429, {'retry-after': '600'}, b'')):
        with pytest.raises(FetchError, match='429'):
            search.search('test')
    other = SearXNGSearch(Store(store.path))
    with patch.object(other, '_request') as wire, pytest.raises(FetchError, match='cooldown'):
        other.search('different test')
    wire.assert_not_called()


def test_searxng_engine_failure_reported(store):
    search = SearXNGSearch(store)
    with patch.object(search, '_request', return_value=(200, {}, b'{"results":[],"unresponsive_engines":[["bing","CAPTCHA"]]}')), pytest.raises(FetchError, match='engines_unavailable'):
        search.search('test')


def test_searxng_bang_cannot_override_selected_engines(store):
    search = SearXNGSearch(store)
    with patch.object(search, '_request', return_value=(200, {}, b'{"results":[]}')) as wire:
        search.search('!google !!bing :en care homes site:cqc.org.uk')
    query = parse_qs(urlsplit(wire.call_args.args[0]).query)['q'][0]
    assert '!' not in query and ':en' not in query
    assert 'site:cqc.org.uk' in query


def test_searxng_daily_cap_blocks_before_wire(store):
    search = SearXNGSearch(store, daily_limit=0)
    with patch.object(search, '_request') as wire, pytest.raises(FetchError, match='daily_budget'):
        search.search('test')
    wire.assert_not_called()


def test_local_candidates_survive_upstream_outage(seeded):
    upstream = SearXNGSearch(seeded)
    with patch.object(upstream, 'search', side_effect=FetchError('searxng_cooldown')):
        rows, evidence = CandidateSearch(seeded, upstream).search('Acme Care')
    assert rows and evidence['upstream_error']['code'] == 'searxng_cooldown'
    assert all(r['status'] == 'discovered_unverified' for r in rows)


def test_brave_http_disabled_by_default_even_with_key(store):
    fetch = Fetcher(store)
    with patch.object(fetch, '_wire') as wire, pytest.raises(FetchError, match='monthly_budget'):
        fetch.json('https://api.search.brave.com/res/v1/web/search?q=test', provider='brave', token='existing-key')
    wire.assert_not_called()


def test_monthly_cap_counts_actual_http_requests_and_retries(store):
    fetch = Fetcher(store, search_monthly_limit=1)
    with patch.object(fetch, '_wire', return_value=(503, {}, b'')) as wire, patch('enrichment_v4.http.time.sleep'), pytest.raises(FetchError, match='monthly_budget'):
        fetch.json('https://api.search.brave.com/res/v1/web/search?q=test', provider='brave', token='existing-key')
    assert wire.call_count == 1


def test_monthly_budget_atomic_and_survives_pruning(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(lambda _: store.monthly_search_budget('brave', 2), range(4)))
    assert sum(accepted) == 2
    store.prune()
    assert not Store(store.path).monthly_search_budget('brave', 2)
