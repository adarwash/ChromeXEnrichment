import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from enrichment_v4.api import create_app
from enrichment_v4.evidence import parse_page, legal_binding, director_support, valid_phone
from enrichment_v4.http import Fetcher, FetchError, canonical_url, public_addresses, retry_delay, publisher
from enrichment_v4.models import EnrichRequest, DiscoveryRequest, company_number
from enrichment_v4.providers import CompaniesHouse, person_name
from enrichment_v4.service import EnrichmentService
from enrichment_v4.store import Store

PROFILE = {'company_number': '01234567', 'company_name': 'ACME CARE LIMITED', 'company_status': 'active', 'registered_office_address': {'postal_code': 'PO15 7AD'}}
OFFICER = {'name': 'SMITH, John David', 'officer_role': 'director', 'appointed_on': '2010-02-01', 'links': {'self': '/company/01234567/appointments/abc'}}
OFFICIAL = '<html><p>Acme Care Limited. Company number: 01234567.</p><p>John David Smith — Managing Director</p><p>Call 020 7946 0958</p><p>info@acme-care.co.uk</p><a href="/team">Team</a></html>'
EXTERNAL = '<html><p>Acme Care Limited. Company number: 01234567. Office PO15 7AD.</p><p>Telephone: +44 20 7946 0958</p></html>'


def response(url, html, age=0):
    return {'url': url, 'text': html, 'sha256': hashlib.sha256(html.encode()).hexdigest(), 'fetched_at': time.time() - age, 'content_type': 'text/html'}


class FakeFetch:
    def __init__(self):
        self.profile = copy.deepcopy(PROFILE)
        self.officers = [copy.deepcopy(OFFICER)]
        self.pages = {'https://acme-care.co.uk/': OFFICIAL, 'https://acme-care.co.uk/team': OFFICIAL, 'https://cqc.org.uk/location/acme': EXTERNAL}
        self.fail_officers = False
        self.searches = []
        self.calls = []
        self.age = 0

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url not in self.pages:
            raise FetchError('http_404', url)
        if isinstance(self.pages[url], Exception):
            raise self.pages[url]
        return response(url, self.pages[url], self.age)

    def json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        ev = {'url': url, 'fetched_at': time.time() - self.age, 'sha256': 'fixture-hash'}
        if '/officers?' in url:
            if self.fail_officers:
                raise FetchError('http_429', url)
            return {'items': copy.deepcopy(self.officers), 'total_results': len(self.officers), 'start_index': 0}, ev
        if '/search/companies?' in url:
            rows = self.searches or [{'title': self.profile['company_name'], 'company_number': '01234567', 'address': {'postal_code': 'PO15 7AD'}}]
            return {'items': rows}, ev
        if 'api.search.brave.com' in url:
            return {'web': {'results': [{'url': 'https://acme-care.co.uk/', 'title': 'Made-up director: Jane Doe'}, {'url': 'https://cqc.org.uk/location/acme'}]}}, ev
        return copy.deepcopy(self.profile), ev


@pytest.fixture
def setup(tmp_path):
    fetch = FakeFetch()
    store = Store(tmp_path / 'state.db')
    service = EnrichmentService(fetch, store, 'test-ch', 'test-search', search_provider='brave')
    return service, fetch, store


def request(**kwargs):
    return EnrichRequest(company_number='01234567', company_name='ACME CARE LIMITED', domain='https://acme-care.co.uk/', **kwargs)


def test_complete_evidence_is_ready_and_auditable(setup):
    service, _, store = setup
    record = service.enrich(request())
    assert record['call_ready'], record['blocking_reasons']
    assert record['call_contact']['verified_director_names'] == ['John David Smith']
    assert record['call_contact']['business_phone'] == '+442079460958'
    assert record['emails'][0]['deliverability'] == 'unknown'
    assert len(record['field_evidence']['phones']) >= 2
    assert store.get(record['record_id'])['result'] == record
    assert 'date_of_birth' not in json.dumps(record)


@pytest.mark.parametrize('status', ['dissolved', 'liquidation', 'dormant', 'closed', 'receivership'])
def test_non_active_company_blocked(setup, status):
    service, fetch, _ = setup
    fetch.profile['company_status'] = status
    record = service.enrich(request())
    assert record['status'] == 'blocked'
    assert not record['call_ready'] and record['call_contact'] is None


@pytest.mark.parametrize('change, reason', [
    ({'resigned_on': '2026-01-01'}, 'resigned_or_ceased'),
    ({'ceased_on': '2026-01-01'}, 'resigned_or_ceased'),
    ({'officer_role': 'corporate-director'}, 'not_natural_person_director'),
    ({'officer_role': 'nominee-director'}, 'not_natural_person_director'),
    ({'officer_role': 'secretary'}, 'not_natural_person_director'),
    ({'officer_role': 'llp-member'}, 'not_natural_person_director'),
    ({'appointed_on': '2999-01-01'}, 'future_appointment'),
    ({'appointed_on': 'not-a-date'}, 'invalid_appointment_date'),
    ({'appointed_on': None}, 'appointment_date_missing'),
])
def test_invalid_directors_never_clear(setup, change, reason):
    service, fetch, _ = setup
    fetch.officers[0].update(change)
    result = service.enrich(request())
    assert not result['call_ready']
    assert result['directors'] == []
    assert result['excluded_officers'][0]['reason'] == reason


def test_pre_1992_director_allowed(setup):
    service, fetch, _ = setup
    fetch.officers[0].pop('appointed_on')
    fetch.officers[0]['is_pre_1992_appointment'] = True
    assert service.enrich(request())['call_ready']


def test_wrong_legal_entity_name_is_not_overridden_by_number(setup):
    service, _, _ = setup
    result = service.enrich(EnrichRequest(company_number='01234567', company_name='ACME HOLDINGS LIMITED', domain='https://acme-care.co.uk/'))
    assert not result['call_ready']
    assert 'requested_name_differs_from_legal_entity' in result['blocking_reasons']


def test_ambiguous_legal_name_is_not_first_result(setup):
    service, fetch, _ = setup
    fetch.searches = [{'title': 'ACME CARE LIMITED', 'company_number': '01234567'}, {'title': 'ACME CARE LIMITED', 'company_number': '08765432'}]
    result = service.enrich(EnrichRequest(company_name='ACME CARE LIMITED'))
    assert 'company_identity_ambiguous' in result['blocking_reasons']
    assert not result['registered_directors']


def test_fuzzy_name_not_used(setup):
    service, _, _ = setup
    result = service.enrich(EnrichRequest(company_name='ACME CAR LIMITED'))
    assert 'no_exact_legal_name_match' in result['blocking_reasons']


def test_domain_only_discovers_registry_identity(setup):
    service, _, _ = setup
    assert service.enrich(EnrichRequest(domain='https://acme-care.co.uk/'))['call_ready']


def test_missing_registry_key_fails_closed(setup):
    service, _, _ = setup
    service.ch.token = ''
    result = service.enrich(request())
    assert not result['call_ready'] and not result['directors']


def test_no_search_key_can_use_explicit_sources(setup):
    service, _, _ = setup
    service.search.upstream.token = ''
    result = service.enrich(request(corroboration_urls=['https://cqc.org.uk/location/acme']))
    assert result['call_ready']
    assert any(e['code'] == 'brave_search_key_missing' for e in result['source_errors'])


def test_registry_failure_does_not_fall_back_to_website_people(setup):
    service, fetch, _ = setup
    fetch.fail_officers = True
    result = service.enrich(request())
    assert not result['call_ready']
    assert result['directors'] == []
    assert result['call_contact'] is None


@pytest.mark.parametrize('replacement', ['John Smith — Managing Director', 'J D Smith — Director', 'Jane Doe — Managing Director', 'John David Smith — Former Director'])
def test_unmatched_or_former_website_director_not_cleared(setup, replacement):
    service, fetch, _ = setup
    for url in ('https://acme-care.co.uk/', 'https://acme-care.co.uk/team'):
        fetch.pages[url] = OFFICIAL.replace('John David Smith — Managing Director', replacement)
    result = service.enrich(request())
    assert not result['call_ready']
    assert not result['directors']


def test_requested_person_must_match_current_verified_director(setup):
    service, _, _ = setup
    result = service.enrich(request(director_name='Jane Doe'))
    assert 'requested_director_not_cross_verified' in result['blocking_reasons']
    assert not result['call_ready']


def test_same_name_multiple_appointments_requires_review(setup):
    service, fetch, _ = setup
    other = copy.deepcopy(OFFICER)
    other['links']['self'] += '2'
    fetch.officers.append(other)
    assert not service.enrich(request())['call_ready']


def test_wrong_phone_hint_is_never_inserted(setup):
    service, _, _ = setup
    result = service.enrich(request(phone_number='020 7946 0959'))
    assert not result['call_ready']
    assert '+442079460959' not in result['phones']
    assert 'supplied_phone_not_cross_verified' in result['blocking_reasons']


def test_other_entity_external_phone_not_evidence(setup):
    service, fetch, _ = setup
    fetch.pages['https://cqc.org.uk/location/acme'] = EXTERNAL.replace('01234567', '08765432')
    assert not service.enrich(request())['call_ready']


def test_conflicting_phone_blocks_even_with_other_support(setup):
    service, fetch, _ = setup
    fetch.pages['https://nhs.uk/acme'] = EXTERNAL.replace('0958', '0959')
    result = service.enrich(request(corroboration_urls=['https://nhs.uk/acme']))
    assert not result['call_ready']
    assert 'conflicting_business_phones' in result['blocking_reasons']


def test_official_multiple_pages_not_independent(setup):
    service, fetch, _ = setup
    fetch.pages['https://cqc.org.uk/location/acme'] = FetchError('robots_disallowed')
    assert not service.enrich(request(corroboration_urls=['https://acme-care.co.uk/team']))['call_ready']


def test_identical_mirrored_html_not_independent(setup):
    service, fetch, _ = setup
    fetch.pages['https://cqc.org.uk/location/acme'] = OFFICIAL
    assert not service.enrich(request())['call_ready']


def test_unapproved_source_cannot_be_client_promoted(setup):
    service, fetch, _ = setup
    fetch.pages['https://cqc.org.uk/location/acme'] = FetchError('http_404')
    fetch.pages['https://fake-directory.co.uk/acme'] = EXTERNAL
    result = service.enrich(request(corroboration_urls=['https://fake-directory.co.uk/acme']))
    assert not result['call_ready']
    assert any('Unapproved' in s for s in result['warnings'])


def test_stale_sources_fail_closed(setup):
    service, fetch, _ = setup
    fetch.age = 86401
    result = service.enrich(request())
    assert not result['call_ready']
    assert 'evidence_expired_or_clock_invalid' in result['blocking_reasons']


def test_wrong_website_does_not_silently_switch(setup):
    service, fetch, _ = setup
    fetch.pages['https://acme-care.co.uk/'] = '<p>Different Ltd company number: 99999999</p>'
    result = service.enrich(request())
    assert not result['call_ready']
    assert 'official_website_ambiguous_or_unconfirmed' in result['blocking_reasons']


def test_publisher_uses_registrable_domain():
    assert publisher('https://www.acme.co.uk/team') == publisher('https://sales.acme.co.uk/')
    assert publisher('https://a.github.io/') != publisher('https://b.github.io/')


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'https://user:pass@example.com/', 'http://localhost/', 'https://example.com:8080/', 'https://example.com/\r\nX-Test:bad'])
def test_unsafe_url_rejected(url):
    with pytest.raises(FetchError):
        canonical_url(url)


def test_dns_mixed_public_private_rejected():
    answers = [(2, 1, 6, '', ('8.8.8.8', 443)), (2, 1, 6, '', ('127.0.0.1', 443))]
    with patch('socket.getaddrinfo', return_value=answers), pytest.raises(FetchError, match='non_public_address'):
        public_addresses('example.com', 443)


def test_dns_pin_connects_to_checked_numeric_address():
    from enrichment_v4.http import _public_connection
    with patch('enrichment_v4.http.public_addresses', return_value=['8.8.8.8']), patch('socket.create_connection') as connect:
        _public_connection(('example.com', 443))
        assert connect.call_args.args[0] == ('8.8.8.8', 443)


def test_retry_after_seconds_and_date():
    assert retry_delay('600') == 600
    assert retry_delay('bad') == 300
    assert retry_delay('Wed, 21 Oct 2015 07:28:00 GMT', now=1445412180) == 300
    assert retry_delay('Wed, 21 Oct 2015 07:28:00 GMT', now=1445411880) == 600


def test_persistent_cooldown_shared_across_instances(tmp_path):
    first, second = Store(tmp_path / 's.db'), Store(tmp_path / 's.db')
    first.block('host', 500)
    assert second.reserve('host', 3, now=100)['blocked_until'] == 500


def test_concurrent_rate_reservations_are_atomic(tmp_path):
    store = Store(tmp_path / 's.db')
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: store.reserve('host', 1, now=100), range(5)))
    assert sorted(r['wait'] for r in results) == [0, 1, 2, 3, 4]


def test_expired_cache_not_returned(tmp_path):
    store = Store(tmp_path / 's.db')
    store.cache('x', {'ok': True}, -1)
    assert store.cached('x') is None


def test_429_not_retried_and_block_persists(tmp_path):
    store = Store(tmp_path / 's.db')
    fetch = Fetcher(store)
    with patch.object(fetch, '_wire', return_value=(429, {'retry-after': '700'}, b'')) as wire:
        with pytest.raises(FetchError, match='http_429'):
            fetch.get('https://api.company-information.service.gov.uk/company/01234567', kind='api')
        with pytest.raises(FetchError, match='source_cooldown'):
            fetch.get('https://api.company-information.service.gov.uk/company/08765432', kind='api')
        assert wire.call_count == 1


def test_api_credentials_never_follow_cross_origin_redirect(tmp_path):
    fetch = Fetcher(Store(tmp_path / 's.db'))
    with patch.object(fetch, '_wire', return_value=(302, {'location': 'https://evil.example/'}, b'')) as wire:
        with pytest.raises(FetchError, match='api_cross_origin_redirect'):
            fetch.get('https://api.company-information.service.gov.uk/company/01234567', kind='api', headers={'Authorization': 'secret'})
        assert wire.call_count == 1


def test_robots_disallow_prevents_page_request(tmp_path):
    fetch = Fetcher(Store(tmp_path / 's.db'))
    with patch.object(fetch, '_wire', return_value=(200, {'content-type': 'text/plain'}, b'User-agent: *\nDisallow: /private')) as wire:
        with pytest.raises(FetchError, match='robots_disallowed'):
            fetch.get('https://example.com/private')
        assert wire.call_count == 1
        assert wire.call_args.args[0].endswith('/robots.txt')


def test_officers_paginate_and_filter_all_pages():
    ch = CompaniesHouse(None, 'key')
    first = copy.deepcopy(OFFICER)
    first['resigned_on'] = '2020-01-01'
    second = copy.deepcopy(OFFICER)
    second['links']['self'] += '2'
    pages = [({'items': [first], 'total_results': 2, 'start_index': 0}, {'fetched_at': time.time()}), ({'items': [second], 'total_results': 2, 'start_index': 1}, {'fetched_at': time.time()})]
    with patch.object(ch, '_get', side_effect=pages) as get:
        directors, excluded, evidence = ch.officers('01234567')
    assert len(directors) == len(excluded) == 1
    assert len(evidence) == 2 and 'start_index=1' in get.call_args.args[0]


@pytest.mark.parametrize('second', [
    {'items': [], 'total_results': 2, 'start_index': 1},
    {'items': [OFFICER], 'total_results': 3, 'start_index': 1},
    {'items': [OFFICER], 'total_results': 2, 'start_index': 0},
    {'items': [OFFICER], 'total_results': 2, 'start_index': 1},
])
def test_partial_changed_duplicate_officer_pages_fail(second):
    ch = CompaniesHouse(None, 'key')
    with patch.object(ch, '_get', side_effect=[({'items': [OFFICER], 'total_results': 2, 'start_index': 0}, {}), (second, {})]), pytest.raises(FetchError):
        ch.officers('01234567')


def test_api_auth_and_pre_call_refresh(setup):
    service, fetch, _ = setup
    key = 'x' * 32
    with TestClient(create_app(service, key)) as client:
        assert client.get('/health').status_code == 200
        assert client.post('/enrich', json=request().model_dump()).status_code == 401
        headers = {'X-API-Key': key}
        first = client.post('/enrich-verified', headers=headers, json=request().model_dump()).json()
        assert first['call_ready']
        fetch.officers[0]['resigned_on'] = '2026-01-01'
        fetch.calls.clear()
        second = client.post('/call-check/' + first['verified_record']['record_id'], headers=headers).json()
        assert not second['call_ready']
        assert second['verified_record']['call_contact'] is None
        assert all(kwargs.get('refresh') for url, kwargs in fetch.calls if 'api.search.brave.com' not in url)


def test_api_rejects_legacy_bypass_flags(setup):
    service, _, _ = setup
    key = 'x' * 32
    with TestClient(create_app(service, key)) as client:
        result = client.post('/enrich-verified', headers={'X-API-Key': key}, json={**request().model_dump(), 'require_verified_record': False})
        assert result.status_code == 422


def test_expired_stored_record_not_ready(setup):
    service, _, store = setup
    result = service.enrich(request())
    result['expires_at'] = time.time() - 1
    with store.connect() as db:
        db.execute('UPDATE records SET result=? WHERE id=?', (json.dumps(result), result['record_id']))
    with TestClient(create_app(service, 'x' * 32)) as client:
        fetched = client.get('/records/' + result['record_id'], headers={'X-API-Key': 'x' * 32}).json()
        assert not fetched['call_ready']
        assert fetched['verified_record']['call_contact'] is None


def test_discovery_is_unverified_without_enrichment(setup):
    service, _, _ = setup
    result = service.discover(DiscoveryRequest(query='care homes'))
    assert result['results']
    assert all(not row['call_ready'] for row in result['results'])


def test_company_numbers_preserve_prefixes_and_leading_zeroes():
    assert company_number('1234567') == '01234567'
    assert company_number('sc123456') == 'SC123456'
    with pytest.raises(ValueError):
        company_number('../../passwords')

@pytest.mark.parametrize('text', [
    'John David Smith works with Jane Doe, Managing Director',
    'John David Smith is not a director',
    'John David Smith is no longer Managing Director',
    'John David Smith met the director yesterday',
])
def test_unrelated_or_negated_role_not_verified(text):
    page = parse_page(response('https://acme-care.co.uk/team', '<p>' + text + '</p>'))
    current, _ = director_support(page, 'SMITH, John David')
    assert not current


def test_historical_news_article_not_director_evidence():
    page = parse_page(response('https://acme-care.co.uk/news/2015/appointment', OFFICIAL))
    assert director_support(page, 'SMITH, John David')[0] == []


def test_daily_source_budget_persists(tmp_path):
    store = Store(tmp_path / 's.db')
    assert store.budget('website', 1, now=100) == 0
    assert Store(tmp_path / 's.db').budget('website', 1, now=101) == 86400
    assert store.budget('website', 1, now=86401) == 0


def test_text_copy_with_changed_markup_does_not_corroborate(setup):
    service, fetch, _ = setup
    fetch.pages['https://cqc.org.uk/location/acme'] = OFFICIAL.replace('<html>', '<html><div>').replace('</html>', '</div></html>')
    assert not service.enrich(request())['call_ready']


def test_multicast_dns_rejected():
    with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', ('224.0.0.1', 443))]), pytest.raises(FetchError):
        public_addresses('example.com', 443)


def test_missing_robots_cached_without_repeated_requests(tmp_path):
    fetch = Fetcher(Store(tmp_path / 's.db'))
    with patch.object(fetch, '_wire', return_value=(404, {}, b'')) as wire:
        fetch._robots('https://example.com/one')
        fetch._robots('https://example.com/two')
        assert wire.call_count == 1


def test_expired_request_budget_makes_no_network_request(tmp_path):
    from enrichment_v4.http import DEADLINE
    fetch = Fetcher(Store(tmp_path / 's.db'))
    token = DEADLINE.set(time.monotonic() - 1)
    try:
        with patch.object(fetch, '_wire') as wire, pytest.raises(FetchError, match='request_budget_exhausted'):
            fetch.get('https://example.com/')
        wire.assert_not_called()
    finally:
        DEADLINE.reset(token)
