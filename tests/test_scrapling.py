import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from enrichment_v4.api import create_app
from enrichment_v4.http import DEADLINE, FetchError
from enrichment_v4.models import DiscoveryRequest
from enrichment_v4.scrapling_search import ScraplingSearch, parse_results, result_url, _BROWSER_SLOT
from enrichment_v4.scrapling_worker import allowed_request, browser_setup, pinned_hosts, render
from enrichment_v4.service import EnrichmentService
from enrichment_v4.store import Store
from test_v4 import FakeFetch, request

BING = '<ol><li class="b_algo"><h2><a href="https://acme-care.co.uk/">Acme Care Limited</a></h2><p>John Smith verified director</p></li></ol>'


@pytest.fixture
def search(tmp_path):
    return ScraplingSearch(Store(tmp_path / 'state.db'), sys.executable)


def response(html=BING, status=200):
    return {'url': 'https://www.bing.com/search?q=care', 'html': html, 'status': status}


@pytest.mark.parametrize('engine,html', [
    ('bing', BING),
    ('google', '<div id="search"><a href="/url?q=https%3A%2F%2Facme-care.co.uk"><h3>Acme Care Limited</h3></a></div>'),
    ('duckduckgo', '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Facme-care.co.uk">Acme Care Limited</a>'),
])
def test_organic_links_are_candidates_not_people(engine, html):
    rows = parse_results(html, engine)
    assert rows == [{'url': 'https://acme-care.co.uk/', 'title': 'Acme Care Limited', 'snippet': '', 'status': 'discovered_unverified'}]


def test_bing_tracking_link_decoded_without_fetching():
    value = base64.urlsafe_b64encode(b'https://acme-care.co.uk/').decode().rstrip('=')
    assert result_url('https://www.bing.com/ck/a?u=a1' + value, 'bing') == 'https://acme-care.co.uk/'


@pytest.mark.parametrize('link', ['javascript:alert(1)', 'file:///etc/passwd', 'http://127.0.0.1/', 'https://169.254.169.254/', 'https://user:pass@example.com', 'https://host.internal/', 'https://www.bing.com/search?q=x'])
def test_unsafe_or_engine_links_rejected(link):
    with pytest.raises(FetchError):
        result_url(link, 'bing')


def test_cache_and_no_automatic_pagination(search):
    with patch.object(search, '_render', return_value=response()) as browser:
        rows, meta = search.search('care')
        assert meta['engine'] == 'bing'
        assert search.search('care')[0] == rows
        assert search.search('care', offset=1)[0] == []
        assert browser.call_count == 1


@pytest.mark.parametrize('data,error', [
    (response('<h1>Verify you are human</h1>'), 'bot_challenge'),
    (response('<h1>New search layout</h1>'), 'no_recognised_results'),
    ({**response(), 'url': 'https://consent.google.com/'}, 'consent_or_redirect'),
    ({**response(status=429), 'retry_after': '7200'}, 'http_429'),
])
def test_block_or_unknown_page_stops_all_engines_after_restart(search, data, error):
    with patch.object(search, '_render', return_value=data), pytest.raises(FetchError, match=error):
        search.search('care')
    other = ScraplingSearch(Store(search.store.path), sys.executable, engine='google')
    with patch.object(other, '_render') as browser, pytest.raises(FetchError, match='cooldown'):
        other.search('different query')
    browser.assert_not_called()
    if data['status'] == 429:
        assert search.store.reserve('scrapling-search', 30)['blocked_until'] >= time.time() + 7100


def test_budget_blocks_before_browser(search):
    search.daily_limit = 0
    with patch.object(search, '_render') as browser, pytest.raises(FetchError, match='daily_budget'):
        search.search('care')
    browser.assert_not_called()


def test_one_browser_at_a_time(search):
    _BROWSER_SLOT.acquire()
    try:
        with pytest.raises(FetchError, match='busy'):
            search.search('care')
    finally:
        _BROWSER_SLOT.release()


def test_deadline_applies_before_browser(search):
    token = DEADLINE.set(time.monotonic() - 1)
    try:
        with patch.object(search, '_render') as browser, pytest.raises(FetchError, match='budget'):
            search.search('care')
        browser.assert_not_called()
    finally:
        DEADLINE.reset(token)


def test_browser_failure_keeps_local_candidates(search):
    search.store.index_candidates([{'url': 'https://acme-care.co.uk', 'title': 'Acme care home Hampshire'}])
    service = EnrichmentService(FakeFetch(), search.store, 'ch', search_provider='scrapling', scrapling_options={'python': sys.executable})
    with patch.object(service.search.upstream, '_render', side_effect=FetchError('scrapling_browser_failed')):
        rows, evidence = service.search.search('Acme care')
    assert rows and evidence['upstream_error']['code'] == 'scrapling_browser_failed'


def test_scraped_director_claim_cannot_clear_resigned_officer(search):
    fake = FakeFetch()
    fake.officers[0]['resigned_on'] = '2026-01-01'
    service = EnrichmentService(fake, search.store, 'ch', search_provider='scrapling', scrapling_options={'python': sys.executable})
    with patch.object(service.search.upstream, '_render', return_value=response()), patch('enrichment_v4.scrapling_search.time.sleep'):
        record = service.enrich(request())
    assert not record['call_ready'] and record['call_contact'] is None


def test_api_discovery_reaches_browser_adapter(search):
    service = EnrichmentService(FakeFetch(), search.store, 'ch', search_provider='scrapling', scrapling_options={'python': sys.executable})
    with TestClient(create_app(service, 'k' * 32)) as client, patch.object(service.search.upstream, '_render', return_value=response()) as browser:
        result = client.post('/discover-businesses', headers={'X-API-Key': 'k' * 32}, json={'query': 'care', 'location': 'Hampshire'})
        assert result.status_code == 200
        assert browser.call_count == 1
        assert not result.json()['results'][0]['call_ready']


def test_worker_pins_public_dns_and_denies_unknown_hosts():
    with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', ('93.184.216.34', 443))]):
        rules, hosts = pinned_hosts('bing')
    assert 'MAP www.bing.com 93.184.216.34' in rules and rules.endswith('MAP * ~NOTFOUND')
    assert allowed_request('https://www.bing.com/search?q=x', hosts)
    for url in ('http://www.bing.com', 'https://www.bing.com:8443', 'https://www.bing.com.evil.com', 'https://127.0.0.1', 'https://user@www.bing.com'):
        assert not allowed_request(url, hosts)


def test_mixed_private_dns_answers_fail_closed():
    with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', ('93.184.216.34', 443)), (2, 1, 6, '', ('127.0.0.1', 443))]), pytest.raises(ValueError, match='dns_unavailable'):
        pinned_hosts('bing')


def test_browser_guard_blocks_subresources_and_caps_requests():
    page = Mock()
    browser_setup({'www.bing.com'})(page)
    handler = page.context.route.call_args.args[1]
    unsafe = Mock(request=SimpleNamespace(url='http://169.254.169.254/', method='GET'))
    handler(unsafe)
    unsafe.abort.assert_called_once()
    safe = Mock(request=SimpleNamespace(url='https://www.bing.com/search', method='GET'))
    for _ in range(80):
        handler(safe)
    assert safe.continue_.call_count == 79
    safe.abort.assert_called_once()
    page.context.route_web_socket.assert_called_once()


def test_setup_error_closes_context_before_scrapling_can_swallow_it():
    page = Mock()
    page.context.route.side_effect = RuntimeError('API changed')
    with pytest.raises(RuntimeError):
        browser_setup({'www.bing.com'})(page)
    page.context.close.assert_called_once()


def test_worker_passes_supported_scrapling_arguments():
    fetcher = Mock()
    fetcher.fetch.return_value = SimpleNamespace(url='https://www.bing.com/search', status=200, html_content=BING, headers={})
    with patch.dict(sys.modules, {'scrapling.fetchers': SimpleNamespace(StealthyFetcher=fetcher)}), patch('enrichment_v4.scrapling_worker.pinned_hosts', return_value=('MAP * ~NOTFOUND', {'www.bing.com'})):
        assert render({'url': 'https://www.bing.com/search?q=x', 'engine': 'bing', 'solve_cloudflare': True})['html'] == BING
    args = fetcher.fetch.call_args.kwargs
    assert args['solve_cloudflare'] is True and args['retries'] == 1
    assert args['additional_args']['service_workers'] == 'block'
    assert '--no-proxy-server' in args['extra_flags']


def test_worker_never_accepts_arbitrary_target():
    with pytest.raises(ValueError, match='invalid_search_url'):
        render({'engine': 'google', 'url': 'http://169.254.169.254/'})


def test_subprocess_has_no_api_keys_or_proxy_environment(search, monkeypatch):
    monkeypatch.setenv('COMPANIES_HOUSE_API_KEY', 'not-for-browser')
    monkeypatch.setenv('HTTPS_PROXY', 'http://not-for-browser')
    class FakeProcess:
        pid = 1234567
        returncode = 0
        def __init__(self, args, **kwargs):
            assert 'COMPANIES_HOUSE_API_KEY' not in kwargs['env'] and 'HTTPS_PROXY' not in kwargs['env']
            assert args[1] == '-I' and kwargs['start_new_session']
            kwargs['stdout'].write(json.dumps(response()).encode())
        def communicate(self, *args, **kwargs):
            pass
        def wait(self):
            pass
    with patch('subprocess.Popen', FakeProcess), patch('os.killpg') as kill:
        assert search._render('https://www.bing.com/search?q=x')['status'] == 200
    kill.assert_called_once()


def test_subprocess_timeout_reaps_browser(search):
    process = Mock(pid=1234567)
    process.communicate.side_effect = subprocess.TimeoutExpired('browser', 75)
    with patch('subprocess.Popen', return_value=process), patch('os.killpg') as kill, pytest.raises(FetchError, match='timeout'):
        search._render('https://www.bing.com/search?q=x')
    kill.assert_called_once()
    process.wait.assert_called_once()
