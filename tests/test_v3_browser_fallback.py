"""Exercise the legacy function in isolation without GPU/browser dependencies."""
import ast
import logging
from pathlib import Path
import re
import time
from urllib.parse import quote_plus, urlsplit
from bs4 import BeautifulSoup
import pytest


@pytest.mark.parametrize('filename', ['main.py', 'EnrichmentV3.sh'])
def test_legacy_search_fallback_uses_browser_not_static_http(filename):
    source = Path(filename).read_text()
    if filename.endswith('.sh'):
        source = source.split("cat <<'PYEOF' > $APP_DIR/app/main.py\n", 1)[1].split('\nPYEOF', 1)[0]
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_search_with_crawl4ai')
    visited = []

    def browser(url):
        visited.append(url)
        if 'duckduckgo' in url:
            return '<a class="result__a" href="https://acme.example/">Acme care</a>'
        return '<li class="b_algo"><h2><a href="https://acme2.example/">Acme care</a></h2></li>'

    def static(url):
        raise AssertionError('Browser fallback incorrectly invoked static HTTP')

    namespace = {'List': list, 'Dict': dict, '_CRAWL4AI_AVAILABLE': True, 'quote_plus': quote_plus, 'BeautifulSoup': BeautifulSoup,
                 'fetch_html_with_crawl4ai': browser, 'fetch_html_from_url': static, 'parse_ddg_result_url': lambda x: x,
                 'normalize_domain': lambda u: urlsplit(u).hostname, 're': re, 'time': time, 'logger': logging.getLogger('test')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
    hits = namespace['_search_with_crawl4ai']('Acme care', 5)
    assert len(visited) == 2
    assert {h['domain'] for h in hits} == {'acme.example', 'acme2.example'}
