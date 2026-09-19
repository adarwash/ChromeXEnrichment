#!/usr/bin/env bash
# Optional browser environment, independent from the verification service venv.
set -Eeuo pipefail
umask 022
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRAPLING_VENV="${SCRAPLING_VENV:-/opt/chromex-enrichment-v4/scrapling}"
[[ "$SCRAPLING_VENV" = /* && "$SCRAPLING_VENV" != *..* ]] || { echo 'Use an absolute SCRAPLING_VENV path.' >&2; exit 2; }
python3 -m venv "$SCRAPLING_VENV"
"$SCRAPLING_VENV/bin/pip" install --disable-pip-version-check -r "$SCRIPT_DIR/requirements-lock.txt"
"$SCRAPLING_VENV/bin/pip" check
export PLAYWRIGHT_BROWSERS_PATH="$SCRAPLING_VENV/browsers"
"$SCRAPLING_VENV/bin/patchright" install --with-deps chromium
chmod -R a+rX "$SCRAPLING_VENV"
echo 'Browser dependencies installed. To enable, edit the V4 environment file:'
echo 'SEARCH_PROVIDER="scrapling"'
echo "SCRAPLING_PYTHON=\"$SCRAPLING_VENV/bin/python\""
echo 'SCRAPLING_ENGINE="bing"'
echo 'SCRAPLING_BROWSER_PROFILE="native"'
echo 'SCRAPLING_DAILY_LIMIT="25"'
echo 'SCRAPLING_SOLVE_CLOUDFLARE="0"'
echo 'Then restart chromex-enrichment-v4. See SEARCH-OPTIONS.md for limitations.'
