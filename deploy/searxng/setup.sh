#!/usr/bin/env bash
# Run explicitly on your Linux host with an existing Docker Compose installation.
# Does not change V4's provider selection or billing settings.
set -Eeuo pipefail
umask 077
cd -- "$(dirname -- "$0")"
command -v docker >/dev/null || { echo 'Install Docker and Compose before running this optional companion.' >&2; exit 1; }
docker compose version >/dev/null
if [[ ! -e .env ]]; then
  docker pull docker.io/searxng/searxng:latest
  image_ref="$(docker image inspect docker.io/searxng/searxng:latest --format '{{index .RepoDigests 0}}')"
  [[ "$image_ref" == *@sha256:* ]] || { echo 'Could not resolve an immutable image digest.' >&2; exit 1; }
  secret="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  printf 'SEARXNG_IMAGE=%s\nSEARXNG_SECRET=%s\n' "$image_ref" "$secret" > .env
  chmod 0600 .env
fi
docker compose config --quiet
docker compose up -d
echo 'SearXNG launched on http://127.0.0.1:8888. Check its logs for engine availability.'
echo 'To opt in, set SEARCH_PROVIDER=searxng in the V4 environment file and restart V4.'
