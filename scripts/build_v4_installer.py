"""Rebuild the deterministic standalone shell installer from reviewable sources."""
import base64
import hashlib
import io
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
files = sorted((ROOT / 'enrichment_v4').glob('*.py')) + [ROOT / p for p in ('requirements-v4.txt', 'requirements-v4-lock.txt', 'README-V4.md')]
files += sorted(p for p in (ROOT / "deploy/searxng").glob("*") if p.name in {"compose.yml", "settings.yml", "setup.sh"})
files += sorted(p for p in (ROOT / "deploy/scrapling").glob("*") if p.name in {"requirements-lock.txt", "setup.sh"})
files.append(ROOT / "SEARCH-OPTIONS.md")
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        entry = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o100644 << 16
        archive.writestr(entry, path.read_bytes())
payload = buffer.getvalue()
template = (ROOT / 'scripts/installer-v4.template.sh').read_text()
script = template.replace('__PAYLOAD_SHA256__', hashlib.sha256(payload).hexdigest()) + base64.encodebytes(payload).decode()
(ROOT / 'EnrichmentV4.sh').write_text(script)
(ROOT / 'EnrichmentV4.sh').chmod(0o755)
print('Built EnrichmentV4.sh')
