"""Shared helpers: HTTP with retries, paged Parquet output, completion report."""
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

USER_AGENT = 'PermitKey/1.0 (+https://github.com/MIT-SUL-Team/PermitKey)'

SCHEMA = pa.schema([
    ('source_key', pa.string()),
    ('source_row_ordinal', pa.int64()),
    ('source_system_row_id', pa.string()),
    ('permit_number', pa.string()),
    ('description', pa.string()),
    ('record_type', pa.string()),
    ('raw', pa.string()),
])


def get_json(url, params=None, post=False, retries=6, timeout=120):
    """GET (or form-POST for long queries) a JSON document, retrying with backoff."""
    query = urllib.parse.urlencode(params or {})
    for attempt in range(retries):
        try:
            if post:
                req = urllib.request.Request(url, data=query.encode(), headers={
                    'User-Agent': USER_AGENT, 'Content-Type': 'application/x-www-form-urlencoded'})
            else:
                req = urllib.request.Request(url + ('?' + query if query else ''), headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            if isinstance(data, dict) and data.get('error'):
                raise RuntimeError(data['error'])
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))


def first_text(record, fields):
    """First non-empty value among the named fields, as text."""
    for f in fields or []:
        v = record.get(f)
        if v not in (None, ''):
            return str(v)
    return ''


def joined_text(record, fields):
    return ' | '.join(str(record[f]) for f in fields or [] if record.get(f) not in (None, ''))


class PageWriter:
    """Writes one Parquet page per request; every source row is kept, including repeated permit numbers."""

    def __init__(self, spec, out_dir):
        self.spec = spec
        self.dir = Path(out_dir) / spec['key']
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows = 0
        self.pages = []

    def write(self, records, row_id):
        out = [{
            'source_key': self.spec['key'],
            'source_row_ordinal': self.rows + i,
            'source_system_row_id': str(row_id(r)),
            'permit_number': str(r.get(self.spec.get('permit_number_field')) or ''),
            'description': first_text(r, self.spec.get('description_fields')),
            'record_type': joined_text(r, self.spec.get('record_type_fields')),
            'raw': json.dumps(r, ensure_ascii=False, default=str, separators=(',', ':')),
        } for i, r in enumerate(records)]
        target = self.dir / f'page_{self.rows:012d}.parquet'
        partial = target.with_suffix('.partial')
        pq.write_table(pa.Table.from_pylist(out, schema=SCHEMA), partial, compression='zstd')
        partial.replace(target)
        self.pages.append(target)
        self.rows += len(out)
        print(self.spec['key'], self.rows, flush=True)

    def finish(self, **extra):
        report = {
            'source_key': self.spec['key'],
            'platform': self.spec['platform'],
            'jurisdiction': self.spec.get('jurisdiction'),
            'source_url': self.spec['url'],
            'records': self.rows,
            'pages': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.pages],
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'permit_number_deduplication': False,
        }
        report.update(extra)
        (self.dir / 'acquisition_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report
