"""Download one permit source into data/raw/<key>/page_*.parquet.

    python -m acquisition.fetch acquisition/sources/austin_tx.json --out data/raw
    python -m acquisition.fetch acquisition/sources/austin_tx.json --out data/raw --limit 100   # quick check
"""
import argparse
import json
from pathlib import Path

from . import arcgis, carto, socrata

PLATFORMS = {'socrata': socrata.fetch, 'arcgis': arcgis.fetch, 'carto': carto.fetch}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spec', nargs='+', help='source specification JSON file(s)')
    p.add_argument('--out', default='data/raw')
    p.add_argument('--limit', type=int, help='fetch at most this many records per source')
    a = p.parse_args()
    for path in a.spec:
        spec = json.loads(Path(path).read_text(encoding='utf-8'))
        report = PLATFORMS[spec['platform']](spec, a.out, limit=a.limit)
        print(json.dumps({k: report[k] for k in ('source_key', 'records', 'source_count')}))


if __name__ == '__main__':
    main()
