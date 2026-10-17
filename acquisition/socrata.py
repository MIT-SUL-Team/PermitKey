"""Socrata (SODA) datasets, paged by the system row id (:id) so pages never skip or repeat rows."""
from .common import PageWriter, get_json


def fetch(spec, out_dir, page_size=10000, limit=None):
    base = spec['url'].rstrip('/')
    resource = f"{base}/resource/{spec['dataset']}.json"
    meta = get_json(f"{base}/api/views/{spec['dataset']}.json")
    fields = [c['fieldName'] for c in meta['columns'] if not c['fieldName'].startswith(':')]
    select = ':id as source_system_row_id,' + ','.join(fields)
    expected = int(get_json(resource, {'$select': 'count(*)', '$where': spec.get('where', '1=1')})[0]['count'])
    writer, last = PageWriter(spec, out_dir), None
    target = expected if limit is None else min(expected, limit)
    while writer.rows < target:
        where = spec.get('where', '1=1') + (f" AND :id > '{last}'" if last else '')
        rows = get_json(resource, {'$select': select, '$where': where, '$order': ':id',
                                   '$limit': min(page_size, target - writer.rows)})
        if not rows:
            break
        last = rows[-1]['source_system_row_id']
        writer.write(rows, lambda r: r['source_system_row_id'])
    if limit is None and writer.rows != expected:
        raise RuntimeError(f'{spec["key"]}: fetched {writer.rows}, source reports {expected}')
    return writer.finish(source_count=expected, dataset_name=meta.get('name'))
