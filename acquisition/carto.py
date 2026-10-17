"""Carto SQL API tables, paged by an increasing integer id (cartodb_id by default)."""
from .common import PageWriter, get_json


def fetch(spec, out_dir, page_size=20000, limit=None):
    api, table = spec['url'], spec['table']
    key = spec.get('id_field', 'cartodb_id')
    expected = int(get_json(api, {'q': f'select count(*) from {table}'})['rows'][0]['count'])
    target = expected if limit is None else min(expected, limit)
    writer, last = PageWriter(spec, out_dir), -1
    while writer.rows < target:
        n = min(page_size, target - writer.rows)
        rows = get_json(api, {'q': f'select * from {table} where {key} > {last} order by {key} limit {n}'})['rows']
        if not rows:
            break
        last = int(rows[-1][key])
        writer.write(rows, lambda r: r[key])
    if limit is None and writer.rows != expected:
        raise RuntimeError(f'{spec["key"]}: fetched {writer.rows}, source reports {expected}')
    return writer.finish(source_count=expected)
