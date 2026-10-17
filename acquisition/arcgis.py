"""ArcGIS REST feature layers: snapshot the object ids first, then fetch them in fixed batches."""
from .common import PageWriter, get_json


def fetch(spec, out_dir, batch=500, limit=None):
    layer = spec['url'].rstrip('/')
    meta = get_json(layer, {'f': 'json'})
    oid = meta.get('objectIdField') or next(f['name'] for f in meta['fields'] if f['type'] == 'esriFieldTypeOID')
    ids = sorted(get_json(f'{layer}/query', {'f': 'json', 'where': spec.get('where', '1=1'), 'returnIdsOnly': 'true'})['objectIds'] or [])
    total = len(ids)
    if limit is not None:
        ids = ids[:limit]
    writer = PageWriter(spec, out_dir)
    for start in range(0, len(ids), batch):
        wanted = ids[start:start + batch]
        result = get_json(f'{layer}/query', {'f': 'json', 'objectIds': ','.join(map(str, wanted)),
                                              'outFields': '*', 'returnGeometry': 'false'}, post=True)
        rows = [f['attributes'] for f in result['features']]
        if {r[oid] for r in rows} != set(wanted):
            raise RuntimeError(f'{spec["key"]}: batch at {start} returned a different set of object ids')
        writer.write(rows, lambda r: r[oid])
    return writer.finish(source_count=total, dataset_name=meta.get('name'), object_id_field=oid)
