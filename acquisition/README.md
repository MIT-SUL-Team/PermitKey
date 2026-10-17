# Acquisition

General downloaders for the three open-data platforms that supply most PermitKey sources.
Each writes every source row (no de-duplication on permit number) to
`data/raw/<key>/page_*.parquet` with the raw record preserved as JSON, plus an
`acquisition_report.json` with page checksums.

| Platform | Module | Paging | Example source |
| --- | --- | --- | --- |
| Socrata | `socrata.py` | system row id (`:id`) | `sources/austin_tx.json` |
| ArcGIS REST | `arcgis.py` | object-id snapshot, fixed batches | `sources/mecklenburg_county_nc.json` |
| Carto SQL | `carto.py` | increasing `cartodb_id` | `sources/philadelphia_pa.json` |

```
python -m acquisition.fetch acquisition/sources/austin_tx.json --out data/raw
python -m acquisition.fetch acquisition/sources/*.json --out data/raw --limit 100
```

A source specification names the platform, endpoint, permit-number field, description
fields and native type fields. To add a jurisdiction, copy the example for its platform and
change these values. Every dataset in the released corpus is listed in
`permitkey_sources.csv` on the Harvard Dataverse record. Records are then normalized with
`pipeline/master_normalization_copy.py` and classified with `pipeline/classify_partition.py`.
