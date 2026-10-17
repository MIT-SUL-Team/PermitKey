"""Rebuild tract_year.parquet/csv from permitkey_tract_year_sources.parquet (Dataverse)."""
import argparse,json
from pathlib import Path
import duckdb
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
o=Path(a.output);o.mkdir(parents=True,exist_ok=False)
c=duckdb.connect();c.execute('SET threads=2');c.read_parquet(a.input).create_view('eligible')
cols=[x[0] for x in c.execute('DESCRIBE eligible').fetchall()];cats=[x for x in cols if x.startswith('cnt_')];assert len(cats)==14
assert c.execute("SELECT count(*) FROM eligible WHERE year IS NULL OR year_status NOT IN ('source_date','source_date_normalized') OR NOT regexp_full_match(geoid,'[0-9]{11}')").fetchone()[0]==0
agg=','.join('sum('+x+')::BIGINT '+x for x in ['n_records','n_uncategorized']+cats)
c.execute(f"CREATE TABLE final AS SELECT geoid,year,{agg},CASE WHEN count(*) FILTER(WHERE authority_id IS NULL)=0 THEN count(DISTINCT authority_id) ELSE NULL END::BIGINT n_jurisdictions,count(DISTINCT dataset_key)::BIGINT n_datasets,'permitkey-v1.0' release_version FROM eligible GROUP BY 1,2 ORDER BY 1,2")
assert c.execute('SELECT count(*)=count(DISTINCT(geoid,year)) FROM final').fetchone()[0]
for ext in ['parquet','csv']:c.execute('COPY final TO ? (FORMAT '+ext+(',COMPRESSION ZSTD' if ext=='parquet' else ',HEADER TRUE,FORCE_QUOTE *')+')',[str(o/('tract_year.'+ext))])
print(json.dumps(dict(zip(['cells','records','uncategorized','classified'],c.execute('SELECT count(*),sum(n_records),sum(n_uncategorized),sum(n_records-n_uncategorized) FROM final').fetchone()))))
