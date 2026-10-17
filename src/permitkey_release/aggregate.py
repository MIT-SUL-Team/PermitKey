"""Aggregate classified records to tract-year (or jurisdiction-year) counts."""
from pathlib import Path
import duckdb
from .common import HEADS, SELECTED, MASK, fresh, literal, write_json

REQUIRED = {'stable_row_key','keep','issue_year','date_basis','geoid','tract_verified',
            'jurisdiction_id','dataset_id','label_bits','evidence_tier'}

def aggregate(source, destination, version, level='tract'):
    source = Path(source).resolve()
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    c = duckdb.connect()
    c.execute("SET threads=2")
    reader = 'read_parquet' if source.suffix.lower()=='.parquet' else 'read_json_auto'
    c.execute(f'CREATE VIEW input AS SELECT * FROM {reader}({literal(source)})')
    columns = {r[0] for r in c.execute('DESCRIBE input').fetchall()}
    if missing := REQUIRED-columns:
        raise ValueError(f'Missing canonical fields: {sorted(missing)}')
    total, keys = c.execute('SELECT count(*),count(DISTINCT stable_row_key) FROM input').fetchone()
    if total != keys:
        raise ValueError('Null or repeated stable_row_key; resolve input identity before aggregation')
    if c.execute(f'''SELECT count(*) FROM input WHERE label_bits IS NULL OR
        label_bits<0 OR label_bits>={(1<<len(HEADS))} OR label_bits<>floor(label_bits)
        OR keep IS NULL OR evidence_tier IS NULL
        OR evidence_tier NOT IN ('source_evidence','provisional')''').fetchone()[0]:
        raise ValueError('Invalid bits, keep flag, or evidence tier')
    c.execute("CREATE VIEW retained AS SELECT * FROM input WHERE keep AND evidence_tier='source_evidence'")
    c.execute("""CREATE VIEW dated AS SELECT * FROM retained
        WHERE issue_year BETWEEN 1800 AND 2026 AND issue_year=floor(issue_year)
        AND date_basis='issue'""")
    if level=='tract':
        condition="tract_verified IS TRUE AND regexp_full_match(CAST(geoid AS VARCHAR),'[0-9]{11}')"
        group='CAST(geoid AS VARCHAR) AS geoid'
        key='geoid'
    elif level=='jurisdiction':
        condition="jurisdiction_id IS NOT NULL AND trim(jurisdiction_id)<>''"
        group='jurisdiction_id'
        key='jurisdiction_id'
    else:
        raise ValueError(level)
    c.execute('CREATE VIEW eligible AS SELECT * FROM dated WHERE '+condition)
    if c.execute("""SELECT count(*) FROM eligible WHERE dataset_id IS NULL OR trim(dataset_id)=''
         OR jurisdiction_id IS NULL OR trim(jurisdiction_id)=''""").fetchone()[0]:
        raise ValueError('Eligible rows need canonical source and issuing-authority IDs')
    expr=[f"sum(CASE WHEN (CAST(label_bits AS UBIGINT) & {1<<HEADS.index(k)})<>0 THEN 1 ELSE 0 END)::BIGINT AS cnt_{k}" for k in SELECTED]
    q=f'''SELECT {group}, CAST(issue_year AS INTEGER) AS year,
        count(*)::BIGINT AS n_records,
        sum(CASE WHEN (CAST(label_bits AS UBIGINT)&{MASK})=0 THEN 1 ELSE 0 END)::BIGINT AS n_uncategorized,
        {','.join(expr)},count(DISTINCT jurisdiction_id)::INTEGER AS n_jurisdictions,
        count(DISTINCT dataset_id)::INTEGER AS n_datasets,
        {literal(version)} AS release_version
        FROM eligible GROUP BY 1,2 ORDER BY 1,2'''
    c.execute('CREATE TEMP TABLE result AS '+q)
    included=c.execute('SELECT count(*) FROM eligible').fetchone()[0]
    summed=c.execute('SELECT coalesce(sum(n_records),0) FROM result').fetchone()[0]
    assert included==summed
    counts={name:c.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in ['input','retained','dated','eligible','result']}
    kept=c.execute('SELECT count(*) FROM input WHERE keep').fetchone()[0]
    counts.update(excluded_by_keep=total-kept,provisional_excluded=kept-counts['retained'],
                  missing_or_invalid_issue_year=counts['retained']-counts['dated'],
                  unresolved_target_geography=counts['dated']-included,
                  positive14=c.execute(f'SELECT count(*) FROM eligible WHERE (CAST(label_bits AS UBIGINT)&{MASK})<>0').fetchone()[0])
    target.mkdir(parents=True)
    for suffix,options in [('parquet','FORMAT PARQUET, COMPRESSION ZSTD'),('csv','FORMAT CSV, HEADER TRUE')]:
        c.execute(f'COPY result TO {literal(target/(level+"_year."+suffix))} ({options})')
    # Records contributed by each source to each cell.
    c.execute(f'''COPY (SELECT {group},CAST(issue_year AS INTEGER) AS year,jurisdiction_id AS source_jurisdiction_id,
        dataset_id,count(*)::BIGINT n_records FROM eligible GROUP BY 1,2,3,4)
        TO {literal(target/'source_contributions.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)''')
    write_json(target/'manifest.json',{'release_version':version,'level':level,'counts':counts,
        'counting_unit':'Retained canonical source-record occurrences; keep is supplied by the upstream counting policy.',
        'year_policy':'Explicit issue years1800-2026 only','zero_policy':'Absent cell is not an observed zero.',
        'geography_policy':'Input tract_verified must be established upstream; identifier syntax is checked here.',
        'validation':'Count reconciliation, record identity, source identifiers and label bits checked.'})
    c.close()
    return counts
