#!/usr/bin/env python
"""Phase 3: concept-by-jurisdiction support and a balanced, jurisdiction-tagged sample per concept."""
import duckdb, time, os, sys
T0=time.time(); BASE="data/taxonomy"; OUT=f"{BASE}/outputs"
PKG="data/corpus_june2026"
BUILTY=f"{PKG}/builty_research_shape_desc_nonempty_geocoding_intact_20260605.parquet"
def log(m): sys.stdout.write(f"[{time.time()-T0:7.1f}s] {m}\n"); sys.stdout.flush()
con=duckdb.connect(); con.execute("PRAGMA threads=16"); con.execute("PRAGMA memory_limit='400GB'")

NORM = r"""trim(regexp_replace(regexp_replace(lower(
   coalesce(RECORD_TYPE_ORIGINAL,'') || ' ' || coalesce(RECORD_SUBTYPE_ORIGINAL,'')
), '[()|/:,_\-]+', ' ', 'g'), '\s+', ' ', 'g'))"""
CAP=150000  # Per-concept cap for the sample

con.execute(f"CREATE TEMP VIEW map AS SELECT label_text, concept, is_process FROM '{OUT}/p1_label_concept_map.parquet'")
log("building row->concept ...")
con.execute(f"""CREATE TEMP TABLE rc AS
  SELECT {NORM} AS label_text, JURISDICTION AS jur, STATE AS state, DESCRIPTION AS descr
  FROM '{BUILTY}'""")
con.execute("""CREATE TEMP TABLE rcc AS
  SELECT r.jur, r.state, r.descr, coalesce(m.concept,'OTHER') AS concept, coalesce(m.is_process,false) AS is_process
  FROM rc r LEFT JOIN map m USING(label_text)""")
log("row->concept joined")

# (a) Concept x jurisdiction support
con.execute(f"""COPY (
  SELECT concept, count(*) AS n_rows, count(DISTINCT jur) AS n_juris, count(DISTINCT state) AS n_states,
         max(is_process) AS is_process
  FROM rcc GROUP BY concept ORDER BY n_rows DESC
) TO '{OUT}/p3_concept_consensus.csv' (HEADER, DELIMITER ',')""")
log("concept consensus matrix written")

# Top jurisdictions per work concept
con.execute(f"""COPY (
  SELECT concept, jur, state, count(*) n FROM rcc
  WHERE NOT is_process AND concept<>'OTHER'
  GROUP BY 1,2,3 QUALIFY row_number() OVER (PARTITION BY concept ORDER BY count(*) DESC) <= 25
  ORDER BY concept, n DESC
) TO '{OUT}/p3_concept_top_jurisdictions.csv' (HEADER, DELIMITER ',')""")
log("per-concept top jurisdictions written")

# (b) Balanced, jurisdiction-tagged sample
con.execute(rf"""CREATE TEMP TABLE clean AS
  SELECT concept, jur, state, substr(regexp_replace(descr, '\s+', ' ', 'g'),1,400) AS descr
  FROM rcc
  WHERE NOT is_process AND concept<>'OTHER'
    AND descr IS NOT NULL AND length(trim(descr)) BETWEEN 8 AND 4000
    AND position('base64' IN descr)=0 AND position('data:application' IN descr)=0""")
tmp=f"{OUT}/p3_machinegold_sample.parquet.tmp"
con.execute(f"""COPY (
  SELECT concept, jur, state, descr FROM clean
  QUALIFY row_number() OVER (PARTITION BY concept ORDER BY hash(descr || jur)) <= {CAP}
) TO '{tmp}' (FORMAT parquet)""")
n=con.execute(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
os.replace(tmp, f"{OUT}/p3_machinegold_sample.parquet")  # Atomic rename
log(f"machine-gold sample rows = {n:,}")
log("DONE -> p3_concept_consensus.csv, p3_concept_top_jurisdictions.csv, p3_machinegold_sample.parquet")
