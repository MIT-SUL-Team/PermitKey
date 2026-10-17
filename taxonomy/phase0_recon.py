#!/usr/bin/env python
"""Phase 0: field coverage and native permit-type vocabularies of the source corpus."""
import duckdb, json, time, os, sys

T0=time.time()
BASE="data/taxonomy"
OUT=f"{BASE}/outputs"; os.makedirs(OUT, exist_ok=True)
PKG="data/corpus_june2026"
BUILTY=f"{PKG}/builty_research_shape_desc_nonempty_geocoding_intact_20260605.parquet"
SLIM=f"{PKG}/final_categorization_hybrid_enriched_slim_date_zstd.parquet"

con=duckdb.connect()
con.execute("PRAGMA threads=16")
con.execute("PRAGMA memory_limit='400GB'")
def q(sql): return con.execute(sql).fetchall()
def one(sql): return con.execute(sql).fetchone()[0]
summary={}
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

# 1. Basic counts
summary["builty_rows"]=one(f"SELECT count(*) FROM '{BUILTY}'")
summary["slim_rows"]=one(f"SELECT count(*) FROM '{SLIM}'")
log(f"rows builty={summary['builty_rows']:,} slim={summary['slim_rows']:,}")

# 2. Join-key validation
summary["builty_distinct_permit_number"]=one(f"SELECT count(DISTINCT PERMIT_NUMBER) FROM '{BUILTY}'")
summary["builty_distinct_jur_permit"]=one(f"SELECT count(DISTINCT (JURISDICTION || '|' || PERMIT_NUMBER)) FROM '{BUILTY}'")
summary["builty_null_permit_number"]=one(f"SELECT count(*) FROM '{BUILTY}' WHERE PERMIT_NUMBER IS NULL OR PERMIT_NUMBER=''")
summary["slim_distinct_permit_number"]=one(f"SELECT count(DISTINCT permit_number) FROM '{SLIM}'")
summary["slim_distinct_jur_permit"]=one(f"SELECT count(DISTINCT (jurisdiction || '|' || permit_number)) FROM '{SLIM}'")
log("join-key stats done")
summary["permit_number_unique_in_builty"]= summary["builty_distinct_permit_number"]==summary["builty_rows"]
summary["jur_permit_unique_in_builty"]= summary["builty_distinct_jur_permit"]==summary["builty_rows"]

# 3. Description coverage and length
r=one(f"SELECT count(*) FROM '{BUILTY}' WHERE DESCRIPTION IS NOT NULL AND length(trim(DESCRIPTION))>0")
summary["builty_nonempty_desc"]=r
summary["builty_desc_avg_len"]=round(one(f"SELECT avg(length(DESCRIPTION)) FROM '{BUILTY}' WHERE DESCRIPTION IS NOT NULL"),1)
log("description coverage done")

# 4. Native category field coverage
for col in ["RECORD_TYPE_ORIGINAL","RECORD_SUBTYPE_ORIGINAL","STATUS_NORMALIZED"]:
    summary[f"nonnull_{col}"]=one(f"SELECT count(*) FROM '{BUILTY}' WHERE {col} IS NOT NULL AND {col}<>''")
    summary[f"distinct_{col}"]=one(f"SELECT count(DISTINCT {col}) FROM '{BUILTY}'")
log("native-field coverage done")

# 5. Top jurisdictions by volume
q5=q(f"""SELECT JURISDICTION, STATE, count(*) n,
          count(DISTINCT RECORD_TYPE_ORIGINAL) n_rectype,
          count(DISTINCT RECORD_SUBTYPE_ORIGINAL) n_recsub
        FROM '{BUILTY}' GROUP BY 1,2 ORDER BY n DESC LIMIT 60""")
with open(f"{OUT}/p0_top_jurisdictions.csv","w") as f:
    f.write("jurisdiction,state,n,n_rectype,n_recsubtype\n")
    for j,s,n,a,b in q5: f.write(f'"{j}","{s}",{n},{a},{b}\n')
summary["n_jurisdictions"]=one(f"SELECT count(DISTINCT JURISDICTION) FROM '{BUILTY}'")
log(f"jurisdictions={summary['n_jurisdictions']:,}; top dumped")

# 6. Native vocabulary: (jurisdiction, type, subtype) counts
con.execute(f"""CREATE TEMP TABLE vocab AS
  SELECT JURISDICTION jurisdiction, STATE state,
         coalesce(RECORD_TYPE_ORIGINAL,'') rectype,
         coalesce(RECORD_SUBTYPE_ORIGINAL,'') recsub,
         count(*) n
  FROM '{BUILTY}' GROUP BY 1,2,3,4""")
summary["distinct_native_combos"]=one("SELECT count(*) FROM vocab")
con.execute(f"COPY (SELECT * FROM vocab ORDER BY n DESC) TO '{OUT}/p0_native_vocab_full.csv' (HEADER, DELIMITER ',')")
log(f"native vocab combos={summary['distinct_native_combos']:,} dumped")

# 7. Concept-flag prevalence
bools=["ACCESSORY_STRUCTURE","AIR_SOURCE_HEAT_PUMP","BATH_REMODEL","COMMERCIAL","DEMOLITION","EFFICIENCY",
 "ELECTRICAL_PANEL_UPGRADE","ELECTRICAL_SERVICE_UPGRADE","EV_CHARGER","EXTERIOR_REMODEL","FOUNDATION",
 "GROUND_SOURCE_HEAT_PUMP","HEAT_PUMP_WATER_HEATER","INDUCTION_STOVE","KITCHEN_REMODEL","MULTIFAMILY",
 "NEW_ADU","NEW_DWELLING","POOL_SPA","PUBLIC_WORK","REROOF","RESIDENTIAL","ROOM_ADDITION","SITEWORK",
 "SOLAR","SOLAR_BATTERY_STORAGE","TENANT_IMPROVEMENT","UTILITIES","WIND","WINDOW_DOOR"]
sel=",".join([f"sum(CASE WHEN {b} THEN 1 ELSE 0 END) {b}" for b in bools])
row=one_row=con.execute(f"SELECT {sel} FROM '{BUILTY}'").fetchone()
boolprev=dict(zip(bools,[int(x) for x in row]))
with open(f"{OUT}/p0_bool_concept_prevalence.csv","w") as f:
    f.write("concept_flag,positive_rows,prevalence\n")
    for b in bools: f.write(f"{b},{boolprev[b]},{boolprev[b]/summary['builty_rows']:.6f}\n")
summary["bool_concepts"]=boolprev
log("bool concept prevalence done")

summary["elapsed_s"]=round(time.time()-T0,1)
json.dump(summary, open(f"{OUT}/p0_summary.json","w"), indent=1)
log("DONE -> p0_summary.json")
print(json.dumps({k:v for k,v in summary.items() if k!="bool_concepts"}, indent=1))
