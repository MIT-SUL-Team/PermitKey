#!/usr/bin/env python3
"""Build the weak-label training table for the added heads from native permit type and subtype fields."""
import argparse, json
from pathlib import Path
import duckdb

B = "data/corpus_june2026/builty_research_shape_desc_nonempty_geocoding_intact_20260605.parquet"
HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "config" / "training_concepts.json"

# Terms matched against the native category field.
SEEDS = {
 "solar":["solar pv","photovoltaic","pv system","solar panel","solar electric","\\bsolar\\b"],
 "ev_charger":["ev charger","evse","electric vehicle charg","charging station"],
 "battery":["battery storage","energy storage system","powerwall","\\bess\\b"],
 "heat_pump":["heat pump","mini split","mini-split"],
 "panel":["panel upgrade","service upgrade","service change","amp service","electrical service"],
 "reroof":["reroof","re-roof","roof replacement","roofing","\\broof\\b","shingle"],
 "pool_spa":["\\bpool\\b","\\bspa\\b","hot tub"],
 "demolition":["demolition","demolish","wrecking","tear down"],
 "water_heater":["water heater","tankless","hot water"],
 "fire_protection":["fire alarm","fire sprinkler","fire suppression","smoke detection","standpipe","fire protection"],
 "window_door":["window","\\bdoor\\b","glazing","storefront","fenestration"],
 "fence_wall":["\\bfence\\b","retaining wall"],
 "deck_patio_porch":["\\bdeck\\b","patio","porch"],
 "driveway_paving":["driveway","paving","pavement","sidewalk","drive approach"],
 "grading_sitework":["grading","sitework","site work","earthwork","excavation","erosion"],
 "garage_accessory":["garage","carport","\\bshed\\b","accessory dwelling","\\badu\\b","accessory structure"],
 "sewer_utility":["sewer","water service","utility connection","water main","sewer lateral"],
 "gas":["fuel gas","gas piping","gas meter","natural gas","gas line"],
 "sign":["\\bsign\\b","signage","monument sign","wall sign","awning"],
 "structural":["structural","foundation","seismic","load bearing","underpinning"],
 "new_construction":["new construction","new building","new residence","new dwelling","new single family","new sfr"],
 "alteration_remodel":["alteration","remodel","renovation","tenant improvement","interior remodel","addition"],
 "electrical":["\\belectrical\\b","\\belectric\\b"],
 "plumbing":["\\bplumbing\\b"],
 "mechanical_hvac":["mechanical","\\bhvac\\b","heating ventilation","air conditioning"],
}

# Source flag column for the concept; None means no flag.
FLAGCOL = {
 "solar":"SOLAR","ev_charger":"EV_CHARGER","battery":"SOLAR_BATTERY_STORAGE",
 "heat_pump":"(AIR_SOURCE_HEAT_PUMP OR GROUND_SOURCE_HEAT_PUMP)",
 "panel":"(ELECTRICAL_PANEL_UPGRADE OR ELECTRICAL_SERVICE_UPGRADE)",
 "reroof":"REROOF","pool_spa":"POOL_SPA","demolition":"DEMOLITION",
 "water_heater":"HEAT_PUMP_WATER_HEATER","window_door":"WINDOW_DOOR",
 "new_construction":"(NEW_DWELLING OR NEW_ADU)","garage_accessory":"ACCESSORY_STRUCTURE",
 "alteration_remodel":"(EXTERIOR_REMODEL OR KITCHEN_REMODEL OR BATH_REMODEL OR ROOM_ADDITION OR TENANT_IMPROVEMENT)",
 "structural":"FOUNDATION","grading_sitework":"SITEWORK","sewer_utility":"UTILITIES",
}


def rx(seeds):
    return "(" + "|".join(s.replace(" ", r"\s+") for s in seeds) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=3_000_000)
    ap.add_argument("--neg-cap", type=int, default=400_000)
    args = ap.parse_args()
    cfg = json.loads(CFG.read_text())
    techs = [c["tech"] for c in cfg["frozen_8"]] + [c["tech"] for c in cfg["new_17"]]

    con = duckdb.connect(); con.execute("PRAGMA threads=8"); con.execute("PRAGMA memory_limit='40GB'")
    native = "lower(coalesce(RECORD_TYPE_ORIGINAL,'')||' | '||coalesce(RECORD_SUBTYPE_ORIGINAL,''))"
    label_exprs, flag_exprs = [], []
    for t in techs:
        label_exprs.append(f"CASE WHEN regexp_matches({native}, '{rx(SEEDS[t])}') THEN 1 ELSE 0 END AS label_{t}")
        col = FLAGCOL.get(t)
        flag_exprs.append((f"CASE WHEN {col} THEN 1 ELSE 0 END" if col else "0") + f" AS flag_{t}")
    psum = "+".join([f"label_{t}+flag_{t}" for t in techs])

    con.execute(f"""
      CREATE OR REPLACE TABLE base AS
      SELECT lower(coalesce(DESCRIPTION,'')) AS descr,
             upper(trim(coalesce(JURISDICTION, CITY, 'UNK'))) AS city,
             {', '.join(label_exprs)}, {', '.join(flag_exprs)}
      FROM '{B}' USING SAMPLE {args.sample} ROWS (reservoir, 20260903)
    """)
    con.execute(f"""
      COPY (
        SELECT * FROM base WHERE ({psum})>0
        UNION ALL BY NAME
        SELECT * FROM base WHERE ({psum})=0 USING SAMPLE {args.neg_cap} ROWS (reservoir, 7)
      ) TO '{args.out}' (FORMAT parquet)
    """)
    n = con.execute(f"SELECT count(*) FROM '{args.out}'").fetchone()[0]
    print("wrote", args.out, "rows", n)
    for t in techs:
        p = con.execute(f"SELECT sum(label_{t}), sum(flag_{t}) FROM '{args.out}'").fetchone()
        print(f"  {t:<18} native_label_pos={p[0]:>8}  builty_flag_pos={p[1]:>8}")


if __name__ == "__main__":
    main()
