#!/usr/bin/env python
"""Phase 1: assign the 60,000 most frequent native labels to their nearest concept anchor."""
import pyarrow.parquet as pq, pyarrow as pa, numpy as np, pandas as pd, json, time, sys
T0=time.time(); BASE="data/taxonomy"; OUT=f"{BASE}/outputs"
def log(m): sys.stdout.write(f"[{time.time()-T0:7.1f}s] {m}\n"); sys.stdout.flush()

# Concept anchors: name -> descriptive phrase.
ANCHORS = {
 # Trades
 "ELECTRICAL":"electrical wiring permit electrical work",
 "PLUMBING":"plumbing permit plumbing work piping",
 "MECHANICAL_HVAC":"mechanical hvac heating ventilation air conditioning furnace",
 "ROOFING":"roof reroof roofing shingles tearoff",
 "BUILDING_GENERAL":"building permit general construction structure",
 "ALTERATION_REMODEL":"alteration remodel renovation interior tenant improvement addition",
 "NEW_CONSTRUCTION":"new construction new building new dwelling new residence",
 "DEMOLITION":"demolition wrecking teardown raze",
 "GRADING_SITEWORK":"grading excavation sitework earthwork land disturbance",
 "DRIVEWAY_PAVING":"driveway sidewalk paving concrete flatwork curb",
 "FENCE_WALL":"fence wall retaining wall screen",
 "STRUCTURAL":"structural foundation framing shear wall",
 # Discrete improvements
 "POOL_SPA":"swimming pool spa hot tub",
 "SIGN":"sign signage banner billboard",
 "FIRE_PROTECTION":"fire sprinkler fire alarm fire protection suppression",
 "WINDOW_DOOR":"window door replacement fenestration glazing",
 "DECK_PATIO_PORCH":"deck patio porch pergola",
 "GARAGE_ACCESSORY":"garage carport shed accessory structure adu",
 "WATER_HEATER":"water heater hot water tank",
 "SEWER_UTILITY":"sewer water utility connection main service lateral",
 "GAS":"gas line gas piping fuel gas",
 # Electrification and climate
 "SOLAR_PV":"solar photovoltaic pv panels rooftop solar",
 "BATTERY_STORAGE":"battery energy storage powerwall",
 "EV_CHARGER":"electric vehicle charger ev charging station",
 "HEAT_PUMP":"heat pump air source heat pump mini split",
 "GENERATOR":"backup generator standby generator",
 "ELECTRICAL_PANEL_UPGRADE":"electrical panel upgrade service upgrade amperage",
 "INSULATION_WEATHERIZATION":"insulation weatherization energy efficiency envelope",
 # Administrative and legacy labels (excluded)
 "PROC_OCCUPANCY":"certificate of occupancy temporary occupancy final inspection",
 "PROC_ENFORCEMENT":"code violation complaint enforcement order tall grass trash nuisance",
 "PROC_LICENSE_ADMIN":"contractor license registration business administrative record revision",
 "PROC_ZONING_PLANNING":"zoning planning land use variance subdivision plat",
 "PROC_RIGHT_OF_WAY":"right of way encroachment street use public way",
 "PROC_LEGACY_UNKNOWN":"legacy permit historical permit unknown miscellaneous no plan",
}
PROCESS = {k for k in ANCHORS if k.startswith("PROC_")}
THRESH = 0.45  # Below threshold -> OTHER

log("importing torch/sentence_transformers")
import torch
from sentence_transformers import SentenceTransformer
log("loading model")
m=SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
names=list(ANCHORS); A=m.encode([ANCHORS[n] for n in names], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
log(f"anchors embedded {A.shape}")

voc=pq.read_table(f"{OUT}/p2_label_vocab_top60k.parquet", columns=["label_text","n_rows","n_juris","n_states"]).to_pandas()
emb=pq.read_table(f"{OUT}/p2_label_emb.parquet").to_pandas()
df=voc.merge(emb, on="label_text", how="inner")
X=np.array(df["emb"].tolist(), dtype=np.float32); X/=(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
log(f"labels {len(df):,}; scoring vs {len(names)} anchors")

S=X @ A.T  # Cosine similarity (unit vectors)
best=S.argmax(1); bestsim=S.max(1)
concept=np.where(bestsim>=THRESH, np.array(names)[best], "OTHER")
df["concept"]=concept; df["concept_sim"]=bestsim
df["is_process"]=df["concept"].isin(PROCESS)

# Per-concept summary
rows=[]
for c,g in df.groupby("concept"):
    rows.append(dict(concept=c, kind=("process" if c in PROCESS else ("other" if c=="OTHER" else "work")),
        depth_rows=int(g.n_rows.sum()), max_juris=int(g.n_juris.max()), max_states=int(g.n_states.max()),
        n_labels=int(len(g)), mean_sim=round(float(g.concept_sim.mean()),3),
        top_labels=" | ".join(f"{r.label_text}({r.n_rows})" for r in g.sort_values('n_rows',ascending=False).head(6).itertuples())))
summ=pd.DataFrame(rows).sort_values("depth_rows", ascending=False)
summ.to_csv(f"{OUT}/p1_concept_summary.csv", index=False)
df[["label_text","n_rows","n_juris","concept","concept_sim","is_process"]].to_parquet(f"{OUT}/p1_label_concept_map.parquet")

tot=df.n_rows.sum()
work=summ[summ.kind=="work"].depth_rows.sum(); proc=summ[summ.kind=="process"].depth_rows.sum(); oth=summ[summ.kind=="other"].depth_rows.sum()
log(f"coverage(of top60k rows): work={work/tot:.3f} process={proc/tot:.3f} other={oth/tot:.3f}")
log(f"DONE -> p1_concept_summary.csv ({len(summ)} concepts), p1_label_concept_map.parquet")
print(summ[["concept","kind","depth_rows","max_juris","n_labels","mean_sim"]].to_string(index=False))
