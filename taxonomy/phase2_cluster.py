#!/usr/bin/env python
"""Phase 2: cluster native-label embeddings into candidate concepts (k = 25, 60 and 150)."""
import pyarrow.parquet as pq, numpy as np, json, time, os
from sklearn.cluster import MiniBatchKMeans
T0=time.time(); BASE="data/taxonomy"; OUT=f"{BASE}/outputs"
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

voc=pq.read_table(f"{OUT}/p2_label_vocab_top60k.parquet").to_pandas()
emb=pq.read_table(f"{OUT}/p2_label_emb.parquet").to_pandas()
df=voc.merge(emb, on="label_text", how="inner")
log(f"labels with emb: {len(df):,}")
X=np.array(df["emb"].tolist(), dtype=np.float32)
X/= (np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
w=df["n_rows"].to_numpy()

for K in [25, 60, 150]:
    km=MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=4096, n_init=5, max_iter=200)
    lab=km.fit_predict(X, sample_weight=w)
    df[f"k{K}"]=lab
    rows=[]
    for c in range(K):
        m=df[df[f"k{K}"]==c]
        depth=int(m["n_rows"].sum())
        breadth=int(m["n_juris"].max())  # Max jurisdictions on any member label
        breadth_sum=int(m["n_juris"].sum())
        states=int(m["n_states"].max())
        top=m.sort_values("n_rows",ascending=False).head(8)
        toplabels=" | ".join(f"{r.label_text}({r.n_rows})" for r in top.itertuples())
        rows.append((c, depth, breadth, states, len(m), toplabels))
    rows.sort(key=lambda r:-r[1])
    with open(f"{OUT}/p2_concepts_k{K}.csv","w") as f:
        f.write("cluster,depth_rows,max_juris,max_states,n_labels,top_labels\n")
        for c,depth,breadth,states,nl,tl in rows:
            f.write(f'{c},{depth},{breadth},{states},{nl},"{tl.replace(chr(34),chr(39))}"\n')
    log(f"K={K}: wrote p2_concepts_k{K}.csv (top depth={rows[0][1]:,})")

df_out=df.drop(columns=["emb","examples"])
df_out.to_parquet(f"{OUT}/p2_label_concept_assignments.parquet")
log("DONE -> concept cards + assignments")
