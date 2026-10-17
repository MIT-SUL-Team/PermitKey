#!/usr/bin/env python
"""Embed text in shards with a sentence encoder."""
import argparse, os, time, sys
T0=time.time()
def log(m): sys.stdout.write(f"[{time.time()-T0:7.1f}s] {m}\n"); sys.stdout.flush()
log("process start")
ap=argparse.ArgumentParser()
ap.add_argument("--in_parquet", required=True)
ap.add_argument("--out_parquet", required=True)
ap.add_argument("--id_col", default="label_text")
ap.add_argument("--text_col", default="label_text")
ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--max_chars", type=int, default=512)
ap.add_argument("--normalize", type=int, default=1)
a=ap.parse_args()
log("importing numpy/pyarrow"); import numpy as np, pyarrow.parquet as pq, pyarrow as pa
log("importing torch"); import torch
log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
log("importing sentence_transformers"); from sentence_transformers import SentenceTransformer
log("reads parquet")
tbl=pq.read_table(a.in_parquet, columns=list({a.id_col, a.text_col}))
ids=tbl.column(a.id_col).to_pylist(); txt=tbl.column(a.text_col).to_pylist()
idx=[i for i in range(len(ids)) if i % a.nshard == a.shard]
ids=[ids[i] for i in idx]; txt=[(txt[i] or "")[:a.max_chars] for i in idx]
log(f"rows this shard={len(ids):,}; loading model {a.model}")
m=SentenceTransformer(a.model, device="cuda"); m.max_seq_length=min(256, m.max_seq_length)
log(f"model loaded on {torch.cuda.get_device_name(0)}; encoding")
embs=m.encode(txt, batch_size=a.batch, normalize_embeddings=bool(a.normalize),
              show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
log(f"encoded {embs.shape}; writing")
dim=int(embs.shape[1])
emb_arr=pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(embs).reshape(-1)), dim)
out=pa.table({a.id_col: pa.array(ids), "emb": emb_arr})
pq.write_table(out, a.out_parquet)
log(f"DONE -> {a.out_parquet}")
