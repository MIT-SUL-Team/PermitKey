#!/bin/bash
# Embed the 60,000 most frequent native labels (run from the repository root).
export HF_HOME=data/hf_cache PYTHONUNBUFFERED=1
python taxonomy/embed_worker.py \
  --in_parquet data/taxonomy/outputs/p2_label_vocab_top60k.parquet \
  --out_parquet data/taxonomy/outputs/p2_label_emb.parquet \
  --id_col label_text --text_col label_text --batch 2048 \
  --model BAAI/bge-small-en-v1.5
