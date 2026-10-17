#!/usr/bin/env python3
"""City-held-out evaluation of all 25 heads, optionally limited to the largest N cities."""
import argparse, json
from pathlib import Path
import duckdb
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

HERE = Path(__file__).resolve().parent
DEFAULT_CFG = HERE.parent / "config" / "training_concepts.json"


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(F, 3), support=tp + fn)


def fit_fold(X, y, flag, tr, te):
    """One (concept, city) held-out fit. Returns counts per method or None if gated out."""
    if int(y[tr].sum()) < 20 or int(y[te].sum()) < 5:
        return None
    clf = SGDClassifier(loss="log_loss", class_weight="balanced", alpha=1e-6,
                        max_iter=12, tol=1e-3, random_state=0)
    clf.fit(X[tr], y[tr])
    pm = clf.predict(X[te]); yf = flag[te]; yt = y[te]
    pe = ((pm == 1) | (yf == 1)).astype(int)
    r = {}
    for name, pred in [("flag", yf), ("model", pm), ("ensemble", pe)]:
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        r[name] = (tp, fp, fn)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only-frozen", action="store_true")
    ap.add_argument("--holdout-cities", type=int, default=0, help="0=all (original); N=top-N by volume")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    frozen = [c["tech"] for c in cfg["frozen_8"]]
    tech = frozen if args.only_frozen else frozen + [c["tech"] for c in cfg["new_17"]]

    con = duckdb.connect(); con.execute("PRAGMA threads=8")
    df = con.execute(f"SELECT * FROM '{args.train}'").df()
    print("rows", len(df), flush=True)

    if args.holdout_cities and args.holdout_cities > 0:
        top = df["city"].value_counts().head(args.holdout_cities).index.tolist()
        test_cities = sorted(top)
        print(f"held-out folds = top {len(test_cities)} cities:", test_cities, flush=True)
    else:
        test_cities = sorted(df["city"].unique().tolist())

    vec = HashingVectorizer(n_features=2**18, ngram_range=(1, 2), alternate_sign=False, norm="l2")
    X = vec.transform(df["descr"].fillna("").values)
    city = df["city"].values

    # Independent (concept, city) fits, run in parallel.
    tasks = []
    for t in tech:
        y = df[f"label_{t}"].values.astype(int)
        flag = df[f"flag_{t}"].values.astype(int)
        for c in test_cities:
            te = (city == c)
            tr = ~te
            tasks.append((t, c, y, flag, tr, te))
    print(f"{len(tasks)} fits over {len(tech)} concepts x {len(test_cities)} cities, n_jobs={args.n_jobs}", flush=True)

    results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(lambda t, c, y, flag, tr, te: (t, fit_fold(X, y, flag, tr, te)))(*task) for task in tasks
    )

    out = {t: {k: [0, 0, 0] for k in ["flag", "model", "ensemble"]} for t in tech}
    raw = {k: [0, 0, 0] for k in ["flag", "model", "ensemble"]}
    for t, r in results:
        if r is None:
            continue
        for name, (tp, fp, fn) in r.items():
            out[t][name][0] += tp; out[t][name][1] += fp; out[t][name][2] += fn
            raw[name][0] += tp; raw[name][1] += fp; raw[name][2] += fn
    out = {t: {k: prf(*v) for k, v in d.items()} for t, d in out.items()}
    for t in tech:
        print(t, "flag_f1", out[t]["flag"]["f1"], "-> ens_f1", out[t]["ensemble"]["f1"], flush=True)
    out["MICRO_ALL"] = {k: prf(*v) for k, v in raw.items()}
    args.out.write_text(json.dumps(out, indent=1))
    print("RB"); print(json.dumps(out, indent=1)); print("RE")


if __name__ == "__main__":
    main()
