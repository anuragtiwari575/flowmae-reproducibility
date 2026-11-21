#!/usr/bin/env python3
"""
gbdt_ref_v1.py -- a gradient-boosted-tree reference on the same folds.

WHY
---
The review's point is that a reviewer in tabular intrusion detection
will ask whether FlowMAE beats strong tabular learners, not only the
convolutional--recurrent baseline. Gradient-boosted trees are the right
reference: on tabular data they are hard to beat, and they need no
federation, no adversarial training and no GPU, so this is a cheap
answer to a question that would otherwise stand open.

WHAT IT IS AND IS NOT
---------------------
This is a CENTRALISED, CLEAN-PERFORMANCE reference. It trains on the
union of the clients' data for each fold -- that is, on more
information than any federated arm sees -- and it is not attacked. It
therefore bounds what the federated arms could reach on clean traffic
rather than competing with them, which is exactly the context the
paper needs and is how it must be described: not as a baseline that
FlowMAE beats, but as the ceiling the protocol is being measured
against.

The folds, the preprocessing, the rebalancing and the test rows are
the same ones every other arm uses, because they come from the same
make_folds call.

XGBoost or LightGBM are used when installed; otherwise scikit-learn's
HistGradientBoostingClassifier, which is the same family and is
already a dependency of this repository.

USAGE
  python -u gbdt_ref_v1.py --corpus unsw  --task multiclass10 \
      --out out/gbdt_unsw.json
  python -u gbdt_ref_v1.py --corpus cicids --task multiclass8 \
      --root data/cicids2017 --out out/gbdt_cicids.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as ds                       # noqa: E402
import ciciot_loader_v7 as ciot            # noqa: E402


def make_model(kind, n_cls, seed):
    if kind in ("auto", "xgboost"):
        try:
            from xgboost import XGBClassifier
            return "xgboost", XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                objective="multi:softprob", num_class=n_cls,
                random_state=seed, n_jobs=-1, verbosity=0)
        except Exception:                               # noqa: BLE001
            if kind == "xgboost":
                raise SystemExit("xgboost requested but not importable")
    if kind in ("auto", "lightgbm"):
        try:
            from lightgbm import LGBMClassifier
            return "lightgbm", LGBMClassifier(
                n_estimators=400, num_leaves=63, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                random_state=seed, n_jobs=-1, verbose=-1)
        except Exception:                               # noqa: BLE001
            if kind == "lightgbm":
                raise SystemExit("lightgbm requested but not importable")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return "hist_gbdt", HistGradientBoostingClassifier(
        max_iter=400, max_depth=None, learning_rate=0.1,
        random_state=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="unsw",
                    choices=("unsw", "cicids", "ciciot"))
    ap.add_argument("--task", default="multiclass8")
    ap.add_argument("--root", default="data")
    ap.add_argument("--ciciot-rate", type=float, default=ciot.DEFAULT_RATE)
    ap.add_argument("--ciciot-seed", type=int, default=ciot.DEFAULT_SEED)
    ap.add_argument("--ciciot-cache", default=r"out/ciciot_cache")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smote-cap", type=int, default=20_000)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--model", default="auto",
                    choices=("auto", "xgboost", "lightgbm", "hist"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="ignored; CPU only")
    ap.add_argument("--out", default="out/gbdt_ref.json")
    a = ap.parse_args()

    import phase1_arms_v17 as drv
    args = SimpleNamespace(corpus=a.corpus, task=a.task, root=a.root,
                           ciciot_rate=a.ciciot_rate,
                           ciciot_seed=a.ciciot_seed,
                           ciciot_cache=a.ciciot_cache)
    df, _ = drv.load_corpus(args)
    folds = ds.make_folds(df, n_clients=a.clients, n_folds=a.folds,
                          alpha=a.alpha, smote=True,
                          smote_cap=a.smote_cap, subsample=a.subsample)
    del df

    print("=" * 74)
    print("GRADIENT-BOOSTED-TREE REFERENCE (centralised, clean only)")
    print(f"  corpus {a.corpus}  task {a.task}  {len(folds)} folds")
    print("  trains on the union of the clients' data, so it sees more")
    print("  than any federated arm; it is a ceiling, not a competitor.")
    print("=" * 74)

    per_fold = []
    kind_used = None
    for f in folds:
        Xtr = np.concatenate([f.clients[c][0] for c in sorted(f.clients)], 0)
        ytr = np.concatenate([f.clients[c][1] for c in sorted(f.clients)], 0)
        kind_used, clf = make_model(a.model if a.model != "hist" else "hist",
                                    len(f.classes), a.seed)
        t0 = time.time()
        clf.fit(Xtr, ytr)
        secs = time.time() - t0
        pred = clf.predict(f.X_test)
        mf1 = float(f1_score(f.y_test, pred, average="macro",
                             zero_division=0))
        acc = float(accuracy_score(f.y_test, pred))
        per_fold.append({"fold": int(f.fold), "macro_f1": mf1,
                         "accuracy": acc, "train_rows": int(len(Xtr)),
                         "secs": round(secs, 1)})
        print(f"  fold {f.fold}: macro-F1 {mf1:.4f}  acc {acc:.4f}  "
              f"({secs:.0f}s, {len(Xtr):,} train rows)")

    mf = np.array([d["macro_f1"] for d in per_fold])
    res = {"corpus": a.corpus, "task": a.task, "model": kind_used,
           "seed": a.seed, "per_fold": per_fold,
           "macro_f1_mean": float(mf.mean()),
           "macro_f1_sd": float(mf.std(ddof=1)) if len(mf) > 1 else 0.0}
    print()
    print(f"  {kind_used}: macro-F1 {mf.mean():.4f} +/- "
          f"{mf.std(ddof=1) if len(mf) > 1 else 0:.4f} over "
          f"{len(mf)} folds")
    print("  Report this as a centralised clean ceiling measured under")
    print("  the same protocol, not as a federated baseline.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
