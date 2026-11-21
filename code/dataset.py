"""
UNSW-NB15 loading and federated partitioning for the supplementary study.

Corpus
------
The curated ten-class partition released by the dataset authors
(UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv, 257,673 records).
Every attack class is carried over whole from the full capture; only Normal
and Generic were subsampled by the dataset authors.  So the eight attack
classes here hold exactly the counts the manuscript's Table 3 reports, while
Normal and Generic are smaller.

This is declared in the supplementary section.  The two files are pooled and
re-split by this module rather than used as the authors' split, because the
protocol under test needs five rotating test partitions, not one fixed one.

Ordering, which is the part that matters
----------------------------------------
    stratified 70/10/20 split
            |                      <- validation and test frozen here
    Dirichlet(alpha) over the TRAINING split only
            |
    per-client SMOTE, on that client's own shard
            |
    min-max normalisation, fitted on training rows only

SMOTE and the normalisation statistics never see validation or test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

log = logging.getLogger(__name__)

# Manuscript Table 3.  Seven of these match the curated partition exactly;
# Normal and Generic are the subsampled ones, and DoS is a transposition in
# the manuscript (16,535 printed for a true 16,353).
TABLE3 = {
    "Normal": 2_218_761, "Generic": 215_481, "Exploits": 44_525,
    "Fuzzers": 24_246, "DoS": 16_535, "Reconnaissance": 13_987,
    "Analysis": 2_677, "Backdoor": 2_329, "Shellcode": 1_511, "Worms": 174,
}

DROP = ["id", "label"]           # 'label' is the binary flag; attack_cat is the target
CATEGORICAL = ["proto", "service", "state"]


def load_unsw(root: str | Path, verbose: bool = True) -> pd.DataFrame:
    root = Path(root)
    parts = []
    for fn in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
        p = root / fn
        if not p.exists():
            raise FileNotFoundError(f"{fn} not found in {root}")
        parts.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(parts, ignore_index=True)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]

    df["target"] = df["attack_cat"].astype(str).str.strip()
    df = df.drop(columns=[c for c in DROP + ["attack_cat"] if c in df.columns])

    # One-hot the three categorical fields; everything else is already numeric.
    present_cat = [c for c in CATEGORICAL if c in df.columns]
    if present_cat:
        # cap cardinality: rare levels folded into 'other' so the one-hot stays small
        for c in present_cat:
            vc = df[c].value_counts()
            keep = set(vc[vc >= 500].index)
            df[c] = np.where(df[c].isin(keep), df[c], "other")
        df = pd.get_dummies(df, columns=present_cat, prefix=present_cat)

    feat = [c for c in df.columns if c != "target"]
    for c in feat:
        if df[c].dtype == object or df[c].dtype == bool:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    nun = df[feat].nunique()
    const = [c for c in feat if nun[c] <= 1]
    if const:
        df = df.drop(columns=const)

    if verbose:
        log.info("UNSW: %d rows, %d features, %d classes",
                 len(df), df.shape[1] - 1, df["target"].nunique())
    return df


def check_counts(df: pd.DataFrame) -> dict:
    """Compare class counts with Table 3 and report, without failing."""
    got = df["target"].value_counts().to_dict()
    rows = []
    for k, exp in sorted(TABLE3.items(), key=lambda kv: -kv[1]):
        f = int(got.get(k, 0))
        rows.append((k, exp, f, "exact" if f == exp else "differs"))
    return {"rows": rows, "total": len(df)}


@dataclass
class Fold:
    fold: int
    seed: int
    clients: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    X_val: np.ndarray = None
    y_val: np.ndarray = None
    X_test: np.ndarray = None
    y_test: np.ndarray = None
    classes: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


def dirichlet_partition(y, n_clients, alpha, rng):
    shards = {c: [] for c in range(n_clients)}
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        p = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for cid, part in enumerate(np.split(idx, cuts)):
            if len(part):
                shards[cid].append(part)
    return {c: (np.concatenate(v) if v else np.array([], dtype=int))
            for c, v in shards.items()}


def per_client_smote(X, y, seed, cap=None, k_neighbors=5):
    """SMOTE on one client's own shard.

    A client can only rebalance classes it actually holds, so globally rare
    classes stay unevenly represented across the federation -- the intended
    difficulty.  `cap` bounds the target count so a client holding a large
    share of Normal does not blow its shard up.
    """
    from imblearn.over_sampling import SMOTE
    counts = np.bincount(y)
    present = np.where(counts > 0)[0]
    if len(present) < 2:
        return X, y
    smallest = counts[present].min()
    k = int(min(k_neighbors, max(1, smallest - 1)))
    viable = present[counts[present] > k]
    if len(viable) < 2:
        return X, y
    target = int(counts[present].max())
    if cap is not None:
        target = min(target, int(cap))
    strategy = {int(c): max(int(counts[c]), target) for c in viable}
    mask = np.isin(y, viable)
    try:
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
        Xr, yr = sm.fit_resample(X[mask], y[mask])
    except ValueError as e:
        log.warning("SMOTE skipped on a shard: %s", e)
        return X, y
    if (~mask).any():
        Xr = np.vstack([Xr, X[~mask]])
        yr = np.concatenate([yr, y[~mask]])
    return Xr.astype(np.float32), yr.astype(np.int64)


def make_folds(df, n_clients=10, n_folds=3, alpha=0.5, smote=True,
               smote_cap=20_000, seeds=(0, 1, 2, 3, 4),
               subsample=None, rng_seed=0, keep_whole_below=5_000) -> list[Fold]:
    feature_names = [c for c in df.columns if c != "target"]
    classes = sorted(df["target"].unique())
    c2i = {c: i for i, c in enumerate(classes)}

    if subsample is not None and subsample < len(df):
        # Rare classes are carried over whole.  Subsampling them proportionally
        # would leave Worm with a handful of records, which destroys both the
        # per-class metrics and the balanced-test comparison; the cost of
        # keeping them is negligible because they are rare by definition.
        frac = subsample / len(df)
        parts = []
        for cls, g in df.groupby("target", sort=True):
            n = len(g) if len(g) <= keep_whole_below else max(
                min(len(g), keep_whole_below), int(round(len(g) * frac)))
            parts.append(g.sample(min(n, len(g)), random_state=rng_seed))
        df = pd.concat(parts).sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)

    X_all = df[feature_names].to_numpy(dtype=np.float32)
    y_all = df["target"].map(c2i).to_numpy(dtype=np.int64)

    skf = StratifiedKFold(n_splits=max(n_folds, 5), shuffle=True, random_state=seeds[0])
    folds = []
    for k, (trval, test_idx) in enumerate(skf.split(X_all, y_all)):
        if k >= n_folds:
            break
        seed = seeds[k % len(seeds)]
        rng = np.random.default_rng(seed)
        tr_idx, val_idx = train_test_split(
            trval, test_size=1 / 8, stratify=y_all[trval], random_state=seed)

        lo = X_all[tr_idx].min(axis=0)
        hi = X_all[tr_idx].max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0).astype(np.float32)
        norm = lambda idx: np.clip((X_all[idx] - lo) / span, 0.0, 1.0).astype(np.float32)

        f = Fold(fold=k, seed=seed, classes=classes, feature_names=feature_names)
        f.X_val, f.y_val = norm(val_idx), y_all[val_idx]
        f.X_test, f.y_test = norm(test_idx), y_all[test_idx]

        shards = dirichlet_partition(y_all[tr_idx], n_clients, alpha, rng)
        for cid, local in shards.items():
            if len(local) < 20:
                continue
            g = tr_idx[local]
            Xc, yc = norm(g), y_all[g]
            if smote:
                Xc, yc = per_client_smote(Xc, yc, seed=seed + 1000 * cid, cap=smote_cap)
            f.clients[cid] = (Xc, yc)
        folds.append(f)
    return folds
