"""
ciciot_loader_v7.py -- CIC-IoT-2023, the release the paper describes.

What changed, and why there is a v7
-----------------------------------
v1-v6 were built against a copy that turned out not to be the release. That
copy (63 merged files of 40 columns, plus a 309-file per-class tree of 39)
carries a column the release does not -- Time_To_Live -- and lacks four the
release has: flow_duration, Duration, Srate, Drate. It is a different run of
the feature extractor, not a repackaging, and it matches Table 3 of Neto et
al. (2023) on no class at all.

The release does. Measured 2026-11-15 over its 169 merged files:
every one of the 34 class counts exact, total 46,686,579 exact, 47 columns
(46 features plus `label`). So G3 here asserts against the PUBLISHED counts
again, which is where a gate belongs when the published numbers are right.

Everything measured on the old copy is withdrawn, not adjusted: the declared
corpus, its sample, the Phase 6 reference points and the frozen-feature list
all describe a corpus nobody else is using.

The header is discovered, never assumed
---------------------------------------
The earlier mistake was comparing a header from memory against one on disk
and trusting the memory. Nothing here hardcodes a column name. The header is
read from the files, checked for consistency across all 169, and the
frozen/perturbable split is derived by rule and printed in full so it can be
checked against the actual header rather than against a recollection of it.

The split, and the reasoning
----------------------------
An adversary shaping traffic can change when packets go out and how big they
are. It cannot set a TCP flag to 0.3, nor a protocol indicator, nor an
integer count of packets.

  perturbable   flow_duration, Duration, Rate, Srate, Drate, IAT
                Tot sum, Tot size, Min, Max, AVG, Std, Variance
                Magnitue, Radius, Covariance, Weight
  frozen        Protocol Type, Header_Length, Number
                the seven *_flag_number indicators
                the five *_count fields
                the fourteen protocol indicators HTTP ... LLC

Usage
-----
    python ciciot_loader_v7.py --root <wataiData> --check
    python ciciot_loader_v7.py --root <wataiData> --build --task multiclass8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ciciot_loader_v2 import (                      # noqa: E402
    COUNTS_34, CATEGORY_8, CATEGORY_TOTALS, TOTAL, MISSING,
    _norm, _NORM_TO_CANON,
)

VERSION = "ciciot_loader_v7"

N_FILES = 169
N_COLUMNS = 47          # 46 features + label

DEFAULT_RATE = 0.05
DEFAULT_SEED = 20231

# Patterns, not names: applied to whatever header the files actually carry,
# and the resolved split is printed for checking.
FROZEN_PATTERNS = (
    "protocol type", "header_length", "flag_number", "_count", "number",
    "http", "https", "dns", "telnet", "smtp", "ssh", "irc",
    "tcp", "udp", "dhcp", "arp", "icmp", "igmp", "ipv", "llc",
)


def is_frozen(col: str) -> bool:
    c = str(col).strip().lower()
    return any(p in c for p in FROZEN_PATTERNS)


def label_col_or_none(cols):
    for c in cols:
        if str(c).strip().lower() == "label":
            return c
    return None


def find_csvs(root: str | Path) -> list[Path]:
    root = Path(root)
    for cand in (root / "csv" / "CICIoT2023", root / "CICIoT2023", root):
        if cand.is_dir():
            files = sorted(p for p in cand.glob("*.csv") if p.is_file())
            if files:
                return files
    raise FileNotFoundError(f"no .csv files found under {root}")


def header_of(root) -> tuple[list[str], str]:
    """One header for the whole release, or a failure naming the variants."""
    files = find_csvs(root)
    variants: dict[str, list[str]] = {}
    for p in files:
        h = "|".join(str(c).strip() for c in pd.read_csv(p, nrows=0).columns)
        variants.setdefault(h, []).append(p.name)
    if len(variants) != 1:
        lines = [f"{len(k.split('|'))} cols in {len(v)} file(s), e.g. {v[:2]}"
                 for k, v in variants.items()]
        raise ValueError("the release does not carry one header: "
                         + "; ".join(lines))
    head = list(variants)[0].split("|")
    lc = label_col_or_none(head)
    if lc is None:
        raise ValueError(f"no label column; header begins {head[:6]}")
    return head, lc


# -------------------------------------------------------------- G3, cached
def stream_counts(root, cache=None, chunk: int = 2_000_000,
                  verbose: bool = True) -> dict:
    files = find_csvs(root)
    cpath = Path(cache) / "ciciot_release_counts.json" if cache else None
    if cpath and cpath.exists():
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        if rec.get("version") == VERSION and rec.get("n_files") == len(files):
            if verbose:
                print(f"  counts from cache: {cpath}")
            return {k: int(v) for k, v in rec["counts"].items()}

    _, lc = header_of(root)
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        for part in pd.read_csv(p, usecols=[lc], chunksize=chunk,
                                low_memory=False):
            col = part[lc]
            nmiss = int(col.isna().sum())
            if nmiss:
                counts[MISSING] = counts.get(MISSING, 0) + nmiss
            for k, v in col.dropna().astype(str).str.strip(
            ).value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
        if verbose and (i % 20 == 0 or i == len(files)):
            print(f"  [{i:3d}/{len(files)}] {sum(counts.values()):,} rows"
                  f"   {time.time() - t0:.0f}s", flush=True)
    if cpath:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(
            {"version": VERSION, "n_files": len(files), "counts": counts},
            indent=1), encoding="utf-8")
    return counts


def fold_counts(raw: dict) -> dict:
    out: dict[str, int] = {}
    for k, v in raw.items():
        c = MISSING if k == MISSING else _NORM_TO_CANON.get(_norm(k), str(k))
        out[c] = out.get(c, 0) + int(v)
    return out


def g3(root, cache=None, verbose=True) -> dict:
    """Assert the release against Table 3. It matches, so the gate holds the
    published numbers rather than this copy's own."""
    got = fold_counts(stream_counts(root, cache=cache, verbose=verbose))
    n_missing = int(got.pop(MISSING, 0))
    total = sum(got.values())
    bad = [f"{k}: got {got.get(k, 0):,}, Table 3 says {v:,}"
           for k, v in COUNTS_34.items() if got.get(k, 0) != v]
    extra = sorted(k for k in got if k not in COUNTS_34)
    if total != TOTAL:
        bad.append(f"TOTAL: got {total:,}, Table 3 says {TOTAL:,}")
    if n_missing:
        bad.append(f"{n_missing} rows carry no label; the release has none")
    if extra:
        bad.append(f"labels not in Table 3: {extra[:5]}")
    return {"counts": got, "total": total, "bad": bad, "ok": not bad}


# ------------------------------------------------------------ the sample
def _mask(n, rate, seed, *key) -> np.ndarray:
    h = hashlib.sha1("|".join(map(str, (seed,) + key)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.random(n) < rate


def load_ciciot(root, task: str = "multiclass8", rate: float = DEFAULT_RATE,
                seed: int = DEFAULT_SEED, cache=None, chunk: int = 1_000_000,
                verbose: bool = True) -> pd.DataFrame:
    if task not in ("binary", "multiclass8", "multiclass34"):
        raise ValueError(task)
    if not 0 < rate <= 1.0:
        raise ValueError(rate)

    cpath = (Path(cache) / f"ciciot_release_{task}_r{rate:g}_s{seed}.parquet"
             if cache else None)
    if cpath and cpath.exists():
        if verbose:
            print(f"  sample from cache: {cpath}")
        return pd.read_parquet(cpath)

    files = find_csvs(root)
    head, lc = header_of(root)
    parts, t0 = [], time.time()
    for fi, p in enumerate(files):
        for ci, raw in enumerate(pd.read_csv(p, chunksize=chunk,
                                             low_memory=False)):
            raw.columns = [str(c).strip() for c in raw.columns]
            m = _mask(len(raw), rate, seed, p.name, ci)
            if m.any():
                parts.append(raw.loc[m].copy())
        if verbose and (fi + 1) % 20 == 0:
            print(f"  [{fi + 1:3d}/{len(files)}] kept "
                  f"{sum(len(q) for q in parts):,}   {time.time() - t0:.0f}s",
                  flush=True)

    df = pd.concat(parts, ignore_index=True)
    del parts

    y = df[lc].map(lambda v: _NORM_TO_CANON.get(_norm(v), str(v).strip()))
    X = df.drop(columns=[lc])
    del df

    unknown = sorted(set(y.unique()) - set(COUNTS_34))
    if unknown:
        raise ValueError(f"labels not in Table 3: {unknown[:8]}")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    nun = X.nunique()
    const = [c for c in X.columns if nun[c] <= 1]
    if const:
        X = X.drop(columns=const)
        if verbose:
            print(f"  dropped {len(const)} constant column(s): {const}")

    if task == "binary":
        y = np.where(np.asarray(y) == "BenignTraffic", "BenignTraffic",
                     "Attack")
    elif task == "multiclass8":
        y = np.asarray([CATEGORY_8[v] for v in y])
    else:
        y = np.asarray(y)

    out = X.reset_index(drop=True)
    out["target"] = y
    if verbose:
        print(f"  CIC-IoT-2023 release: {len(out):,} rows, "
              f"{out.shape[1] - 1} features, {out['target'].nunique()} "
              f"classes  (task={task}, rate={rate:g}, seed={seed})")
    if cpath:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cpath, index=False)
        if verbose:
            print(f"  cached to {cpath}")
    return out


def reference_counts(task: str) -> dict:
    if task == "multiclass34":
        src = COUNTS_34
    elif task == "multiclass8":
        src = CATEGORY_TOTALS
    elif task == "binary":
        b = COUNTS_34["BenignTraffic"]
        src = {"BenignTraffic": b, "Attack": TOTAL - b}
    else:
        raise ValueError(task)
    return {k: v / TOTAL for k, v in src.items()}


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", default="out/ciciot_cache")
    ap.add_argument("--task", default="multiclass8",
                    choices=["binary", "multiclass8", "multiclass34"])
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    print(f"{VERSION}\n")
    files = find_csvs(args.root)
    print(f"[release] {len(files)} files, "
          f"{sum(p.stat().st_size for p in files) / 1e9:.2f} GB")
    if len(files) != N_FILES:
        print(f"  STOP: expected {N_FILES} files, found {len(files)}. This "
              "is not the release this loader describes.")
        return 1

    head, lc = header_of(args.root)
    feats = [c for c in head if c != lc]
    print(f"  {len(head)} columns, label {lc!r}, {len(feats)} features")
    if len(head) != N_COLUMNS:
        print(f"  STOP: expected {N_COLUMNS} columns, found {len(head)}.")
        return 1

    frozen = [c for c in feats if is_frozen(c)]
    perturbable = [c for c in feats if not is_frozen(c)]
    print(f"\n[feature spec] {len(perturbable)}/{len(feats)} perturbable")
    print(f"  perturbable ({len(perturbable)}): {perturbable}")
    print(f"  frozen      ({len(frozen)}): {frozen}")
    print("  Check this split against the header above before any attack "
          "result is believed.")

    print(f"\n[G3] against Table 3 of Neto et al. (2023)")
    rep = g3(args.root, cache=args.cache)
    if not rep["ok"]:
        print(f"  G3 FAILS, {len(rep['bad'])} problem(s):")
        for b in rep["bad"][:10]:
            print(f"    {b}")
        return 1
    print(f"  G3 ok: all 34 class counts and the total "
          f"({rep['total']:,}) match exactly.")

    if args.check or not args.build:
        print("\n  --check: stopping here.")
        return 0

    exp = int(TOTAL * args.rate)
    print(f"\n[sample] uniform rate={args.rate:g}, seed={args.seed}, "
          f"task={args.task}")
    print(f"  expecting about {exp:,} rows "
          f"(~{exp * len(feats) * 4 / 1024 ** 3:.2f} GiB for X in float32)")
    df = load_ciciot(args.root, task=args.task, rate=args.rate,
                     seed=args.seed, cache=args.cache)

    print(f"\n[G2] sample prevalence against the published shares")
    ref = reference_counts(args.task)
    vc = df["target"].value_counts()
    worst = 0.0
    print(f"  {'class':<28} {'corpus':>9} {'sample':>9} {'n':>10}")
    for k in sorted(ref, key=lambda z: -ref[z]):
        n = int(vc.get(k, 0))
        worst = max(worst, abs(n / len(df) - ref[k]))
        print(f"  {k:<28} {ref[k]:>9.5f} {n / len(df):>9.5f} {n:>10,}")
    print(f"\n  largest absolute deviation {worst:.5f}  (tolerance 0.01)")
    if worst > 0.01:
        print("  G2 FAILS.")
        return 1
    print("  G2 ok.")
    print(f"\n  smallest class: {vc.idxmin()} at {vc.min():,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
