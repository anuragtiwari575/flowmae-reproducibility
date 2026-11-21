#!/usr/bin/env python3
"""
strong_pgd_v3.py -- the 200-step ten-restart comparison, on either corpus.

WHY v3
------
v2 runs on UNSW-NB15 but dies on CICIDS-2017 with the same
out-of-memory failure the driver had: it imports flowmae_v2, whose
forward pass is not chunked, and the evaluation batch of 4096 rows over
69 columns asks for more than the card has. v3 imports flowmae_v4, so
inference is chunked, and adds --attack-bs so the attack itself is
batched too. Neither changes a number: chunking is exact for this
architecture, which has no operation that mixes rows.

WHAT IT MEASURES
----------------
For each arm, at each budget, under two attack configurations:

  standard  40 steps, alpha = eps/10, one random start, seed 0
  strong    200 steps, alpha = eps/25, ten random starts with seeds
            0..9, keeping for each ROW the restart that maximises its
            loss -- a row broken by any restart is broken

and reports macro-F1 under each, the per-row flip rate under each, and
whether each curve is monotone in the budget.

The monotonicity test is applied to these raw curves, not to a running
minimum over them. A running minimum is non-increasing by construction
and can never show the rise that certifies attack failure.

USAGE
  python -u strong_pgd_v3.py --corpus cicids --task multiclass8 \
      --root data/cicids2017 --device cuda --attack-bs 512 \
      --entry B:mc3gram:out/models_permissive \
      --entry FB:flowmae:out/models_flowmae_cic \
      --entry FMB:flowmae:out/models_flowmae_cic_mcp \
      --out out/strong_pgd_cicids.txt
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as ds                       # noqa: E402
import ciciot_loader_v7 as ciot            # noqa: E402
import flowmae_v4 as fm                    # noqa: E402  (chunked inference)
import attack_suite_v4 as asuite           # noqa: E402
from model import MC3GRAM                  # noqa: E402


def macro_f1(model, X, y, bs=1024):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def preds(model, X, bs=1024):
    return macro_f1(model, X, None, bs)


def pgd_multi(model, X, y, eps, spec, steps, restarts, bs=1024):
    """Per-row best restart. Averaging restarts would let rows that
    survived one hide behind rows that did not."""
    dev = next(model.parameters()).device
    best_x = best_loss = None
    for r in range(restarts):
        adv = asuite.pgd(model, X, y, eps, spec, steps=steps,
                         alpha=max(eps / 25.0, 1e-6),
                         random_start=True, seed=r)
        losses = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(adv), bs):
                xb = torch.from_numpy(adv[i:i + bs]).to(dev)
                yb = torch.from_numpy(y[i:i + bs]).to(dev)
                losses.append(F.cross_entropy(model(xb), yb,
                                              reduction="none").cpu().numpy())
        loss = np.concatenate(losses)
        if best_x is None:
            best_x, best_loss = adv, loss
        else:
            take = loss > best_loss
            best_x[take] = adv[take]
            best_loss[take] = loss[take]
    return best_x


def load_model(kind, ckpt, d_in, n_cls, a, dev):
    if kind == "mc3gram":
        m = MC3GRAM(d_in, n_cls)
    elif kind == "flowmae":
        m = fm.FlowMAE(d_in, n_cls, d_model=a.fm_dmodel,
                       n_layers=a.fm_layers, n_head=a.fm_nhead,
                       p_drop=a.fm_dropout)
    else:
        raise SystemExit(f"unknown model kind {kind!r}")
    sd = torch.load(ckpt, map_location="cpu")
    try:
        m.load_state_dict(sd)
    except Exception:                                   # noqa: BLE001
        return None
    return m.to(dev).eval()


def pick(cache: Path, arm, fold, kind, d_in, n_cls, a, dev):
    """Select by successful strict load, which is a shape test: one
    cache directory can hold checkpoints from two corpora."""
    for h in sorted(cache.glob(f"{arm}_f{fold}_*.pt")):
        m = load_model(kind, h, d_in, n_cls, a, dev)
        if m is not None:
            return m, h
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="unsw")
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
    ap.add_argument("--fm-dmodel", type=int, default=64)
    ap.add_argument("--fm-layers", type=int, default=3)
    ap.add_argument("--fm-nhead", type=int, default=4)
    ap.add_argument("--fm-dropout", type=float, default=0.1)
    ap.add_argument("--entry", action="append", required=True,
                    help="arm:kind:cachedir")
    ap.add_argument("--eps", default="0.01,0.05,0.1,0.2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--attack-bs", type=int, default=0,
                    help="cap the batch inside the attack suite; 0 "
                         "leaves its default. Does not change results.")
    ap.add_argument("--eval-bs", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/strong_pgd_v3.txt")
    a = ap.parse_args()

    if a.attack_bs > 0:
        for n in ("fgsm", "pgd", "cw_l2"):
            if hasattr(asuite, n):
                setattr(asuite, n,
                        functools.partial(getattr(asuite, n), bs=a.attack_bs))
        print(f"  attack batch size forced to {a.attack_bs}", flush=True)

    dv = torch.device(a.device)
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
    names = list(folds[0].feature_names)

    X0 = folds[0].X_test
    fz = tuple(n for i, n in enumerate(names)
               if (lambda u: u.size <= 2
                   and set(np.round(u, 6).tolist()) <= {0.0, 1.0})(
                       np.unique(X0[:, i])))
    spec = drv.build_spec(names, fz)

    entries = [tuple(e.split(":", 2)) for e in a.entry]
    eps_list = [float(x) for x in a.eps.split(",")]

    lines = []
    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 78)
    say("STRONG PGD  (strong_pgd_v3)")
    say(f"  corpus {a.corpus}  task {a.task}  binary-only spec "
        f"({int(spec.perturbable.sum())}/{spec.d} perturbable)")
    say(f"  standard: 40 steps, alpha eps/10, 1 restart, seed 0")
    say(f"  strong  : {a.steps} steps, alpha eps/25, {a.restarts} "
        f"restarts, per-row best")
    say(f"  {a.rows} test rows per fold")
    say("  monotonicity is tested on these RAW curves, not on a running")
    say("  minimum over them, which would be monotone by construction.")
    say("=" * 78)

    results = {}
    for arm, kind, cache in entries:
        cache = Path(cache)
        std_f1 = {e: [] for e in eps_list}
        str_f1 = {e: [] for e in eps_list}
        std_fl = {e: [] for e in eps_list}
        str_fl = {e: [] for e in eps_list}
        cleans = []
        for f in folds:
            model, used = pick(cache, arm, f.fold, kind, len(names),
                               len(f.classes), a, dv)
            if model is None:
                say(f"  {arm} fold {f.fold}: no loadable checkpoint in "
                    f"{cache}, skipped")
                continue
            rng = np.random.default_rng(f.seed)
            idx = rng.permutation(len(f.X_test))[:a.rows]
            X = np.ascontiguousarray(f.X_test[idx], dtype=np.float32)
            y = np.ascontiguousarray(f.y_test[idx])

            p0 = macro_f1(model, X, y, a.eval_bs)
            ok = (p0 == y)
            cleans.append(float(f1_score(y, p0, average="macro",
                                         zero_division=0)))
            for e in eps_list:
                adv = asuite.pgd(model, X, y, e, spec, steps=40, seed=0)
                p = macro_f1(model, adv, y, a.eval_bs)
                std_f1[e].append(float(f1_score(y, p, average="macro",
                                                zero_division=0)))
                std_fl[e].append(float((p[ok] != y[ok]).mean())
                                 if ok.any() else float("nan"))

                adv = pgd_multi(model, X, y, e, spec, a.steps,
                                a.restarts, a.eval_bs)
                p = macro_f1(model, adv, y, a.eval_bs)
                str_f1[e].append(float(f1_score(y, p, average="macro",
                                                zero_division=0)))
                str_fl[e].append(float((p[ok] != y[ok]).mean())
                                 if ok.any() else float("nan"))
            say(f"  {arm} fold {f.fold}: done")
        if not cleans:
            continue
        results[arm] = {
            "clean": float(np.mean(cleans)),
            "standard": {str(e): float(np.mean(v)) for e, v in std_f1.items()},
            "strong":   {str(e): float(np.mean(v)) for e, v in str_f1.items()},
            "flip_standard": {str(e): float(np.nanmean(v))
                              for e, v in std_fl.items()},
            "flip_strong":   {str(e): float(np.nanmean(v))
                              for e, v in str_fl.items()},
        }

    say()
    say("  " + "arm".ljust(6) + "attack".ljust(10) + "clean".rjust(8)
        + "".join(f"{e:>9}" for e in eps_list) + "   monotone?")
    for arm, r in results.items():
        for which, nice in (("standard", "40-step"), ("strong", "200x10")):
            vals = [r[which][str(e)] for e in eps_list]
            mono = all(vals[i] >= vals[i + 1] - 1e-9
                       for i in range(len(vals) - 1))
            say("  " + arm.ljust(6) + nice.ljust(10)
                + f"{r['clean']:8.4f}"
                + "".join(f"{v:9.4f}" for v in vals)
                + ("   yes" if mono else "   NO  <-- attack failed here"))

    say()
    say("  flip rate (fraction of correctly classified rows changed)")
    say("  " + "arm".ljust(6) + "attack".ljust(10)
        + "".join(f"{e:>9}" for e in eps_list))
    for arm, r in results.items():
        for which, nice in (("flip_standard", "40-step"),
                            ("flip_strong", "200x10")):
            say("  " + arm.ljust(6) + nice.ljust(10)
                + "".join(f"{r[which][str(e)]:9.4f}" for e in eps_list))

    say()
    top = max(eps_list)
    for arm, r in results.items():
        say(f"  {arm}: at eps={top}, the strong attack costs a further "
            f"{r['standard'][str(top)] - r['strong'][str(top)]:+.4f} "
            f"macro-F1 and raises the flip rate from "
            f"{r['flip_standard'][str(top)]:.4f} to "
            f"{r['flip_strong'][str(top)]:.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    say(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
