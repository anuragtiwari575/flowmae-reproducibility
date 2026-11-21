#!/usr/bin/env python3
"""
featvalid_v4.py -- structural validity of the adversarial examples.

This reads the preparation cache that featvalid_v3.py wrote, so it
starts attacking within seconds. It does not rebuild the folds.

WHAT CHANGED FROM v3, AND WHY
-----------------------------
v3 measured three things on CICIDS-2017. One worked and two did not.

  tailfrac  WORKED. 0.162, 0.175, 0.158 on the first three folds of
            arm B: graded, and stable to about +/-0.01 across folds.
            Kept unchanged.

  nnfar     SATURATED at 1.000 on every fold. This is a finding
            rather than a broken metric: at eps = 0.2 over 59
            perturbable columns the attack may displace a row by up
            to 0.2*sqrt(59) = 1.54 in the scaled space, which is far
            larger than the distance from a clean test row to the
            training set. So every adversarial example lies further
            from the corpus than the 99th percentile of real test
            rows. Worth one sentence of prose; useless as a table
            column, because it cannot separate the arms. Retained in
            the JSON, dropped from the table.

  nnratio   UNSTABLE: x443.50, x30.51, x26.08 across three folds of
            one arm. Taking a ratio of medians rather than a median
            of ratios was not enough. CICIDS-2017 contains many
            near-duplicate flow records and per-client SMOTE adds
            interpolated rows, so the median clean-row distance to
            the training set approaches zero in some folds and the
            ratio diverges. Removed. The absolute medians are
            reported in its place, which have a fixed scale and do
            not divide by a near-zero quantity.

WHAT v4 GOT WRONG, FIXED HERE
-----------------------------
v4 introduced `disp`, the displacement of a perturbed value in units
of its column's [0.1%, 99.9%] inter-quantile range. It returned
values in the thousands (2273, 1164, 2415), because that range is
essentially zero for columns which are practically constant in the
corpus -- a flag-like column that is 0 in more than 99.9% of rows --
and the floor on the divisor was only 1e-6. A displacement of up to
0.2 divided by 1e-6 is 2e5, and a handful of such columns dominated
the mean over 59.

v5 computes `disp` only over columns whose inter-quantile range
exceeds --span-min (default 0.01, i.e. one percent of the scaled
range). The exclusion is principled rather than cosmetic: the
measure asks how far the attack moves a feature relative to how far
that feature varies in real traffic, and for a feature that does not
vary the question has no answer. Perturbations of those columns are
trivially outside the data and are already counted by `tailfrac`.
The number of columns used is printed and stored.

v4 also reported `d_clean` per arm. It is arm-independent by
construction -- the clean rows and the reference sample do not depend
on the model -- and indeed came out identical across arms (0.0017 on
fold 0, 0.027 to 0.029 on the others). It is therefore context for
the caption, not a column of the table.

  d_adv     Median Euclidean distance from an adversarial row to its
            nearest training row, absolute, in the scaled
            69-dimensional space where the budget permits a
            displacement of at most eps*sqrt(#perturbable) = 1.54.
            Reported per arm. `d_clean` is reported once per corpus.

WHAT IS STILL NOT TESTED, AND SAID SO IN THE CAPTION
----------------------------------------------------
Sign and [min,max] range, because the [0,1] scaling and the box
projection enforce them; a check of those measures the projection.
Integrality, because scaling an integer column of range ~1e8 into
[0,1] gives a lattice step below float32 resolution -- only 4 of 69
columns retain identifiable integrality, and all four are frozen.
Inter-column identities (a rate against the count and duration it is
computed from), because those need the scaler parameters. A feasible
feature vector is not a realisable packet trace either.

USAGE
  python -u featvalid_v4.py --corpus cicids --task multiclass8 \\
      --device cuda --attack-bs 512 --rows 20000 \\
      --entry B:mc3gram:out/models_permissive \\
      --entry C:mc3gram:out/models_permissive \\
      --entry FB:flowmae:out/models_flowmae_cic \\
      --entry FMB:flowmae:out/models_flowmae_cic_mcp
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attack_suite_v4 as asuite           # noqa: E402
import flowmae_v4 as fm                    # noqa: E402
from model import MC3GRAM                  # noqa: E402


def nn_distance(Q, R, device, chunk=512):
    q = torch.as_tensor(Q, dtype=torch.float32, device=device)
    r = torch.as_tensor(R, dtype=torch.float32, device=device)
    out = torch.empty(q.shape[0], dtype=torch.float32, device=device)
    with torch.no_grad():
        for s in range(0, q.shape[0], chunk):
            e = min(s + chunk, q.shape[0])
            out[s:e] = torch.cdist(q[s:e], r).min(dim=1).values
    return out.detach().cpu().numpy()


def load_model(kind, ckpt, d_in, n_cls, a, dev):
    m = (MC3GRAM(d_in, n_cls) if kind == "mc3gram"
         else fm.FlowMAE(d_in, n_cls, d_model=a.fm_dmodel,
                         n_layers=a.fm_layers, n_head=a.fm_nhead,
                         p_drop=a.fm_dropout))
    try:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    except Exception:                                   # noqa: BLE001
        return None
    return m.to(dev).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="cicids")
    ap.add_argument("--task", default="multiclass8")
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--fm-dmodel", type=int, default=64)
    ap.add_argument("--fm-layers", type=int, default=3)
    ap.add_argument("--fm-nhead", type=int, default=4)
    ap.add_argument("--fm-dropout", type=float, default=0.1)
    ap.add_argument("--entry", action="append", required=True)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--attack-bs", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--nn-chunk", type=int, default=512)
    ap.add_argument("--span-min", type=float, default=0.01,
                    help="a column whose [0.1%%, 99.9%%] training "
                         "range is narrower than this does not vary "
                         "in the corpus, so a displacement relative "
                         "to its spread is undefined; such columns "
                         "are excluded from disp and counted by "
                         "tailfrac instead")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prep", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex", default="tab_feasibility.tex")
    a = ap.parse_args()

    if a.attack_bs > 0:
        for n in ("fgsm", "pgd", "cw_l2"):
            if hasattr(asuite, n):
                setattr(asuite, n,
                        functools.partial(getattr(asuite, n),
                                          bs=a.attack_bs))

    prep_path = Path(a.prep or
                     f"out/featvalid_prep_{a.corpus}_{a.task}"
                     f"_r{a.rows}.npz")
    if not prep_path.exists():
        print(f"  no preparation cache at {prep_path}")
        print("  run featvalid_v3.py once to build it, then re-run "
              "this script")
        return 2
    print(f"  reusing preparation cache {prep_path}")
    z = np.load(prep_path, allow_pickle=True)
    P = {k: z[k] for k in z.files}

    names = [str(x) for x in P["names"]]
    d = len(names)
    q_lo, q_hi = P["q_lo"].astype(np.float32), P["q_hi"].astype(np.float32)
    raw_span = (q_hi - q_lo).astype(np.float32)
    varies = raw_span > a.span_min
    span = np.maximum(raw_span, a.span_min).astype(np.float32)
    nn_ref = P["nn_ref"]
    n_cls = int(P["n_cls"])
    fold_ids = [int(x) for x in P["fold_ids"]]
    blocks = {str(k): np.asarray(v, dtype=np.int64)
              for k, v in zip(P["blocks_k"], P["blocks_v"])}

    import phase1_arms_v17 as drv
    spec = drv.build_spec(names, tuple(str(x) for x in P["frozen"]))
    dv = torch.device(a.device)

    print("=" * 74)
    print("STRUCTURAL VALIDITY OF THE ADVERSARIAL EXAMPLES  (v4)")
    print(f"  corpus {a.corpus}  eps {a.eps}  {a.rows} rows per fold")
    print(f"  {int(P['n_int'])} of {d} columns retain identifiable "
          f"integrality after")
    print("  the [0,1] scaling, so integrality is not testable here;")
    print("  sign and range are enforced by the box projection, so a")
    print("  check of them would measure the projection")
    print("=" * 74)

    res = {}
    for ent in a.entry:
        arm, kind, cache = ent.split(":", 2)
        cache = Path(cache)
        acc = defaultdict(list)
        for fi, fold in enumerate(fold_ids):
            model = None
            for h in sorted(cache.glob(f"{arm}_f{fold}_*.pt")):
                model = load_model(kind, h, d, n_cls, a, dv)
                if model is not None:
                    break
            if model is None:
                print(f"  {arm} fold {fold}: no loadable checkpoint")
                continue

            X = P[f"X{fi}"]
            y = P[f"y{fi}"]
            adv = asuite.pgd(model, X, y, a.eps, spec, steps=40, seed=0)

            moved = (np.abs(adv - X) > a.tol).any(0)
            nmv = int(moved.sum())
            if nmv == 0:
                print(f"  {arm} fold {fold}: attack moved nothing")
                continue

            outside = ((adv < q_lo - a.tol) | (adv > q_hi + a.tol))
            tailfrac = float(outside[:, moved].mean())
            tailany = float(outside.any(1).mean())

            use = moved & varies
            nuse = int(use.sum())
            disp = (float((np.abs(adv[:, use] - X[:, use])
                           / span[use]).mean()) if nuse else float("nan"))

            if blocks:
                b = np.zeros(len(adv), dtype=bool)
                for cols in blocks.values():
                    b |= ((adv[:, cols] >= 0.5).sum(1) != 1)
                onehot = float(b.mean())
            else:
                onehot = 0.0

            d_adv = nn_distance(adv, nn_ref, dv, a.nn_chunk)
            d_cln = nn_distance(X, nn_ref, dv, a.nn_chunk)
            m_adv, m_cln = float(np.median(d_adv)), float(np.median(d_cln))
            nnfar = float((d_adv > np.quantile(d_cln, 0.99)).mean())

            acc["tailfrac"].append(tailfrac)
            acc["disp"].append(disp)
            acc["d_adv"].append(m_adv)
            acc["d_clean"].append(m_cln)
            acc["nnfar"].append(nnfar)
            acc["tailany"].append(tailany)
            acc["onehot"].append(onehot)
            print(f"  {arm} fold {fold}: tailfrac {tailfrac:.3f}"
                  f"  disp {disp:.3f}  d_adv {m_adv:.3f}"
                  f"  d_clean {m_cln:.4f}  nnfar {nnfar:.3f}"
                  f"  ({nuse}/{int(moved.sum())} cols vary)")
        if acc:
            res[arm] = {k: float(np.mean(v)) for k, v in acc.items()}
            res[arm]["d_adv_sd"] = float(np.std(acc["d_adv"]))
            res[arm]["tailfrac_sd"] = float(np.std(acc["tailfrac"]))

    keys = ["tailfrac", "disp", "d_adv"]
    if blocks:
        keys.append("onehot")
    # d_clean does not depend on the arm; report it once
    dcl = (float(np.mean([r["d_clean"] for r in res.values()]))
           if res else float("nan"))
    nvary = int((varies).sum())

    print()
    print("  " + "arm".ljust(8) + "".join(k.rjust(11) for k in keys))
    for arm, r in res.items():
        print("  " + arm.ljust(8)
              + "".join(f"{r[k]:11.3f}" for k in keys))
    print()
    print("  fold-to-fold stability (sd over folds):")
    for arm, r in res.items():
        print(f"    {arm.ljust(8)} tailfrac {r['tailfrac_sd']:.4f}"
              f"   d_adv {r['d_adv_sd']:.4f}")
    if res:
        print(f"\n  nnfar is {min(r['nnfar'] for r in res.values()):.3f}"
              f" to {max(r['nnfar'] for r in res.values()):.3f} across "
              f"arms; if that is 1.000 it belongs in prose, not the "
              f"table")

    out = a.out or f"out/featvalid_{a.corpus}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"corpus": a.corpus, "eps": a.eps, "rows_per_fold": a.rows,
         "n_columns": d, "n_integer_identifiable": int(P["n_int"]),
         "n_onehot_blocks": len(blocks), "result": res}, indent=2),
        encoding="utf-8")

    head = {"tailfrac": r"\textbf{in tail}",
            "disp": r"\textbf{displacement}",
            "d_adv": r"$\boldsymbol{d_{\mathrm{adv}}}$",
            "onehot": r"\textbf{one-hot}"}
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Structural validity of the adversarial examples "
         rf"at $\epsb = {a.eps}$ on {a.corpus.upper()}, mean over five "
         r"folds. \emph{in tail}: the fraction of perturbed feature "
         r"values falling outside the $[0.1\%, 99.9\%]$ quantile "
         r"range of their column in the training partition. "
         rf"\emph{{displacement}}: the mean movement of a perturbed "
         rf"value in units of that column's $[0.1\%, 99.9\%]$ "
         rf"inter-quantile range, over the {nvary} columns whose "
         rf"range exceeds $0.01$; a column that does not vary in the "
         rf"corpus has no spread to measure against, and "
         rf"perturbations of it are counted by \emph{{in tail}} "
         rf"instead. $d_{{\mathrm{{adv}}}}$: median Euclidean "
         rf"distance from an adversarial row to its nearest training "
         rf"row, in the scaled feature space where the budget "
         rf"permits a displacement of at most $\epsb\sqrt{{59}} = "
         rf"1.54$; the same median for the \emph{{clean}} rows is "
         rf"${dcl:.4f}$ and does not depend on the arm. Sign "
         r"and $[\min,\max]$ range are not reported because the "
         r"$[0,1]$ scaling and the box projection enforce them; "
         r"integrality is not reported because the scaling does not "
         r"preserve it. A feasible feature vector is still not a "
         r"realisable packet trace, and the inter-column identities "
         r"tying a rate to its count and duration are not tested "
         r"here. What these figures bound is how far outside the "
         r"corpus the perturbations lie, and the comparison across "
         r"arms is what bears on \Cref{sec:res-strength}.}",
         r"\label{tab:feasibility}", r"\small",
         rf"\begin{{tabular}}{{@{{}}l{'c' * len(keys)}@{{}}}}",
         r"\toprule",
         r"\textbf{Arm} & " + " & ".join(head[k] for k in keys)
         + r" \\", r"\midrule"]
    for arm, r in res.items():
        L.append(rf"{arm} & " + " & ".join(
            f"{r[k]:.3f}" for k in keys) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(a.tex).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  wrote {out} and {a.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
