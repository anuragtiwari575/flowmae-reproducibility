"""
analyze_v6.py -- turn the re-measurement's results.jsonl files into the
manuscript's tables, with the statistics done once and done properly.

The repository's tables.py reads the old supplementary schema and cannot read
these rows, so this is its replacement for the re-measurement.

What changed in v6
------------------
Grouping moves from the attack spec to the spec PAIR that phase1_arms_v17
records. v17 can train a defence against one perturbable set and attack it
under another, and such a run's ATTACK spec equals a matched run's attack
spec -- so grouping on the attack spec alone would put a defence hardened
against the declared set in the same table as one hardened against the
permissive set. `spec_pair` is 'train>attack' when they differ and the single
fingerprint when they match, so every earlier row groups exactly as before.

What changed in v5
------------------
A paired test on CLEAN macro-F1, which v4 and every earlier version omitted.
The augmentation claim is about detection on unperturbed traffic, so it is a
comparison of the clean column -- and that column was printed in the attack
tables as a mean per arm with no test beside it, which invites reading a
difference off two means. D -> A is the contrast that bears on the claim.

What changed in v4, and why it mattered
---------------------------------------
v3 grouped rows by (corpus, task) alone. phase1_arms_v16 added a second
dimension that v3 cannot see: `spec_fingerprint`, the hash of the set of
columns an adversary is permitted to move. Every attack number in the paper
is conditional on that set, so two rows with the same corpus, task, arm and
fold but different specs are two different measurements.

Under v3 they collided. The index was f1[arm][attack][eps][fold], so the
second spec's row did not average with the first -- it OVERWROTE it, silently,
and the printed table was a mixture of two experiments with no sign that it
was. The spec fingerprint was added to the resume key for exactly this reason
and the fix never reached the analysis side.

v4 groups by (corpus, task, spec), reports each spec's perturbable count, and
then does what the second spec was run for: a paired, cross-spec comparison of
envelope retention per arm. That last number -- how much measured robustness
moves when the perturbable set is redeclared and nothing else changes -- is
not reported by any FL-IDS paper we have read, and it cannot be computed from
a single spec.

Rows written before v16 carry no fingerprint. v4 labels them pre16[np=<n>]
rather than assuming they used the same declared set as a v16 run: the
declared list could have changed between drivers, and the row does not record
enough to prove it did not. The label is printed and carried into the tables.

What it enforces
----------------
  G1  every row must carry version, protocol, corpus, arm, fold, attack, eps;
      a row without them is reported and dropped.

      device and fingerprint are required only of rows written by a driver
      that recorded them: device arrived in phase1_arms_v7 and fingerprint in
      v8. Older rows are accepted with the value marked INFERRED, never filled
      in silently -- the v6 driver had no device option and no CUDA path, so
      cpu is a fact about that code rather than a guess, and the label says
      where it came from. A row missing them from a driver that should have
      written them is still a G1 failure.
  G4  FNR = 1 - Recall on every row, and the macro row flagged as mean-of-F1
  G6  a baseline row without repo_url and commit is refused
  G5  macro-F1 non-increasing in eps -- applied to the WORST-CASE ENVELOPE,
      not to each attack separately (see below)
  G9  in a cross-spec comparison the clean macro-F1 of the two specs must
      agree per arm and fold, because redeclaring the perturbable set changes
      the attacks and nothing else. If clean F1 moved, something besides the
      spec moved too and the difference cannot be attributed to the spec; the
      comparison is printed with that warning rather than withheld.

Why G5 moved to an envelope
---------------------------
FGSM is a single step, so it is not monotone in eps: at a large budget it
overshoots and can land in correctly-classified regions. On CICIDS-2017 arm D
fold 1 it went 0.2285, 0.0765, 0.0715, 0.0911 across eps = 0.01 .. 0.2, a rise
of +0.0196 at the top budget. PGD, being iterative over a ball that contains
the smaller ball, stayed monotone on the same model (0.1860, 0.0181, 0.0065,
0.0005). So a non-monotone FGSM curve is a property of the attack, not a
broken operator, and failing G5 on it would be wrong.

The fix is to report what a security evaluation should report anyway: at each
budget, the WORST result over every attack at that budget or below. The
defender does not get to choose the attacker's schedule. That envelope is
non-increasing by construction, so G5 applies to it meaningfully, and FGSM's
own curve is still printed with the non-monotonicity flagged as a note.

A statistical limit, stated rather than discovered later
--------------------------------------------------------
With five folds the two-sided exact Wilcoxon signed-rank test cannot return
less than 2/2^5 = 0.0625, so it CANNOT reject at the 0.05 level however
cleanly the folds separate; one-sided its floor is 0.03125. Both are printed
with that floor beside them, so neither we nor a reviewer reads a p of 0.0625
as a near miss. The paired t-test is the test with power at this n, and the
manuscript says so.

Usage
-----
    python analyze_v5.py out/phase4_ciciot.jsonl
    python analyze_v5.py out/cicids_m8.jsonl out/cicids_bin.jsonl --latex tables.tex
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sps
except Exception:                                  # scipy is in requirements
    sps = None

VERSION = "analyze_v6"
REQUIRED = ("version", "protocol", "corpus", "arm", "fold", "attack", "eps")
SPEC_FROM = 16           # phase1_arms_v16 first recorded "spec_fingerprint"
DEVICE_FROM = 7          # phase1_arms_v7 first recorded "device"
FINGERPRINT_FROM = 8     # v8 first recorded "fingerprint"
ARM_NOTE = {"D": "neither", "A": "generator only",
            "B": "PGD adv. training", "C": "both"}


# ---------------------------------------------------------------- loading
def _vnum(v) -> int | None:
    m = re.search(r"_v(\d+)$", str(v))
    return int(m.group(1)) if m else None


def spec_of(r: dict) -> str:
    """The perturbable-set identity of a row, as a short printable label.

    A v16+ row carries the fingerprint. An older row does not, and we refuse
    to assume it matches a v16 declared run: the declared list lives in the
    driver and could have changed. The perturbable count is the most that can
    honestly be said about such a row, so it goes in the label.
    """
    fp = r.get("spec_pair") or r.get("spec_fingerprint")
    if fp:
        return str(fp)
    np_ = r.get("n_perturbable")
    return f"pre16[np={np_}]" if np_ is not None else "pre16[np=?]"


def spec_descr(sub: list[dict]) -> str:
    """One line describing a spec group, from the rows themselves."""
    nps = sorted({r.get("n_perturbable") for r in sub
                  if r.get("n_perturbable") is not None})
    nfs = sorted({r.get("n_features") for r in sub
                  if r.get("n_features") is not None})
    modes = sorted({str(r.get("spec_mode")) for r in sub
                    if r.get("spec_mode")})
    npt = sorted({r.get("n_perturbable_train") for r in sub
                  if r.get("n_perturbable_train") is not None})
    names = sorted({str(r.get("spec")) for r in sub if r.get("spec")})
    bits = []
    if modes:
        bits.append("train>attack=" + ",".join(modes))
    if nps and nfs:
        bits.append(f"{'/'.join(str(x) for x in nps)} of "
                    f"{'/'.join(str(x) for x in nfs)} features perturbable")
    elif nps:
        bits.append(f"{'/'.join(str(x) for x in nps)} perturbable")
    if names:
        bits.append("spec=" + ",".join(names))
    if npt and nps and npt != nps:
        bits.append(f"MISMATCH: hardened on "
                    f"{'/'.join(str(x) for x in npt)} perturbable, attacked "
                    f"on {'/'.join(str(x) for x in nps)}")
    if len(nps) > 1:
        bits.append("*** one fingerprint, more than one perturbable count: "
                    "the fingerprint is not identifying the spec")
    return "; ".join(bits) if bits else "no spec metadata on these rows"


def load(paths) -> tuple[list[dict], list[str], int]:
    rows, problems, inferred = [], [], 0
    for p in paths:
        for i, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{p}:{i} unparseable ({e})")
                continue
            missing = [f for f in REQUIRED if f not in r]
            if missing:
                problems.append(f"{p}:{i} [G1] missing {missing}")
                continue

            vn = _vnum(r["version"])
            if "device" not in r:
                if vn is not None and vn < DEVICE_FROM:
                    r["device"] = "cpu[INFERRED]"
                    inferred += 1
                else:
                    problems.append(f"{p}:{i} [G1] no device, and "
                                    f"{r['version']} should record one")
                    continue
            if "fingerprint" not in r:
                if vn is not None and vn < FINGERPRINT_FROM:
                    r["fingerprint"] = f"legacy[{r['version']}]"
                else:
                    problems.append(f"{p}:{i} [G1] no fingerprint, and "
                                    f"{r['version']} should record one")
                    continue
            if r.get("is_baseline") and not (r.get("repo_url") and r.get("commit")):
                problems.append(f"{p}:{i} [G6] baseline '{r.get('method','?')}' "
                                "has no repo_url/commit")
                continue
            m = r.get("metrics") or {}
            if "recall" in m and "fnr" in m:
                if abs(m["fnr"] - (1 - m["recall"])) > 5e-4:
                    problems.append(f"{p}:{i} [G4] FNR != 1-Recall "
                                    f"({m['fnr']:.4f} vs {1 - m['recall']:.4f})")
                    continue
            rows.append(r)
    return rows, problems, inferred


def ms(v):
    a = np.asarray(v, dtype=float)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, a.size)


# ------------------------------------------------------------- statistics
def paired(a: dict, b: dict):
    """Paired comparison of b against a over the folds both hold."""
    folds = sorted(set(a) & set(b))
    if len(folds) < 2:
        return None
    da = np.array([b[f] - a[f] for f in folds], dtype=float)
    n = len(da)
    mean = float(da.mean())
    sd = float(da.std(ddof=1))
    out = {"n": n, "folds": folds, "mean": mean, "sd": sd,
           "pos": int((da > 0).sum()), "neg": int((da < 0).sum())}
    if sd > 0:
        t = mean / (sd / math.sqrt(n))
        out["t"] = float(t)
        out["p_t"] = (float(2 * (1 - sps.t.cdf(abs(t), n - 1))) if sps
                      else float("nan"))
    else:
        out["t"], out["p_t"] = float("inf") if mean else 0.0, 0.0 if mean else 1.0
    if sps is not None and n >= 2 and np.any(da != 0):
        try:
            out["p_w"] = float(sps.wilcoxon(da, zero_method="wilcox",
                                            alternative="two-sided",
                                            mode="exact").pvalue)
        except Exception:
            try:
                out["p_w"] = float(sps.wilcoxon(da).pvalue)
            except Exception:
                out["p_w"] = float("nan")
    else:
        out["p_w"] = float("nan")
    out["w_floor"] = 2.0 / (2 ** n)
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni across one family of comparisons."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)),
                   key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = adj
    return out


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--latex", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows, problems, inferred = load(args.paths)
    print(f"{VERSION}: {len(rows)} rows from {len(args.paths)} file(s)")
    if inferred:
        print(f"  {inferred} rows predate the device field; their device is "
              "marked cpu[INFERRED].\n  That driver had no CUDA path, so it is "
              "a fact about the code, not a guess --\n  but it is labelled, and "
              "it must stay labelled in the manuscript.")
    if problems:
        print(f"\n*** {len(problems)} rows dropped by a gate:")
        for p in problems[:15]:
            print(f"    {p}")
        if len(problems) > 15:
            print(f"    ... and {len(problems) - 15} more")
    if not rows:
        raise SystemExit("nothing left to analyse")

    devs = sorted({r["device"] for r in rows})
    fps = sorted({r["fingerprint"] for r in rows})
    print(f"devices: {devs}   distinct model fingerprints: {len(fps)}")
    if len(devs) > 1:
        print("  NOTE: rows from more than one device. Dropout and the "
              "generator draw on-device,\n        so models differ between "
              "devices; do not pool these into one mean.")

    specs_seen = sorted({spec_of(r) for r in rows})
    print(f"perturbable-set specs present: {len(specs_seen)}  "
          f"{specs_seen}")
    nofp = sum(1 for r in rows if not r.get("spec_fingerprint"))
    if nofp:
        print(f"  {nofp} row(s) predate the spec fingerprint. They are "
              "grouped under a pre16 label\n  and are NOT assumed to share a "
              "spec with any v16 run: the declared column list\n  lives in the "
              "driver and the row does not record enough to prove it is the "
              "same.")
    if len(specs_seen) > 1:
        print("  More than one spec is present, so every table below is "
              "reported per spec.\n  v3 would have collided these into one "
              "table with the later row overwriting\n  the earlier one. The "
              "cross-spec comparison is at the end.")

    latex: list[str] = []
    cross: dict = defaultdict(dict)       # (corpus, task) -> spec -> payload

    for (corpus, task, spec_id) in sorted(
            {(r["corpus"], r.get("task", "-"), spec_of(r)) for r in rows}):
        sub = [r for r in rows if r["corpus"] == corpus
               and r.get("task", "-") == task
               and spec_of(r) == spec_id]
        print("\n" + "=" * 74)
        print(f"  {corpus}   task={task}   spec={spec_id}   ({len(sub)} rows)")
        print(f"  {spec_descr(sub)}")
        print("=" * 74)

        # f1[arm][attack][eps][fold]
        f1 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        cw = defaultdict(dict)
        for r in sub:
            m = r.get("metrics") or {}
            if r["attack"] == "cw":
                if r.get("cw") and not r["cw"].get("skipped"):
                    cw[r["arm"]][r["fold"]] = r["cw"]
                continue
            if "f1" not in m:
                continue
            if r["attack"] == "clean":
                for atk in ("fgsm", "pgd"):
                    f1[r["arm"]][atk][0.0][r["fold"]] = m["f1"]
            else:
                f1[r["arm"]][r["attack"]][float(r["eps"])][r["fold"]] = m["f1"]

        arms = sorted(f1)

        # ------------------------------------------- clean detection (C1)
        # v4 and earlier tested retention only. The augmentation claim is
        # about DETECTION, not robustness: does training on generated rows
        # raise macro-F1 on unperturbed traffic? That is a paired test on the
        # clean column and it was never printed, so the clean means were read
        # off the attack tables and compared by eye. This prints the test.
        clean = {a: dict(f1[a]["pgd"].get(0.0, {})) for a in arms}
        clean = {a: v for a, v in clean.items() if v}
        if clean:
            print("\n[clean]  macro-F1 on unperturbed traffic"
                  "  -- what the augmentation claim is about")
            for a in sorted(clean):
                mu, sd, n = ms(list(clean[a].values()))
                print(f"    {a}  {ARM_NOTE.get(a, ''):<20} "
                      f"{mu:.4f} +/- {sd:.4f}  (n={n})")
            famc = {}
            for a, b in combinations(sorted(clean), 2):
                r = paired(clean[a], clean[b])
                if r:
                    famc[(a, b)] = r
            if famc:
                adjc = holm({k: v["p_t"] for k, v in famc.items()})
                print("\n  paired comparisons on clean macro-F1"
                      "  (positive = second arm detects better)")
                for (a, b), r in famc.items():
                    mark = "  *" if adjc.get((a, b), 1) < args.alpha else ""
                    print(f"    {a} -> {b}:  {r['mean']:+.4f} +/- "
                          f"{r['sd']:.4f}   {r['pos']}/{r['n']} folds "
                          f"positive   t={r['t']:+.2f} p={r['p_t']:.3f} "
                          f"(Holm {adjc.get((a, b), float('nan')):.3f})   "
                          f"W p={r['p_w']:.3f} "
                          f"[floor {r['w_floor']:.4f}]{mark}")
                print("  D -> A is the augmentation contrast: D trains on real"
                      " rows only, A adds\n  generated ones and changes "
                      "nothing else. B -> C is the same contrast with\n"
                      "  adversarial training already present in both.")

        for atk in ("fgsm", "pgd"):
            eps_all = sorted({e for a in arms for e in f1[a][atk]})
            if len(eps_all) < 2:
                continue
            print(f"\n[{atk}]  macro-F1 mean +/- sd (n folds), and retention")
            hdr = "  arm  " + "".join(f"{e:>16g}" for e in eps_all)
            print(hdr)
            ret = {}
            for a in arms:
                cells, rr = [], {}
                for e in eps_all:
                    v = f1[a][atk].get(e, {})
                    if not v:
                        cells.append(f"{'--':>16}")
                        continue
                    mu, sd, n = ms(list(v.values()))
                    cells.append(f"{mu:>9.4f}+-{sd:<5.4f}")
                base = f1[a][atk].get(0.0, {})
                for e in eps_all:
                    v = f1[a][atk].get(e, {})
                    rr[e] = {fd: v[fd] / base[fd] for fd in v
                             if fd in base and base[fd]}
                ret[a] = rr
                print(f"  {a:<5}" + "".join(cells))

            e_top = eps_all[-1]
            print(f"\n  retention at eps={e_top:g}")
            for a in arms:
                mu, sd, n = ms(list(ret[a][e_top].values())) if ret[a][e_top] else (float('nan'),) * 3
                print(f"    {a}  {ARM_NOTE.get(a, ''):<20} {mu:.4f} +/- {sd:.4f}  (n={n})")

            # per-attack non-monotonicity is a NOTE, not a gate failure
            for a in arms:
                pts = [(e, ms(list(f1[a][atk][e].values()))[0])
                       for e in eps_all if f1[a][atk][e]]
                rises = [f"{e0:g}->{e1:g}" for (e0, v0), (e1, v1)
                         in zip(pts, pts[1:]) if v1 > v0 + 0.005]
                if rises:
                    kind = ("expected for a single-step attack"
                            if atk == "fgsm" else
                            "NOT expected for an iterative attack -- check it")
                    print(f"    note: arm {a} rose at {', '.join(rises)}  ({kind})")

            print(f"\n  paired comparisons on retention at eps={e_top:g}"
                  "  (positive = second arm retains more)")
            fam = {}
            for a, b in combinations(arms, 2):
                r = paired(ret[a][e_top], ret[b][e_top])
                if r:
                    fam[(a, b)] = r
            adj = holm({k: v["p_t"] for k, v in fam.items()})
            for (a, b), r in fam.items():
                mark = "  *" if adj.get((a, b), 1) < args.alpha else ""
                print(f"    {a} -> {b}:  {r['mean']:+.4f} +/- {r['sd']:.4f}   "
                      f"{r['pos']}/{r['n']} folds positive   "
                      f"t={r['t']:+.2f} p={r['p_t']:.3f} "
                      f"(Holm {adj.get((a, b), float('nan')):.3f})   "
                      f"W p={r['p_w']:.3f} [floor {r['w_floor']:.4f}]{mark}")

        # ------------------------------------------- worst-case envelope + G5
        atks = [a for a in ("fgsm", "pgd") if any(f1[x][a] for x in arms)]
        if atks:
            eps_all = sorted({e for a in arms for t in atks for e in f1[a][t]})
            print("\n[worst case over all attacks at this budget or below]"
                  "  -- what G5 is asserted on")
            print("  arm  " + "".join(f"{e:>12g}" for e in eps_all)
                  + f"{'retention':>12}")
            env = {}
            for a in arms:
                per_fold = defaultdict(dict)        # fold -> eps -> worst f1
                for e in eps_all:
                    for t in atks:
                        for fd, v in f1[a][t].get(e, {}).items():
                            cur = per_fold[fd].get(e)
                            per_fold[fd][e] = v if cur is None else min(cur, v)
                # make it an envelope: worst at this budget OR BELOW
                for fd in per_fold:
                    run = None
                    for e in eps_all:
                        if e in per_fold[fd]:
                            run = (per_fold[fd][e] if run is None
                                   else min(run, per_fold[fd][e]))
                            per_fold[fd][e] = run
                env[a] = {e: {fd: per_fold[fd][e] for fd in per_fold
                              if e in per_fold[fd]} for e in eps_all}
                cells = []
                for e in eps_all:
                    v = env[a][e]
                    cells.append(f"{ms(list(v.values()))[0]:>12.4f}" if v
                                 else f"{'--':>12}")
                base, top = env[a].get(0.0, {}), env[a].get(eps_all[-1], {})
                rv = [top[fd] / base[fd] for fd in top
                      if fd in base and base[fd]]
                mu = ms(rv)[0] if rv else float("nan")
                print(f"  {a:<5}" + "".join(cells) + f"{mu:>12.4f}")

                pts = [(e, ms(list(env[a][e].values()))[0]) for e in eps_all
                       if env[a][e]]
                bad = [f"{e0:g}->{e1:g}" for (e0, v0), (e1, v1)
                       in zip(pts, pts[1:]) if v1 > v0 + 1e-9]
                if bad:
                    print(f"    [G5] arm {a}: the ENVELOPE rose at "
                          f"{', '.join(bad)} -- impossible by construction, "
                          "so this is a defect in the envelope code")

            e_top = eps_all[-1]
            ret_env = {}
            for a in arms:
                base, top = env[a].get(0.0, {}), env[a].get(e_top, {})
                ret_env[a] = {fd: top[fd] / base[fd] for fd in top
                              if fd in base and base[fd]}

            # kept for the cross-spec section: the same quantity, per spec
            cross[(corpus, task)][spec_id] = {
                "e_top": e_top,
                "ret": {a: dict(ret_env[a]) for a in arms},
                "clean": {a: dict(env[a].get(0.0, {})) for a in arms},
                "descr": spec_descr(sub),
            }

            print(f"\n  paired comparisons on ENVELOPE retention at "
                  f"eps={e_top:g}  (this is the headline number)")
            fam = {}
            for a, b in combinations(arms, 2):
                r = paired(ret_env[a], ret_env[b])
                if r:
                    fam[(a, b)] = r
            adjE = holm({k: v["p_t"] for k, v in fam.items()})
            for (a, b), r in fam.items():
                mark = "  *" if adjE.get((a, b), 1) < args.alpha else ""
                print(f"    {a} -> {b}:  {r['mean']:+.4f} +/- {r['sd']:.4f}   "
                      f"{r['pos']}/{r['n']} folds positive   "
                      f"t={r['t']:+.2f} p={r['p_t']:.3f} "
                      f"(Holm {adjE.get((a, b), float('nan')):.3f})   "
                      f"W p={r['p_w']:.3f} [floor {r['w_floor']:.4f}]{mark}")

        # ------------------------------------------------------------- CW
        if cw:
            print("\n[carlini-wagner]  success rate, and fraction flippable "
                  "within an L2 budget")
            grid = sorted({g for a in cw for f in cw[a]
                           for g in (cw[a][f].get("success_at_l2") or {})},
                          key=float)
            print("  arm      success" + "".join(f"{('L2<=' + g):>12}" for g in grid))
            for a in sorted(cw):
                sr = ms([cw[a][f]["success_rate"] for f in cw[a]])
                cells = []
                for g in grid:
                    vals = [cw[a][f]["success_at_l2"][g] for f in cw[a]
                            if g in (cw[a][f].get("success_at_l2") or {})]
                    cells.append(f"{ms(vals)[0]:>12.3f}" if vals else f"{'--':>12}")
                print(f"  {a:<5}{sr[0]:>8.3f}" + "".join(cells))
            print("  mean_l2_successful is conditioned on success and is NOT "
                  "comparable across arms;")
            print("  it is deliberately absent from this table.")

            print("\n  paired comparisons on CW success (negative = second arm "
                  "harder to attack)")
            fam = {}
            for a, b in combinations(sorted(cw), 2):
                r = paired({f: cw[a][f]["success_rate"] for f in cw[a]},
                           {f: cw[b][f]["success_rate"] for f in cw[b]})
                if r:
                    fam[(a, b)] = r
            adj = holm({k: v["p_t"] for k, v in fam.items()})
            for (a, b), r in fam.items():
                mark = "  *" if adj.get((a, b), 1) < args.alpha else ""
                print(f"    {a} -> {b}:  {r['mean']:+.4f} +/- {r['sd']:.4f}   "
                      f"{r['neg']}/{r['n']} folds in {b}'s favour   "
                      f"t={r['t']:+.2f} p={r['p_t']:.3f} "
                      f"(Holm {adj.get((a, b), float('nan')):.3f}){mark}")

        # ------------------------------------------------------------ latex
        if args.latex:
            latex.append(r"\begin{table}[t]\centering")
            latex.append(rf"\caption{{Macro-F1 under PGD, {corpus} ({task}), "
                         rf"perturbable-set spec \texttt{{{spec_id}}}. "
                         r"Mean $\pm$ s.d.\ over folds; retention is "
                         r"$F_1(\varepsilon)/F_1(0)$ per fold.}}")
            eps_all = sorted({e for a in arms for e in f1[a]["pgd"]})
            latex.append(r"\begin{tabular}{l" + "r" * (len(eps_all) + 1) + "}")
            latex.append(r"\hline")
            latex.append("Arm & " + " & ".join(rf"$\varepsilon={e:g}$"
                                               for e in eps_all)
                         + r" & Retention \\")
            latex.append(r"\hline")
            for a in arms:
                cells = []
                for e in eps_all:
                    v = f1[a]["pgd"].get(e, {})
                    cells.append(f"${ms(list(v.values()))[0]:.4f}$" if v else "--")
                base = f1[a]["pgd"].get(0.0, {})
                top = f1[a]["pgd"].get(eps_all[-1], {})
                rv = [top[fd] / base[fd] for fd in top if fd in base and base[fd]]
                mu, sd, _ = ms(rv) if rv else (float("nan"), 0.0, 0)
                latex.append(f"{a} ({ARM_NOTE.get(a, '')}) & " + " & ".join(cells)
                             + rf" & ${mu:.4f} \pm {sd:.4f}$ \\")
            latex.append(r"\hline\end{tabular}\end{table}")

    # --------------------------------------------------- cross-spec (G9)
    multi = {k: v for k, v in cross.items() if len(v) > 1}
    if multi:
        print("\n" + "#" * 74)
        print("  WHAT THE PERTURBABLE SET IS WORTH")
        print("#" * 74)
        print("  Redeclaring which columns an adversary may move changes the")
        print("  attacks and nothing else, so the clean models are the same and")
        print("  envelope retention is directly comparable per arm and fold.")
        print("  Positive below = the second spec retains MORE, i.e. the first")
        print("  spec made the model look less robust than the second did.")
    for (corpus, task), byspec in sorted(multi.items()):
        print(f"\n  {corpus}  task={task}")
        for s in sorted(byspec):
            print(f"    {s:<18} {byspec[s]['descr']}")
        for sa, sb in combinations(sorted(byspec), 2):
            A, B = byspec[sa], byspec[sb]
            if A["e_top"] != B["e_top"]:
                print(f"    REFUSED {sa} vs {sb}: top budget differs "
                      f"({A['e_top']:g} vs {B['e_top']:g}); the envelopes are "
                      "not at the same budget.")
                continue
            arms_c = sorted(set(A["ret"]) & set(B["ret"]))
            if not arms_c:
                print(f"    {sa} vs {sb}: no arm measured under both.")
                continue
            print(f"\n    {sa} -> {sb}  at eps={A['e_top']:g}, "
                  f"{len(arms_c)} arm(s) in common")
            fam = {}
            for a in arms_c:
                # G9: the clean models must agree, or the difference is not
                # attributable to the spec alone.
                ca, cb = A["clean"].get(a, {}), B["clean"].get(a, {})
                shared = sorted(set(ca) & set(cb))
                dc = [abs(ca[f] - cb[f]) for f in shared]
                worst = max(dc) if dc else float("nan")
                r = paired(A["ret"][a], B["ret"][a])
                if not r:
                    print(f"      arm {a}: fewer than 2 shared folds, skipped")
                    continue
                fam[a] = r
                flag = ""
                if dc and worst > 0.02:
                    flag = (f"   [G9] clean F1 differs by up to {worst:.4f} "
                            "-- not the spec alone")
                elif dc:
                    flag = f"   (clean agrees to {worst:.4f})"
                mu_a = ms(list(A["ret"][a].values()))[0] if A["ret"][a] else float("nan")
                mu_b = ms(list(B["ret"][a].values()))[0] if B["ret"][a] else float("nan")
                print(f"      arm {a} {ARM_NOTE.get(a, ''):<18} "
                      f"{mu_a:.4f} -> {mu_b:.4f}   "
                      f"{r['mean']:+.4f} +/- {r['sd']:.4f}   "
                      f"{r['pos']}/{r['n']} folds positive   "
                      f"t={r['t']:+.2f} p={r['p_t']:.3f}{flag}")
            if fam:
                adjS = holm({k: v["p_t"] for k, v in fam.items()})
                print(f"      Holm across {len(fam)} arm(s): " + ", ".join(
                    f"{a} {adjS.get(a, float('nan')):.3f}"
                    + ("*" if adjS.get(a, 1) < args.alpha else "")
                    for a in sorted(fam)))
                biggest = max(fam.values(), key=lambda r: abs(r["mean"]))
                print(f"      largest shift in retention from the spec alone: "
                      f"{biggest['mean']:+.4f}")
                print("      A paper that reports one spec and does not say "
                      "which columns it froze")
                print("      is reporting a number with this much slack in it.")

    if args.latex:
        Path(args.latex).write_text("\n".join(latex) + "\n", encoding="utf-8")
        print(f"\nLaTeX written to {args.latex}")

    if len(devs) > 1 or True:
        print("\nReproducibility: PyTorch on CUDA is not deterministic "
              "run-to-run by default\n(atomics in reductions). The same config "
              "re-run gave clean macro-F1 0.6239\nand 0.6132 on one CICIDS "
              "fold. Report mean +/- sd over folds, record the\ndevice, and do "
              "not claim bit-exact reproducibility without\n"
              "torch.use_deterministic_algorithms(True).")

    print("\nReminder: with n=5 folds the two-sided exact Wilcoxon floor is "
          "0.0625, so it\ncannot reject at 0.05 whatever the data. Report the "
          "paired t, state the floor,\nand do not call 0.0625 a near miss.")


if __name__ == "__main__":
    main()
