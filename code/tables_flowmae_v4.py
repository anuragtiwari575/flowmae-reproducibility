#!/usr/bin/env python3
"""
tables_flowmae_v4.py -- generate tab_perfold.tex from the result records.

CHANGE FROM v3
--------------
The four paired-contrast tables repeated one identical column
definition four times, which cost a page of the manuscript and read
as boilerplate. The first table now carries the explanation and the
other three say "Columns as in Table N". The numbers are unchanged.

CHANGE FROM v1
--------------
The per-fold table carried a leading "Corpus" column that was empty in
every data row, because the corpus is already carried by the
\\multicolumn section headers. The section headers therefore spanned 7
of the 8 columns, and the table was one column wider than it needed to
be. That column is dropped here.

The span is now folds+2, which is the full width of the narrower
table: one column for the arm, one per fold, one for the mean. (v2
dropped the column but also changed the span to folds+1, which left
the same off-by-one in a seven-column table. v2 should not be used.)

Nothing else changes: the same records are read and every number is
identical to v1.

WHY THIS EXISTS
---------------
The review asked for per-fold values, confidence intervals and effect
sizes rather than means and p-values alone, and for every aggregate in
Section 5 to be recomputable from fold-level data. Typing those by hand
is how a table drifts from the records it claims to summarise, so this
reads out/*.jsonl and writes the LaTeX.

WHAT IT WRITES
--------------
  tab_clean.tex     clean macro-F1 by arm and corpus, mean +/- sd,
                    with the headline paired contrasts
  tab_envelope.tex  envelope retention at eps=0.2 and the absolute
                    envelope macro-F1 it is computed from
  tab_perfold.tex   per-fold values for every arm, followed by the
                    paired contrasts with mean difference, 95% CI,
                    Cohen's d_z, fold-wise sign count, paired t, exact
                    Wilcoxon and Holm-corrected p

Every number in Sections 5.2 and 5.3 of the manuscript comes from
these three files. Nothing is typed by hand, because a hand-typed
dispersion is a number with no record behind it.

main.tex \\input{}s it if present, so running this fills Appendix B.

ARM LABELLING
-------------
Three conditions share the arm name FB on disk -- plain, per-column
dropout, gradient dispersion -- and are told apart only by which file
they are in. The MANIFEST here is the same one the figure script uses,
for the same reason.

USAGE
  python -u tables_flowmae_v4.py --out tab_perfold.tex
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

MANIFEST = {
    "UNSW-NB15": [
        ("out/phase13_unsw_permissive.jsonl", {"B": "B", "C": "C"}),
        ("out/flowmae_unsw_plain.jsonl",      {"F": "F", "FB": "FB"}),
        ("out/flowmae_unsw_mcp_v2.jsonl",     {"FM": "FM", "FMB": "FMB"}),
        ("out/flowmae_unsw_cd02.jsonl",       {"FB": "FB+CD"}),
        ("out/flowmae_unsw_gdr05.jsonl",      {"FB": "FB+GDR"}),
    ],
    "CICIDS-2017": [
        ("out/phase13_cicids_m8_permissive.jsonl", {"B": "B", "C": "C"}),
        ("out/flowmae_cicids_m8.jsonl",            {"FB": "FB"}),
        ("out/flowmae_cicids_mcp.jsonl",           {"FMB": "FMB"}),
        ("out/flowmae_cicids_plain.jsonl",         {"F": "F"}),
        ("out/flowmae_cicids_fm.jsonl",            {"FM": "FM"}),
    ],
}
ORDER = ["B", "C", "F", "FB", "FM", "FMB", "FB+CD", "FB+GDR"]
EPS = [0.0, 0.01, 0.05, 0.1, 0.2]
# contrasts reported in the paper, in the order Holm is applied over
FAMILY = [("B", "C"), ("B", "F"), ("B", "FB"), ("B", "FM"), ("B", "FMB"),
          ("C", "FMB"), ("F", "FB"), ("FM", "FMB"), ("FB", "FMB")]


def load():
    data = {}
    for corpus, entries in MANIFEST.items():
        rows = []
        for path, remap in entries:
            q = Path(path)
            if not q.exists():
                print(f"  missing, skipped: {path}")
                continue
            n = 0
            for line in q.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("arm") in remap:
                    r = dict(r)
                    r["label"] = remap[r["arm"]]
                    rows.append(r)
                    n += 1
            print(f"  {path}: {n} row(s)")
        if rows:
            data[corpus] = rows
    return data


def f1_of(r):
    return (r.get("metrics") or {}).get("f1")


def cell(rows, lab, attack, eps):
    out = {}
    for r in rows:
        if r["label"] != lab or r.get("attack") != attack:
            continue
        if abs(float(r.get("eps", -99)) - eps) > 1e-9:
            continue
        v = f1_of(r)
        if v is not None:
            out[int(r["fold"])] = float(v)
    return out


def envelope(rows, lab):
    """Worst case over fgsm and pgd at eps or below, per fold."""
    out = {}
    folds = sorted({int(r["fold"]) for r in rows if r["label"] == lab})
    for f in folds:
        run, best = None, {}
        for e in EPS:
            vals = []
            if e == 0.0:
                v = cell(rows, lab, "clean", 0.0).get(f)
                if v is not None:
                    vals.append(v)
            for atk in ("fgsm", "pgd"):
                v = cell(rows, lab, atk, e).get(f)
                if v is not None:
                    vals.append(v)
            if vals:
                run = min(vals) if run is None else min(run, min(vals))
                best[e] = run
        out[f] = best
    return out


def paired(a: dict, b: dict):
    """b minus a on the folds both have."""
    folds = sorted(set(a) & set(b))
    if len(folds) < 2:
        return None
    d = np.array([b[f] - a[f] for f in folds], dtype=float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    tcrit = stats.t.ppf(0.975, n - 1)
    lo, hi = m - tcrit * se, m + tcrit * se
    dz = m / sd if sd > 0 else np.inf
    t, pt = stats.ttest_rel([b[f] for f in folds], [a[f] for f in folds])
    try:
        pw = stats.wilcoxon([b[f] - a[f] for f in folds],
                            alternative="two-sided",
                            zero_method="wilcox", mode="exact").pvalue
    except Exception:                                   # noqa: BLE001
        pw = float("nan")
    return dict(n=n, folds=folds, d=d, mean=m, sd=sd, lo=lo, hi=hi,
                dz=dz, t=float(t), p=float(pt), pw=float(pw),
                pos=int((d > 0).sum()))


def holm(pvals):
    idx = np.argsort(pvals)
    k = len(pvals)
    adj = np.empty(k)
    running = 0.0
    for rank, i in enumerate(idx):
        v = (k - rank) * pvals[i]
        running = max(running, v)
        adj[i] = min(running, 1.0)
    return adj


def esc(x):
    return f"{x:.4f}" if np.isfinite(x) else "--"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tab_perfold.tex")
    ap.add_argument("--clean-out", default="tab_clean.tex")
    ap.add_argument("--envelope-out", default="tab_envelope.tex")
    ap.add_argument("--cw-out", default="tab_cw.tex")
    ap.add_argument("--csv", default="out/master_results.csv",
                    help="every value behind every table, one row per "
                         "(corpus, arm, quantity, fold), for the final "
                         "numerical audit")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    print("reading:")
    data = load()
    if not data:
        raise SystemExit("no result records found; run from supplementary\\")

    # ---------------- Table: clean macro-F1 -------------------------
    CL = []
    CL.append(r"\begin{table}[htbp]")
    CL.append(r"\centering")
    CL.append(r"\caption{Clean \macroF{} on unperturbed traffic, mean "
              r"$\pm$ standard deviation over five folds. A dash marks an "
              r"arm not run on that corpus.}")
    CL.append(r"\label{tab:clean}")
    CL.append(r"\small")
    CL.append(r"\begin{tabular}{@{}lcc@{}}")
    CL.append(r"\toprule")
    CL.append(r"\textbf{Arm} & \textbf{UNSW-NB15} & \textbf{CICIDS-2017} \\")
    CL.append(r"\midrule")
    corpora = list(data)
    all_labels = [x for x in ORDER
                  if any(x in {r["label"] for r in data[c]} for c in corpora)]
    for lab in all_labels:
        cells = []
        for c in ("UNSW-NB15", "CICIDS-2017"):
            if c not in data:
                cells.append("--"); continue
            d = cell(data[c], lab, "clean", 0.0)
            if not d:
                cells.append("--"); continue
            v = np.array(list(d.values()), float)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            cells.append(f"${v.mean():.4f} \\pm {sd:.4f}$")
        CL.append(f"{lab} & " + " & ".join(cells) + r" \\")
    CL.append(r"\bottomrule")
    CL.append(r"\end{tabular}")
    CL.append(r"\end{table}")
    Path(a.clean_out).write_text("\n".join(CL) + "\n", encoding="utf-8")
    print(f"wrote {a.clean_out}")

    # ---------------- Table: envelope retention ---------------------
    EV = []
    EV.append(r"\begin{table}[htbp]")
    EV.append(r"\centering")
    EV.append(r"\caption{Retention of the empirical attack-suite envelope "
              r"at $\epsb = 0.2$, with the absolute envelope \macroF{} it "
              r"is computed from. Retention is $F_{0.2}/F_0$ per fold, then "
              r"averaged; the absolute column is the mean over folds of the "
              r"envelope itself.}")
    EV.append(r"\label{tab:envelope}")
    EV.append(r"\small")
    EV.append(r"\begin{tabular}{@{}lcccc@{}}")
    EV.append(r"\toprule")
    EV.append(r"& \multicolumn{2}{c}{\textbf{UNSW-NB15}} & "
              r"\multicolumn{2}{c}{\textbf{CICIDS-2017}} \\")
    EV.append(r"\cmidrule(r){2-3}\cmidrule(l){4-5}")
    EV.append(r"\textbf{Arm} & \textbf{retention} & "
              r"\textbf{absolute $F_{0.2}$} & \textbf{retention} & "
              r"\textbf{absolute $F_{0.2}$} \\")
    EV.append(r"\midrule")
    for lab in all_labels:
        cells = []
        for c in ("UNSW-NB15", "CICIDS-2017"):
            if c not in data:
                cells += ["--", "--"]; continue
            env = envelope(data[c], lab)
            cl = cell(data[c], lab, "clean", 0.0)
            ab, re_ = [], []
            for f, best in env.items():
                v = best.get(0.2)
                if v is None:
                    continue
                ab.append(v)
                if cl.get(f):
                    re_.append(v / cl[f])
            if not ab:
                cells += ["--", "--"]; continue
            av = np.array(ab, float); rv = np.array(re_, float)
            rsd = rv.std(ddof=1) if len(rv) > 1 else 0.0
            cells.append(f"${rv.mean():.4f} \\pm {rsd:.4f}$"
                         if len(rv) else "--")
            cells.append(f"${av.mean():.4f}$")
        EV.append(f"{lab} & " + " & ".join(cells) + r" \\")
    EV.append(r"\bottomrule")
    EV.append(r"\end{tabular}")
    EV.append(r"\end{table}")
    Path(a.envelope_out).write_text("\n".join(EV) + "\n", encoding="utf-8")
    print(f"wrote {a.envelope_out}")

    # ---------------- Table: Carlini-Wagner -------------------------
    L2S = ["0.02", "0.05", "0.1", "0.2", "0.4", "0.8"]
    CW = []
    CW.append(r"\begin{table}[htbp]")
    CW.append(r"\centering")
    CW.append(r"\caption{Carlini--Wagner: overall success rate and the "
              r"fraction of correctly classified rows flippable within "
              r"$L_2 \le 0.02$, mean over five folds, 1{,}000 rows per "
              r"fold. The small-radius column is the diagnostic one. "
              r"A row it fails to flip is not shown to be unflippable.}")
    CW.append(r"\label{tab:cw}")
    CW.append(r"\small")
    CW.append(r"\begin{tabular}{@{}lcccc@{}}")
    CW.append(r"\toprule")
    CW.append(r"& \multicolumn{2}{c}{\textbf{UNSW-NB15}} & "
              r"\multicolumn{2}{c}{\textbf{CICIDS-2017}} \\")
    CW.append(r"\cmidrule(r){2-3}\cmidrule(l){4-5}")
    CW.append(r"\textbf{Arm} & \textbf{success} & "
              r"$\boldsymbol{L_2 \le 0.02}$ & \textbf{success} & "
              r"$\boldsymbol{L_2 \le 0.02}$ \\")
    CW.append(r"\midrule")
    for lab in all_labels:
        cells = []
        for c in ("UNSW-NB15", "CICIDS-2017"):
            if c not in data:
                cells += ["--", "--"]; continue
            succ, small = [], []
            for r in data[c]:
                if r["label"] != lab or r.get("attack") != "cw":
                    continue
                cwd = r.get("cw") or {}
                if "success_rate" in cwd:
                    succ.append(float(cwd["success_rate"]))
                sa = cwd.get("success_at_l2") or {}
                if "0.02" in sa:
                    small.append(float(sa["0.02"]))
            cells.append(f"${np.mean(succ):.3f}$" if succ else "--")
            cells.append(f"${np.mean(small):.3f}$" if small else "--")
        CW.append(f"{lab} & " + " & ".join(cells) + r" \\")
    CW.append(r"\bottomrule")
    CW.append(r"\end{tabular}")
    CW.append(r"\end{table}")
    Path(a.cw_out).write_text("\n".join(CW) + "\n", encoding="utf-8")
    print(f"wrote {a.cw_out}")

    L = []
    W = L.append

    # ---------------- per-fold values -------------------------------
    W(r"\begin{table}[htbp]")
    W(r"\centering")
    W(r"\caption{Per-fold clean \macroF{} and worst-case \macroF{} at "
      r"$\epsb = 0.2$, read directly from the result records. Every "
      r"aggregate in \Cref{sec:results} is recomputable from these "
      r"values.}")
    W(r"\label{tab:perfold}")
    W(r"\small")
    # One column per quantity shown, and no leading corpus column: the
    # corpus is carried by the \multicolumn section headers, so a
    # separate column for it is dead width in a single-column layout
    # and leaves the section headers spanning 7 of 8 columns.
    hdr = (r"\begin{tabular}{@{}l" + "c" * a.folds + r"c@{}}")
    W(hdr)
    W(r"\toprule")
    W(r"\textbf{Arm} & "
      + " & ".join(rf"\textbf{{fold {i}}}" for i in range(a.folds))
      + r" & \textbf{mean} \\")
    for corpus, rows in data.items():
        for quantity, getter in (("clean", lambda rw, l: cell(rw, l, "clean", 0.0)),
                                 (r"worst case, $\epsb=0.2$",
                                  lambda rw, l: {f: envelope(rw, l)[f].get(0.2)
                                                 for f in envelope(rw, l)})):
            W(r"\midrule")
            W(rf"\multicolumn{{{a.folds + 2}}}{{@{{}}l}}"
              rf"{{\emph{{{corpus} -- {quantity}}}}} \\")
            labs = [x for x in ORDER
                    if x in {r["label"] for r in rows}]
            for lab in labs:
                d = getter(rows, lab)
                vals = [d.get(i) for i in range(a.folds)]
                cells = " & ".join("--" if v is None else f"{v:.4f}"
                                   for v in vals)
                got = [v for v in vals if v is not None]
                mean = f"{np.mean(got):.4f}" if got else "--"
                W(rf"{lab} & {cells} & {mean} \\")
    W(r"\bottomrule")
    W(r"\end{tabular}")
    W(r"\end{table}")
    W("")

    # ---------------- contrasts -------------------------------------
    # Four contrast tables share one column definition. Repeating it
    # four times cost a page and reads as boilerplate, so the first
    # table carries the explanation and the rest point at it.
    first_contrast = [None]

    for corpus, rows in data.items():
        present = {r["label"] for r in rows}
        for quantity, getter, tag in (
                ("clean \\macroF{}",
                 lambda rw, l: cell(rw, l, "clean", 0.0), "clean"),
                ("worst-case \\macroF{} at $\\epsb = 0.2$",
                 lambda rw, l: {f: envelope(rw, l)[f].get(0.2)
                                for f in envelope(rw, l)}, "worst")):
            pairs, res = [], []
            for x, y in FAMILY:
                if x not in present or y not in present:
                    continue
                A = {k: v for k, v in getter(rows, x).items() if v is not None}
                B = {k: v for k, v in getter(rows, y).items() if v is not None}
                r = paired(A, B)
                if r:
                    pairs.append((x, y))
                    res.append(r)
            if not res:
                continue
            adj = holm([r["p"] for r in res])
            lab = f"tab:contrast-{tag}-{corpus.split('-')[0].lower()}"
            W(r"\begin{table}[htbp]")
            W(r"\centering")
            if first_contrast[0] is None:
                first_contrast[0] = lab
                W(rf"\caption{{Paired contrasts on {quantity}, "
                  rf"{corpus}. "
                  r"$\Delta$ is the mean paired difference (second arm "
                  r"minus first), CI its 95\% $t$-interval, $d_z$ the "
                  r"standardised paired effect size, and ``folds'' the "
                  r"number with a positive difference. Holm correction "
                  r"is applied within this family. The exact Wilcoxon "
                  r"floor at $n=5$ is $0.0625$, so that test cannot "
                  r"reject at $0.05$ for any data; the folds are not "
                  r"independent replications, and these tests are "
                  r"descriptive summaries of fold-level consistency "
                  r"rather than population-level evidence "
                  r"(\Cref{sec:stats}). The three tables that follow "
                  r"use the same columns.}")
            else:
                W(rf"\caption{{Paired contrasts on {quantity}, "
                  rf"{corpus}. Columns as in "
                  rf"\Cref{{{first_contrast[0]}}}.}}")
            W(rf"\label{{{lab}}}")
            W(r"\small")
            W(r"\begin{tabular}{@{}lcccccccc@{}}")
            W(r"\toprule")
            W(r"\textbf{Contrast} & $n$ & $\Delta$ & \textbf{95\% CI} & "
              r"$d_z$ & \textbf{folds} & $t$ & $p$ & \textbf{Holm} \\")
            W(r"\midrule")
            for (x, y), r, pa in zip(pairs, res, adj):
                star = r"\sig" if pa < 0.05 else ""
                W(rf"{x} $\to$ {y} & {r['n']} & ${r['mean']:+.4f}$ & "
                  rf"$[{r['lo']:+.4f}, {r['hi']:+.4f}]$ & "
                  rf"${r['dz']:+.2f}$ & {r['pos']}/{r['n']} & "
                  rf"${r['t']:+.2f}$ & ${r['p']:.3f}$ & "
                  rf"${pa:.3f}${star} \\")
            W(r"\bottomrule")
            W(r"\end{tabular}")
            W(r"\end{table}")
            W("")

    # ---------------- master results sheet --------------------------
    import csv as _csv
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["corpus", "arm", "quantity", "attack", "eps",
                    "fold", "value"])
        for c, rows in data.items():
            for lab in [x for x in ORDER
                        if x in {r["label"] for r in rows}]:
                for f, v in cell(rows, lab, "clean", 0.0).items():
                    w.writerow([c, lab, "clean_macro_f1", "clean", 0.0, f, v])
                for atk in ("fgsm", "pgd"):
                    for e in EPS[1:]:
                        for f, v in cell(rows, lab, atk, e).items():
                            w.writerow([c, lab, "macro_f1", atk, e, f, v])
                env = envelope(rows, lab)
                for f, best in env.items():
                    for e, v in best.items():
                        w.writerow([c, lab, "envelope_macro_f1",
                                    "envelope", e, f, v])
                for r in rows:
                    if r["label"] != lab or r.get("attack") != "cw":
                        continue
                    cwd = r.get("cw") or {}
                    if "success_rate" in cwd:
                        w.writerow([c, lab, "cw_success_rate", "cw", -1.0,
                                    r["fold"], cwd["success_rate"]])
                    for k, v in (cwd.get("success_at_l2") or {}).items():
                        w.writerow([c, lab, f"cw_flippable_L2<={k}", "cw",
                                    -1.0, r["fold"], v])
    print(f"wrote {a.csv}")

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out} ({len(L)} lines)")
    print("main.tex picks it up automatically on the next compile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
