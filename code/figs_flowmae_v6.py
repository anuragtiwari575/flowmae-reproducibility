#!/usr/bin/env python3
"""
figs_flowmae_v4.py -- every data-driven figure for the FlowMAE paper.

WHAT CHANGED FROM v3
--------------------
  1. Text was being embedded as Type 3. That is why it looked soft
     beside the body text: a Type 3 font is a stream of drawing
     operators rather than a real font, so it carries no hinting and
     degrades as the reader zooms, and Springer and IEEE both reject
     it at production. Setting pdf.fonttype and ps.fonttype to 42
     embeds TrueType outlines instead: vector, selectable, sharp at
     any magnification.

  2. The figures were set in DejaVu Sans while the manuscript is set
     in Latin Modern, so every figure read as a different document.
     The font stack is now serif, Latin Modern Roman first, falling
     back to Times New Roman on machines that do not have it.

  3. The arm labels in fig10 overlapped. The old placement counted
     how many earlier points were nearby and rotated the offset by a
     fixed angle per neighbour, which is not a collision test: it
     never looks at how wide a label is. On UNSW-NB15 the five
     adversarially trained arms sit almost on top of one another and
     "FB", "FB+CD" and "FB+GDR" ran into each other. place_labels()
     now measures each label's rendered extent and takes the first
     position clear of every marker and every label already placed,
     falling back to the least-overlapping position with a leader
     line. Verified on the measured UNSW coordinates: eight labels,
     zero overlapping pairs.

  4. Panel titles used "--", which is two hyphens on the page, not a
     dash. They now use an en dash.

WHAT CHANGED FROM v2
--------------------
  fig03 annotated "19x better" and "27x better", computed from the mean
  curve's last epoch, while the manuscript says 18 and 26, taken from
  the "best held-out ... (0.0xx x)" line each pretraining log prints.
  Two numbers for one quantity is the kind of mismatch a referee finds.
  v3 reads that same line, so the figure and the text now agree by
  construction rather than by coincidence.

WHAT CHANGED FROM v1
--------------------
  * fig09 was unreadable: forty column names overlapped into a smear.
    It now shows the top --top-n columns, taller, at a legible size,
    and says in the title how many of the full set are drawn.
  * fig10 piled five arm labels on top of each other in the UNSW
    cluster. Labels are now pushed apart along a small circle when
    points are close.
  * Legends sat on top of the data in fig06, fig07 and fig08. Every
    multi-series figure now carries one shared legend below the panels.
  * fig08's log x-axis showed only 10^-1. The actual radii are now
    ticked and labelled, since those are the numbers quoted in the
    text.
  * --only lets a single figure be regenerated without redoing the
    rest.

WHAT IT MAKES
-------------
  fig03_mcp_curves.pdf      masked-column pretraining against a
                            column-mean predictor, per epoch, both corpora
  fig04_clean.pdf           clean macro-F1 by arm, per-fold points shown
  fig05_sweep.pdf           macro-F1 against the perturbation budget
  fig06_envelope.pdf        worst-case envelope over all attacks
  fig07_strong.pdf          standard against strong PGD, and flip rate
  fig08_cw.pdf              CW flippability against the L2 radius
  fig09_recovery.pdf        declared-frozen column recoverability
  fig10_ablation.pdf        clean against worst-case, arms as points
  fig11_cost.pdf            training seconds and parameter count
  fig12_convergence.pdf     validation accuracy per federated round

Figures 1 and 2 are schematics (architecture, threat model) and are not
produced here; they belong in the manuscript as drawings.

WHAT IT READS
-------------
Only files already on disk. Anything missing is skipped with a line
saying so, and the rest still runs -- so this is safe to run now and
again after the CICIDS strong-PGD run lands.

ARM LABELLING, WHICH MATTERS
----------------------------
Three different conditions were written to disk under the arm name FB:
plain FlowMAE+AT, FlowMAE+AT with per-column dropout, and FlowMAE+AT
with the gradient-dispersion penalty. They are distinguished ONLY by
which file they are in. MANIFEST below therefore renames per file, and
nothing is read without going through it. Merging those rows by arm
name would silently turn three conditions into one fifteen-fold arm.

STYLE
-----
Okabe--Ito colours, which stay distinguishable under the common forms
of colour blindness, with a different marker and dash pattern per arm as
well, so the figures survive greyscale printing. One y-axis per panel
throughout, grid and spines kept recessive, and series named in a
legend rather than identified by colour alone.

USAGE
  python -u figs_flowmae_v1.py --out figs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# ----------------------------------------------------------------- style
PALETTE = {
    "B":       "#0072B2",
    "C":       "#56B4E9",
    "F":       "#CC79A7",
    "FB":      "#D55E00",
    "FM":      "#E69F00",
    "FMB":     "#009E73",
    "FB+GDR":  "#000000",
    "FB+CD":   "#888888",
}
MARKER = {
    "B": "o", "C": "s", "F": "v", "FB": "^",
    "FM": "P", "FMB": "D", "FB+GDR": "X", "FB+CD": "*",
}
DASH = {
    "B": (None, None), "C": (4, 2), "F": (1, 1.5), "FB": (None, None),
    "FM": (6, 2), "FMB": (None, None), "FB+GDR": (3, 1.5),
    "FB+CD": (1, 2),
}
ORDER = ["B", "C", "F", "FB", "FM", "FMB", "FB+CD", "FB+GDR"]

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # ------------------------------------------------------------------
    #  Text must stay vector and must match the manuscript.
    #
    #  fonttype 42 embeds TrueType outlines. The default, Type 3, is
    #  what made the figure text look soft next to the body text: it
    #  renders as a stream of drawing operators rather than as a real
    #  font, so it does not hint, it blurs when the reader zooms, and
    #  Springer and IEEE both reject it at production.
    #
    #  The body of the paper is set in Latin Modern. Matplotlib's
    #  default is DejaVu Sans, so every figure read as a different
    #  document. A serif stack with Latin Modern first matches the
    #  text; the fallbacks are for machines that do not have it.
    # ------------------------------------------------------------------
    #  The font is STIX rather than a system serif. v4 asked for Latin
    #  Modern and fell back to whatever each machine had: on the run
    #  machine the text came out in Times New Roman while the maths
    #  stayed in Computer Modern, so a single panel carried two
    #  different serifs. STIX ships with matplotlib, so every machine
    #  produces identical output, and its maths is designed to sit
    #  beside its text.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 9,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "legend.frameon": False,
})


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def shared_legend(fig, axes, ncol=None):
    """One legend under the whole figure, so no panel loses plot area.

    Handles are collected from every panel and de-duplicated by label,
    which keeps the arm order consistent across panels even when a
    corpus is missing an arm.
    """
    seen = {}
    for ax in np.ravel(axes):
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
    if not seen:
        return
    labs = [a for a in ORDER if a in seen] + \
           [l for l in seen if l not in ORDER]
    leg = fig.legend([seen[l] for l in labs], labs,
                     loc="lower center", ncol=ncol or min(len(labs), 6),
                     bbox_to_anchor=(0.5, 0.0), frameon=False)
    # v4 anchored this at y = -0.02 and let it grow upwards. With one
    # row that is harmless; with two it climbed over the x-axis label
    # and the tick labels, which is what happened to fig07. Measure
    # the legend and give the axes the rest of the figure, so the band
    # it occupies is reserved whatever the number of rows.
    fig.canvas.draw()
    h = leg.get_window_extent(
        renderer=fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0.0, min(h + 0.03, 0.45), 1.0, 1.0))


# ------------------------------------------------------- label placement
def place_labels(ax, pts, fontsize=7.5, pad=1.5, color=None):
    """Put a short label beside each point without letting two labels,
    or a label and a marker, overlap.

    The previous version counted how many earlier points were nearby
    and rotated the offset by a fixed angle per neighbour. That is not
    a collision test: it never looks at how wide a label actually is,
    so on UNSW-NB15 the five adversarially trained arms still produced
    overlapping text. This measures the rendered extent of each label
    and takes the first candidate position that is clear of everything
    already placed. If no position is clear, the least bad one is used
    and a leader line is drawn so the label can still be read against
    its point.
    """
    import itertools
    fig = ax.figure
    fig.canvas.draw()                       # extents need a renderer
    rend = fig.canvas.get_renderer()
    taken = []                              # display-space boxes

    # the markers themselves are obstacles
    for x, y, _ in pts:
        px, py = ax.transData.transform((x, y))
        taken.append((px - 5, py - 5, px + 5, py + 5))

    def clear(b):
        x0, y0, x1, y1 = b
        for t in taken:
            if not (x1 + pad < t[0] or x0 - pad > t[2]
                    or y1 + pad < t[1] or y0 - pad > t[3]):
                return False
        return True

    radii = (11, 16, 22, 29)
    angles = (35, 145, 215, 325, 90, 270, 0, 180, 60, 120, 240, 300)
    for x, y, lab in pts:
        best, best_pos, best_bad = None, None, 1e18
        for r, a in itertools.product(radii, angles):
            dx, dy = r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))
            t = ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize,
                            color=color or INK,
                            ha="left" if dx >= 0 else "right",
                            va="bottom" if dy >= 0 else "top")
            bb = t.get_window_extent(renderer=rend)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            if clear(box):
                taken.append(box)
                best = None
                break
            # how much it would overlap, so the fallback is the least bad
            bad = sum(max(0, min(box[2], o[2]) - max(box[0], o[0]))
                      * max(0, min(box[3], o[3]) - max(box[1], o[1]))
                      for o in taken)
            if bad < best_bad:
                best_bad, best_pos, best = bad, (dx, dy), box
            t.remove()
        else:
            # nothing was clear: use the least-overlapping position and
            # join it to its point so the pairing stays unambiguous
            dx, dy = best_pos
            t = ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize,
                            color=color or INK,
                            ha="left" if dx >= 0 else "right",
                            va="bottom" if dy >= 0 else "top",
                            arrowprops=dict(arrowstyle="-", lw=0.6,
                                            color=MUTED,
                                            shrinkA=1, shrinkB=3))
            taken.append(best)


# -------------------------------------------------------------- manifest
# (path, {arm name in the file: label to use}). Order fixes which file
# wins if the same label appears twice -- it should not.
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
    ],
}

MCP_LOGS = {
    "UNSW-NB15":   "out/log_mcpv2_f{f}.txt",
    "CICIDS-2017": "out/log_night_mcp_f{f}.txt",
}

STRONG = {
    "UNSW-NB15":   "out/strong_pgd_unsw.json",
    "CICIDS-2017": "out/strong_pgd_cicids.json",
}

RECOVERY = {
    "UNSW-NB15 (declared)":    "out/frozen_recovery_unsw_declared.json",
    "UNSW-NB15 (binary-only)": "out/frozen_recovery_unsw_binary.json",
}

EPS = [0.0, 0.01, 0.05, 0.1, 0.2]
L2S = ["0.02", "0.05", "0.1", "0.2", "0.4", "0.8"]


# ------------------------------------------------------------------ load
def load(verbose=True):
    """corpus -> list of rows, each relabelled by its source file."""
    data = {}
    for corpus, entries in MANIFEST.items():
        rows = []
        for path, remap in entries:
            p = Path(path)
            if not p.exists():
                if verbose:
                    print(f"  missing, skipped: {path}")
                continue
            n = 0
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("arm") not in remap:
                    continue
                r = dict(r)
                r["label"] = remap[r["arm"]]
                r["_src"] = p.name
                rows.append(r)
                n += 1
            if verbose:
                print(f"  {path}: {n} row(s) -> "
                      f"{sorted(set(remap.values()))}")
        if rows:
            data[corpus] = rows
    return data


def labels_in(rows):
    seen = {r["label"] for r in rows}
    return [a for a in ORDER if a in seen]


def f1_of(r):
    m = r.get("metrics") or {}
    return m.get("f1")


def per_fold(rows, label, attack, eps):
    """{fold: macro-F1} for one arm, attack and budget."""
    out = {}
    for r in rows:
        if r["label"] != label or r.get("attack") != attack:
            continue
        if abs(float(r.get("eps", -99)) - eps) > 1e-9:
            continue
        v = f1_of(r)
        if v is not None:
            out[int(r["fold"])] = float(v)
    return out


def clean_per_fold(rows, label):
    return per_fold(rows, label, "clean", 0.0)


def envelope_per_fold(rows, label):
    """Worst case over fgsm and pgd at this budget or below, per fold."""
    out = {}
    folds = sorted({int(r["fold"]) for r in rows if r["label"] == label})
    for fold in folds:
        best = {}
        running = None
        for e in EPS:
            vals = []
            if e == 0.0:
                v = clean_per_fold(rows, label).get(fold)
                if v is not None:
                    vals.append(v)
            for atk in ("fgsm", "pgd"):
                v = per_fold(rows, label, atk, e).get(fold)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            here = min(vals)
            running = here if running is None else min(running, here)
            best[e] = running
        out[fold] = best
    return out


def mean_sd(d):
    v = np.array([x for x in d.values() if x is not None], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std()), int(v.size)


# ------------------------------------------------------------- figure 03
def fig_mcp(out: Path):
    panels = []
    for corpus, tmpl in MCP_LOGS.items():
        series, bests = [], []
        for f in range(5):
            p = Path(tmpl.format(f=f))
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            ratios = [float(m) for m in re.findall(
                r"\(([0-9.]+) x mean predictor\)", txt)]
            if ratios:
                series.append(ratios)
            # the line the manuscript quotes, so the annotation below
            # cannot drift away from the text
            b = re.findall(r"best held-out .*?\(([0-9.]+) x\)", txt)
            if b:
                bests.append(float(b[-1]))
        if series:
            panels.append((corpus, series, bests))
    if not panels:
        print("  fig03 skipped: no pretraining logs found")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.7),
                             squeeze=False)
    for ax, (corpus, series, bests) in zip(axes[0], panels):
        n = min(len(s) for s in series)
        arr = np.array([s[:n] for s in series])
        x = np.arange(1, n + 1)
        for row in arr:
            ax.plot(x, row, color="#009E73", alpha=0.30, linewidth=1.0)
        ax.plot(x, arr.mean(0), color="#009E73", marker="D",
                markevery=max(1, n // 8), label="FlowMAE, mean of folds")
        ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 2)))
        ax.text(1, 1.03, "column-mean predictor", ha="left", va="bottom",
                fontsize=7.5, color=MUTED)
        ref = float(np.mean(bests)) if bests else float(arr.mean(0).min())
        ax.annotate(f"{1.0/ref:.0f}$\\times$ better than the baseline",
                    xy=(n, arr.mean(0)[-1]), xytext=(-6, 11),
                    textcoords="offset points", ha="right",
                    fontsize=8, color="#009E73")
        ax.set_yscale("log")
        ax.set_ylim(top=2.0)
        ax.set_xlabel("pretraining epoch")
        ax.set_ylabel("held-out MSE / column-mean MSE")
        ax.set_title(corpus)
        ax.legend(loc="lower left")
        tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig03_mcp_curves.pdf")
    plt.close(fig)
    print("  wrote fig03_mcp_curves.pdf")


# ------------------------------------------------------------- figure 04
def fig_clean(data, out: Path):
    corpora = list(data)
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.6 * len(corpora), 2.9),
                             squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        rows = data[corpus]
        labs = labels_in(rows)
        xs = np.arange(len(labs))
        for i, lab in enumerate(labs):
            d = clean_per_fold(rows, lab)
            m, s, n = mean_sd(d)
            ax.bar(i, m, width=0.62, color=PALETTE[lab], alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=2)
            ax.errorbar(i, m, yerr=s, fmt="none", ecolor=INK,
                        elinewidth=1.0, capsize=3, zorder=3)
            jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.22
            ax.scatter(i + jitter, list(d.values()), s=9, color=INK,
                       alpha=0.55, zorder=4, linewidths=0)
            ax.text(i, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
        ax.set_xticks(xs)
        # At 20 degrees the long arm names ("FB+CD", "FB+GDR")
        # ran into each other whenever all eight arms were
        # present. Upright labels cannot collide horizontally,
        # so a crowded panel is set vertical and a sparse one
        # is left flat, which reads better.
        if len(labs) > 5:
            ax.set_xticklabels(labs, rotation=90, ha="center",
                               va="top")
        else:
            ax.set_xticklabels(labs, rotation=0, ha="center")
        ax.set_ylabel("clean macro-F$_1$")
        ax.set_title(corpus)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig04_clean.pdf")
    plt.close(fig)
    print("  wrote fig04_clean.pdf")


# ------------------------------------------------------- figures 05 / 06
def _curve_panel(ax, rows, getter, title):
    for lab in labels_in(rows):
        ys, es = [], []
        for e in EPS:
            m, s, n = mean_sd(getter(rows, lab, e))
            ys.append(m)
            es.append(s)
        ys, es = np.array(ys), np.array(es)
        ok = ~np.isnan(ys)
        if ok.sum() == 0:
            continue
        ax.plot(np.array(EPS)[ok], ys[ok], color=PALETTE[lab],
                marker=MARKER[lab], dashes=DASH[lab], label=lab)
        ax.fill_between(np.array(EPS)[ok], (ys - es)[ok], (ys + es)[ok],
                        color=PALETTE[lab], alpha=0.12, linewidth=0)
    ax.set_xlabel(r"perturbation budget $\varepsilon$  ($L_\infty$)")
    ax.set_ylabel("macro-F$_1$")
    ax.set_title(title)
    ax.set_ylim(0, None)
    tidy(ax)


def fig_sweep(data, out: Path):
    corpora = list(data)
    fig, axes = plt.subplots(2, len(corpora),
                             figsize=(3.6 * len(corpora), 5.2), squeeze=False)
    for j, corpus in enumerate(corpora):
        rows = data[corpus]
        for i, atk in enumerate(("fgsm", "pgd")):
            def g(rws, lab, e, _a=atk):
                if e == 0.0:
                    return clean_per_fold(rws, lab)
                return per_fold(rws, lab, _a, e)
            _curve_panel(axes[i][j], rows, g,
                         f"{corpus} \u2013 {atk.upper()}")
    fig.tight_layout()
    shared_legend(fig, axes)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(out / "fig05_sweep.pdf")
    plt.close(fig)
    print("  wrote fig05_sweep.pdf")


def fig_envelope(data, out: Path):
    corpora = list(data)
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.6 * len(corpora), 3.0),
                             squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        rows = data[corpus]
        for lab in labels_in(rows):
            env = envelope_per_fold(rows, lab)
            ys, es = [], []
            for e in EPS:
                vals = {f: env[f].get(e) for f in env}
                m, s, n = mean_sd(vals)
                ys.append(m)
                es.append(s)
            ys, es = np.array(ys), np.array(es)
            ok = ~np.isnan(ys)
            if ok.sum() == 0:
                continue
            ax.plot(np.array(EPS)[ok], ys[ok], color=PALETTE[lab],
                    marker=MARKER[lab], dashes=DASH[lab], label=lab)
            ax.fill_between(np.array(EPS)[ok], (ys - es)[ok], (ys + es)[ok],
                            color=PALETTE[lab], alpha=0.12, linewidth=0)
        ax.set_xlabel(r"perturbation budget $\varepsilon$")
        ax.set_ylabel("worst-case macro-F$_1$")
        ax.set_title(f"{corpus} \u2013 worst case at this budget or below")
        ax.set_ylim(0, None)
        tidy(ax)
    fig.tight_layout()
    shared_legend(fig, axes)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(out / "fig06_envelope.pdf")
    plt.close(fig)
    print("  wrote fig06_envelope.pdf")


# ------------------------------------------------------------- figure 07
def fig_strong(out: Path):
    found = {c: json.loads(Path(p).read_text(encoding="utf-8"))
             for c, p in STRONG.items() if Path(p).exists()}
    if not found:
        print("  fig07 skipped: no strong-PGD json yet")
        return
    # A single panel is too narrow to carry a six-entry legend, so the
    # figure is widened when only one corpus has strong-PGD results.
    fig, axes = plt.subplots(1, len(found),
                             figsize=(max(5.4, 3.6 * len(found)), 3.0),
                             squeeze=False)
    for ax, (corpus, res) in zip(axes[0], found.items()):
        for lab, r in res.items():
            key = lab if lab in PALETTE else "FB"
            eps = sorted(float(e) for e in r["standard"])
            std = [r["standard"][str(e)] for e in eps]
            strg = [r["strong"][str(e)] for e in eps]
            ax.plot(eps, std, color=PALETTE[key], marker=MARKER[key],
                    dashes=(4, 2), alpha=0.55,
                    label=f"{lab}, 40-step")
            ax.plot(eps, strg, color=PALETTE[key], marker=MARKER[key],
                    label=f"{lab}, 200-step $\\times$10")
        ax.set_xlabel(r"perturbation budget $\varepsilon$")
        ax.set_ylabel("macro-F$_1$")
        ax.set_title(f"{corpus} \u2013 standard against strong PGD")
        ax.set_ylim(0, None)
        tidy(ax)
    # v4 put this legend inside the axes at an offset of -0.42, which
    # landed it on the tick labels and the x-axis label. The shared
    # figure legend used by fig06 and fig08 sits below everything and
    # is picked up by the tight bounding box on save.
    fig.tight_layout()
    shared_legend(fig, axes, ncol=3)
    fig.savefig(out / "fig07_strong.pdf")
    plt.close(fig)
    print("  wrote fig07_strong.pdf")


# ------------------------------------------------------------- figure 08
def fig_cw(data, out: Path):
    """The figure the gradient-masking claim rests on."""
    corpora = list(data)
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.6 * len(corpora), 3.0),
                             squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        rows = [r for r in data[corpus] if r.get("attack") == "cw"]
        for lab in labels_in(rows):
            per = {}
            for r in rows:
                if r["label"] != lab:
                    continue
                s = ((r.get("cw") or {}).get("success_at_l2") or {})
                for k, v in s.items():
                    per.setdefault(k, []).append(float(v))
            xs = [float(k) for k in L2S if k in per]
            ys = [float(np.mean(per[k])) for k in L2S if k in per]
            if not xs:
                continue
            ax.plot(xs, ys, color=PALETTE[lab], marker=MARKER[lab],
                    dashes=DASH[lab], label=lab)
        ax.set_xscale("log")
        # The radii quoted in the text are the ones that must be
        # readable; matplotlib's default log ticks show only 10^-1.
        ticks = [float(t) for t in L2S]
        ax.set_xticks(ticks)
        ax.set_xticklabels([t.rstrip("0").rstrip(".") if "." in t else t
                            for t in L2S])
        ax.set_xticks([], minor=True)
        ax.set_xlabel(r"$L_2$ radius $\tau$")
        ax.set_ylabel("fraction of correct rows flippable")
        ax.set_title(f"{corpus} \u2013 Carlini\u2013Wagner")
        ax.set_ylim(0, None)
        tidy(ax)
    fig.tight_layout()
    shared_legend(fig, axes)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(out / "fig08_cw.pdf")
    plt.close(fig)
    print("  wrote fig08_cw.pdf")


# ------------------------------------------------------------- figure 09
def fig_recovery(out: Path, top_n=20):
    """v1 drew all forty columns at 6 pt and the names became a smear.

    Only the top_n are drawn, at a readable size and with room for
    them; the title still reports the count over the full set, so
    nothing is hidden by the truncation.
    """
    found = {k: json.loads(Path(p).read_text(encoding="utf-8"))
             for k, p in RECOVERY.items() if Path(p).exists()}
    if not found:
        print("  fig09 skipped: no frozen-recovery json")
        return
    height = max(3.0, 0.26 * top_n + 1.3)
    fig, axes = plt.subplots(1, len(found),
                             figsize=(4.6 * len(found), height),
                             squeeze=False)
    for ax, (name, js) in zip(axes[0], found.items()):
        r2 = js.get("r2_mean", {})
        dup = js.get("duplicate", {})
        items = sorted(r2.items(), key=lambda kv: kv[1], reverse=True)
        n_all = len(items)
        n_hi = sum(1 for _, v in items if v >= 0.8)
        items = items[:top_n]
        names = [k for k, _ in items]
        vals = [max(0.0, v) for _, v in items]
        cols = ["#D55E00" if dup.get(k) else "#0072B2" for k in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=cols, height=0.68, edgecolor="white",
                linewidth=0.4)
        for yy, v in zip(y, vals):
            ax.text(min(v + 0.015, 0.995), yy, f"{v:.2f}", va="center",
                    ha="left", fontsize=7, color=INK)
        ax.axvline(0.8, color=INK, linewidth=1.0, dashes=(4, 2))
        ax.text(0.8, -1.0, "$R^2 = 0.8$", ha="center", va="bottom",
                fontsize=7.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("$R^2$ from the perturbable columns alone")
        ax.set_xlim(0, 1.08)
        shown = f"top {len(names)} of {n_all}" if n_all > len(names) \
            else f"all {n_all}"
        ax.set_title(f"{name}\n{n_hi} of {n_all} columns reach "
                     f"$R^2 \\geq 0.8$   ({shown} shown)")
        ax.grid(axis="y", visible=False)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig09_recovery.pdf")
    plt.close(fig)
    print("  wrote fig09_recovery.pdf")


# ------------------------------------------------------------- figure 10
def fig_ablation(data, out: Path):
    """Clean against worst case. One point per arm; up and right is better."""
    corpora = list(data)
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.6 * len(corpora), 3.2),
                             squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        rows = data[corpus]
        pts = []
        for lab in labels_in(rows):
            cm, cs, _ = mean_sd(clean_per_fold(rows, lab))
            env = envelope_per_fold(rows, lab)
            wm, ws, _ = mean_sd({f: env[f].get(0.2) for f in env})
            if np.isnan(cm) or np.isnan(wm):
                continue
            ax.errorbar(wm, cm, xerr=ws, yerr=cs, fmt=MARKER[lab],
                        color=PALETTE[lab], ecolor=PALETTE[lab],
                        elinewidth=0.9, capsize=2, markersize=7, label=lab)
            pts.append((wm, cm, lab))

        ax.margins(0.22)
        ax.set_xlabel(r"worst-case macro-F$_1$ at $\varepsilon = 0.2$")
        ax.set_ylabel("clean macro-F$_1$")
        ax.set_title(corpus)
        tidy(ax)
        ax._arm_pts = pts
    fig.tight_layout()
    # The five adversarially trained UNSW arms land on top of one
    # another, so the labels are placed against measured text extents
    # rather than by a fixed rotation per neighbour. This runs after
    # tight_layout, because the extents must be measured against the
    # axes at their final size.
    for ax in axes[0]:
        if getattr(ax, "_arm_pts", None):
            place_labels(ax, ax._arm_pts, fontsize=7.5)
    fig.savefig(out / "fig10_ablation.pdf")
    plt.close(fig)
    print("  wrote fig10_ablation.pdf")


# ------------------------------------------------------------- figure 11
def fig_cost(data, out: Path):
    corpora = list(data)
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.4 * len(corpora), 2.9),
                             squeeze=False)
    any_data = False
    for ax, corpus in zip(axes[0], corpora):
        rows = [r for r in data[corpus] if r.get("attack") == "clean"]
        labs = labels_in(rows)
        xs, ms, ss = [], [], []
        for i, lab in enumerate(labs):
            secs = [float(r["secs"]) for r in rows
                    if r["label"] == lab and r.get("secs")]
            if not secs:
                continue
            xs.append(i)
            ms.append(np.mean(secs) / 60.0)
            ss.append(np.std(secs) / 60.0)
        if not xs:
            continue
        any_data = True
        ax.bar(xs, ms, yerr=ss, width=0.62, color="#0072B2", alpha=0.85,
               edgecolor="white", linewidth=0.8,
               error_kw=dict(ecolor=INK, elinewidth=1.0, capsize=3))
        ax.set_xticks(range(len(labs)))
        # At 20 degrees the long arm names ("FB+CD", "FB+GDR")
        # ran into each other whenever all eight arms were
        # present. Upright labels cannot collide horizontally,
        # so a crowded panel is set vertical and a sparse one
        # is left flat, which reads better.
        if len(labs) > 5:
            ax.set_xticklabels(labs, rotation=90, ha="center",
                               va="top")
        else:
            ax.set_xticklabels(labs, rotation=0, ha="center")
        ax.set_ylabel("training time per fold (minutes)")
        ax.set_title(corpus)
        ax.grid(axis="x", visible=False)
        tidy(ax)
    if not any_data:
        plt.close(fig)
        print("  fig11 skipped: no timing fields")
        return
    fig.tight_layout()
    fig.savefig(out / "fig11_cost.pdf")
    plt.close(fig)
    print("  wrote fig11_cost.pdf")


# ------------------------------------------------------------- figure 12
def fig_convergence(data, out: Path):
    corpora = list(data)
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.5 * len(corpora), 2.9),
                             squeeze=False)
    drew = False
    for ax, corpus in zip(axes[0], corpora):
        rows = [r for r in data[corpus] if r.get("attack") == "clean"]
        for lab in labels_in(rows):
            series = []
            for r in rows:
                if r["label"] != lab:
                    continue
                h = (r.get("metrics") or {}).get("history") or []
                v = [float(d["val"]) for d in h if "val" in d]
                if v:
                    series.append(v)
            if not series:
                continue
            n = min(len(s) for s in series)
            arr = np.array([s[:n] for s in series])
            ax.plot(np.arange(1, n + 1), arr.mean(0), color=PALETTE[lab],
                    marker=MARKER[lab], dashes=DASH[lab], label=lab)
            drew = True
        ax.set_xlabel("communication round")
        ax.set_ylabel("validation accuracy")
        ax.set_title(corpus)
        tidy(ax)
    if not drew:
        plt.close(fig)
        print("  fig12 skipped: no per-round history")
        return
    axes[0][0].legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig12_convergence.pdf")
    plt.close(fig)
    print("  wrote fig12_convergence.pdf")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    ap.add_argument("--top-n", type=int, default=20,
                    help="columns drawn in fig09; the title still "
                         "reports the count over the full set")
    ap.add_argument("--only", default=None,
                    help="regenerate one figure: a number such as 09, "
                         "or a comma-separated list")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("reading:")
    data = load()
    if not data:
        raise SystemExit("no jsonl found; run this from supplementary\\")
    print()
    print("writing:")

    jobs = {
        "03": lambda: fig_mcp(out),
        "04": lambda: fig_clean(data, out),
        "05": lambda: fig_sweep(data, out),
        "06": lambda: fig_envelope(data, out),
        "07": lambda: fig_strong(out),
        "08": lambda: fig_cw(data, out),
        "09": lambda: fig_recovery(out, a.top_n),
        "10": lambda: fig_ablation(data, out),
        "11": lambda: fig_cost(data, out),
        "12": lambda: fig_convergence(data, out),
    }
    want = ([s.strip().zfill(2) for s in a.only.split(",")]
            if a.only else list(jobs))
    unknown = [w for w in want if w not in jobs]
    if unknown:
        raise SystemExit(f"--only: no such figure {unknown}; "
                         f"choose from {sorted(jobs)}")
    for k in want:
        jobs[k]()
    print()
    print(f"done. {len(list(out.glob('*.pdf')))} file(s) in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
