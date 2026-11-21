"""
phase1_arms_v1.py -- Phase 1 of the MC3-GRAM re-measurement protocol.

The decision point. Three training arms, identical in every other respect,
evaluated under real attacks on the curated UNSW-NB15 partition:

    A   mask-conditioned generator only        (what the paper had)
    B   PGD adversarial training only          (the baseline it never had)
    C   both                                   (does the generator add anything?)

Arm C against arm B is the strongest testable form of the paper's central
claim. If C does not separate from B, that is the result, and we find it out
here on 257k records rather than after the full sweep.

Place this in `supplementary/` beside run.py, with attack_suite_v4.py,
gates_v1.py, cicids_loader_v1.py and ciciot_loader_v2/v4/v5.py in the same
folder. It modifies none of the repository's files:
`fed.local_train` is wrapped at runtime so arms B and C can add adversarial
examples, and everything else -- the federated loop, aggregation, mask
consensus, evaluation -- is the repository's own code.

Usage
-----
    python phase1_arms_v1.py --root data --smoke          # ~minutes, 1 fold
    python phase1_arms_v1.py --root data                  # the real Phase 1

Rows land in out/phase1.jsonl via gate G1 and are skipped if already present,
so the run can be stopped and restarted freely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as ds                      # noqa: E402
import cicids_loader_v1 as cic            # noqa: E402
import ciciot_loader_v7 as ciot           # noqa: E402
import fed                                # noqa: E402
from model import MC3GRAM                 # noqa: E402
from attack_suite_v4 import (             # noqa: E402
    FeatureSpec, build_eval_set, cw_l2,
)
import gates_v1 as G                      # noqa: E402

VERSION = "phase1_arms_v17"

# v16 puts the feature spec into the run record and into the resume key.
#
# Which columns an adversary may perturb decides every attack number here,
# and it is a DECLARED assumption, not a fact the data can settle. The grain
# check on CIC-IoT showed why: Protocol Type takes 3,621 distinct values
# between 0 and 47, Number takes 96 between 1 and 15, ack_count reaches 5.2.
# Every feature in that corpus is averaged over a window of 10 or 100
# packets, so almost nothing is integral -- and fractional in the data does
# not mean settable by an attacker. What the check did settle is the 18
# strictly binary columns, where a step of 0.03 produces a row that cannot
# occur.
#
# So the spec is a choice, different reasonable choices give different
# robustness numbers, and the choice has to travel with the numbers. Each
# row now carries `spec_fingerprint`, a hash of the frozen column names, and
# the resume key carries it too. Without that, changing the spec and
# re-running would skip every attack cell as already done and leave the old
# spec's numbers in place with nothing to show for it -- the same class of
# silent incompleteness v13 fixed for the corpus.

# v17 splits the spec in two.
#
# v16 used ONE perturbable set for both jobs the spec does: it decides which
# columns adversarial training defends, and it decides which columns the
# evaluation attack is allowed to move. Running v16 twice under two specs
# therefore changes the defence and the attack together, and the two effects
# cannot be separated. That is why v16's sensitivity run was restricted to
# arms D and A, which do no adversarial training and so depend on the spec
# only through the attack.
#
# The question that restriction leaves unanswered is the one that matters: a
# defence is built for a declared threat model, and the attacker is under no
# obligation to use it. v17 takes --train-spec and --attack-spec separately,
# so a model hardened against the declared set can be attacked under the
# permissive one. Matched specs reproduce v16 exactly, fingerprint included.
#
# Two consequences worth stating:
#
#   The resume key carries the PAIR. A mismatch run's attack spec equals the
#   matched run's attack spec, so keying on the attack spec alone would make
#   a mismatch cell look already done -- the same silent-skip family as v13
#   and v16. `spec_pair` is the key; for a matched run it is the single
#   fingerprint, so old rows still resume.
#
#   The model fingerprint now carries the TRAIN spec, and only when
#   adversarial training is actually on. In v16 it carried the one spec
#   unconditionally, so arms D and A -- whose training cannot depend on any
#   spec -- were retrained under the second spec for nothing. The correction
#   also means a mismatch run finds the matched run's hardened models already
#   in the cache and recomputes only the attacks.
#
# v15 moves CIC-IoT onto the release the paper describes.
#
# v12-v14 ran against a copy that is not that release: 63 merged files of 40
# columns, or a 309-file per-class tree of 39, carrying Time_To_Live and
# lacking flow_duration, Duration, Srate and Drate, and matching Table 3 of
# Neto et al. (2023) on no class. The release has 169 merged files, 47
# columns, and reproduces Table 3 exactly -- every class, total 46,686,579.
#
# So: G3 for CIC-IoT asserts against the PUBLISHED counts again, --packaging
# is gone because the release has no packagings, and the frozen-feature list
# comes from ciciot_loader_v7, which derives it from the header on disk
# rather than from a list written here. Every CIC-IoT row written by v12-v14
# describes the other corpus and is withdrawn.

# v14 moved to ciciot_loader_v6 and matched its split of the merged
# packaging's row count into labelled (45,019,234) and unlabelled (9). v13
# compared a labelled count against a line count and would have failed G3 by
# nine rows on the merged arm. Both quantities are now asserted separately.

# v13 fixes two resume bugs, both found by the first CIC-IoT smoke run, and
# both of the same family as the one v10 fixed.
#
#  1. THE RESUME KEY WAS CORPUS-BLIND. done_keys stored (arm, fold, attack,
#     eps) -- four fields that say nothing about which corpus a row came
#     from. out/phase1.jsonl already held UNSW and CICIDS rows, so the
#     CIC-IoT run matched arm A fold 0 fgsm eps=0.01 against a UNSW row and
#     skipped every FGSM and PGD sweep for arms A, B and C. Arm D ran because
#     no arm-D row happened to be there. Nothing written was wrong; the table
#     was simply incomplete, and silently so. The key now carries corpus,
#     task and packaging, and a tally per group prints at startup.
#
#  2. THE CW ROW NEVER RESUMED. The check read (arm, fold, "cw", 0.0) while
#     the row is written with eps=-1.0, so it never matched: CW was recomputed
#     on every run and an extra row appended each time. The analysis indexes
#     by fold and a duplicate overwrites rather than double-counts, so the
#     numbers survived, but two runs on different devices would have had the
#     later row win without a word.
#
#  A completeness report now runs at the end: expected cells for this
#  corpus/task/packaging against what the file holds, naming anything
#  missing. A silent gap is the failure mode these two bugs shared, and it is
#  the one thing neither G1 nor G5 was watching for.
#
#  attack_suite_v4 replaces v3: identical behaviour, minus a numpy warning
#  that fired whenever a CW attack flipped nothing at all.

# v12 adds CIC-IoT-2023 as a third corpus, and with it an experiment the other
# two cannot run.
#
#   --corpus ciciot --packaging perclass    (the declared corpus)
#   --corpus ciciot --packaging merged      (the same corpus, other packaging)
#
# The release ships the same 39 features in two packagings holding different
# rows: 46,649,056 in the per-class tree against 45,019,243 in the merged set,
# disagreeing on all 34 classes, and the merged set is not a subset -- it holds
# 104,330 MORE DoS-UDP_Flood rows than the per-class tree. Running both as arms
# of one experiment -- same model, same folds, same features, same seed, one
# thing different -- asks whether which file set a study downloaded changes its
# numbers. That is claim C3 stated by a corpus about itself, and it is sharper
# than two different corpora disagreeing.
#
# G3 for this corpus asserts against the observed counts of the identified
# copy, not against Table 3 of Neto et al. (2023), which describes neither
# packaging: it is short 0.08 % against the per-class tree, reports
# VulnerabilityScan at 37,382 where the release holds 373,351, and lists 47
# features where the CSVs carry 39. The gate still demands an exact match on
# all 34 classes, so a truncated or re-cut copy fails it; only the reference
# changed, and ciciot_loader_v5 records why.
#
# Rate and seed for the uniform sample are fixed at the declared 0.05 / 20231.
# They are exposed as flags so a deliberate change is visible in the run
# record and in the model fingerprint, not so they can be tuned.

# v11 fixes an ordering bug introduced in v10: base = dict(...) recorded
# attack_rows=len(Xatk) while the block that computes Xatk sat further down,
# so the first fold died with UnboundLocalError. The subsample is now chosen
# before the record is built.

# v10 fixes a duplicate-row bug and adds an attack-evaluation subsample.
#
#  1. The clean row's resume key read ("clean", fold, "clean", 0.0) while
#     done_keys stores (arm, fold, attack, eps). The literal "clean" sat in the
#     arm position, so the key never matched and the clean row was appended on
#     every re-run -- arm D fold 0 ended up with 12 rows instead of 10. The
#     analysis survived it, because it indexes by fold and a duplicate
#     overwrites rather than double-counts, but two runs on different devices
#     would have had the later clean row win silently.
#
#  2. --attack-rows evaluates the ATTACKS on a uniform random subsample of the
#     test partition while the clean metrics stay on all of it. On CICIDS the
#     attacks, not the training, are the cost: 566k test rows under PGD-40 at
#     four budgets ran about 35 minutes a fold, which puts one task past 16
#     hours.
#
#     The subsample is UNIFORM, not stratified: stratifying would move the
#     partition off the corpus distribution and reintroduce exactly the defect
#     G2 exists to catch. The price is that a class with little support gets
#     very few rows -- Infiltration+Heartbleed has about 9 in a full CICIDS
#     test partition -- so every row records test_rows and attack_rows, and a
#     class whose attacked support falls below 30 must be reported with its
#     count rather than folded into a macro average.

# v9: --corpus and --task, so the same arms, gates and attacks run on
# CICIDS-2017 as on UNSW-NB15. Nothing about the experiment changes; only the
# loader, the G3 spec, the G2 reference distribution and the frozen-feature
# set are corpus-specific.
#
# --smote-cap is exposed because make_folds materialises every client's SMOTE'd
# shard at once. At 2.83M rows and a 20,000 cap that is roughly 300k rows per
# client across ten clients, and the repository's own README warns about this
# at a smaller scale than CICIDS.

# v8: the model cache is keyed on a fingerprint of everything that determines
# the model, not just the arm and the fold.
#
# v7's key was "{arm}_f{fold}.pt", which recorded nothing about rounds,
# clients, alpha, subsample, the adversarial budget or the device. Running
# --smoke therefore loaded the model trained on the FULL corpus and evaluated
# it against a 12k near-balanced smoke partition, reporting a cached clean
# macro-F1 of 0.4298 as if it belonged to that run. The same key would have
# served a stale model to every later sweep over rounds or clients.
#
# The fingerprint includes the device, so CPU and CUDA keep separate caches and
# a device-parity check retrains on each rather than reusing one model. Caches
# written by v7 are now orphaned; they are left on disk rather than deleted.

# v7: --device, for the GPU port of fed.py and model.py.
#
# Two of the device bugs were in this file: local_train_with_adv built its
# tensors on the CPU, and the clean-prediction pass before CW did the same.
# Both followed fed.local_train, which had the same bug, so on CPU nothing
# showed. Device handling now comes from fed.dev_of(model) throughout.
#
# Note on verification: CPU and CUDA will NOT return identical numbers. The
# kernels differ in low-order bits and five rounds of training accumulate
# that, so expect agreement to about the third decimal and the same ordering
# between arms -- not equality. Anything larger than fold noise is a bug.

# v6: five folds by default, attack_suite_v3, and the CW budget curve printed
# instead of a mean L2 that cannot be compared across arms. Folds 0-2 load
# from --cache-dir, so only folds 3 and 4 are trained.
#
# Five folds because every C-vs-B comparison so far points the same way --
# PGD retention 3/3, FGSM retention 3/3, CW success 3/3 -- while each
# individual paired test sits at p around 0.18 to 0.24. The effect is small
# and real-looking; three folds cannot resolve it.

# v5: fixes a crash in the G5 block under --cw-only. With the eps sweeps
# skipped, each fold contributes only the eps=0 point, g5_monotone returns
# {"ok": True, "note": "fewer than two budgets"} with no "degradation" key,
# and the print reached for it. G5 is now reported only when at least two
# budgets are present, which is the only case in which it means anything.

# v4, after the arm-D run:
#
#  MODEL CACHING. Arm D at the raised CW budget reached a 0.54-0.86 success
#  rate where arm A at the old budget reached 0.13-0.18. The constraint was
#  never binding: the optimiser was short of budget, so the CW rows for arms
#  A, B and C are wrong and have to be recomputed. Recomputing them meant
#  retraining every arm, because nothing was checkpointed -- about two hours
#  to redo one table row.
#
#  Each trained model and its clean metrics are now written to --cache-dir
#  and reloaded when present, so changing an attack never costs a retrain
#  again, and extending to five folds will only train the two new ones.

# v3, after the three-fold Phase 1 run:
#
#  1. ARM D ADDED -- neither generator nor adversarial training. Without it
#     arms A and B differ in TWO ways at once, so A's higher clean macro-F1
#     (0.4397 against 0.3898) cannot be attributed to the generator: it is at
#     least as likely to be adversarial training's standard accuracy cost.
#     D-vs-A isolates the generator, D-vs-B isolates adversarial training,
#     B-vs-C isolates the generator on top of adversarial training.
#
#  2. CW budget raised and the pool widened. At steps=120/binary_steps=5 the
#     success rate came back at 0.005-0.18 for an UNBOUNDED attack, which
#     should succeed almost always. Either the optimiser is still short of
#     budget or the frozen-feature constraint genuinely binds (only 20 of 60
#     columns are perturbable, and sttl/dttl/ct_* are among the strongest
#     discriminators in this corpus). The second would be a real finding, so
#     it has to be separated from the first rather than reported as robustness.
#     At success 0.005 on 400 rows the mean L2 was computed over two rows.
#
#  3. Success rate is now the primary CW number; mean L2 is reported only when
#     at least 10 rows were actually flipped.

# v2 fixes, all three found by the v1 smoke run:
#
#  1. --subsample now defaults to None, and G2 ASSERTS against the corpus
#     proportions instead of merely printing them. make_folds has
#     keep_whole_below=5000, so any subsample small enough to bite caps every
#     large class at 5,000 and hands back a near-balanced corpus -- the very
#     defect this protocol exists to prevent. The v1 smoke run produced six
#     classes at exactly 0.1363 and G2 said nothing, because v1 passed no
#     reference counts.
#
#  2. CW now runs only on rows the model classifies CORRECTLY on clean input.
#     With clean accuracy near 0.37 most rows are already misclassified, count
#     as fooled at zero distance, and drag the mean L2 to 0.0003. That number
#     described the error rate, not the margin.
#
#  3. The C-vs-B comparison is now made on RETENTION, F1(eps)/F1(0), with
#     per-fold spread, not on levels. In the smoke run C beat B by +0.043 at
#     every budget INCLUDING eps=0, so the gap was clean accuracy carried
#     through, and C's degradation was in fact slightly worse than B's
#     (0.0435 against 0.0410). Reading levels as robustness is the mistake
#     that produced the rejected submission.
PROTOCOL = "remeasure_v1"

# The curated partition's own counts. NOT dataset.TABLE3, which carries the
# full capture's numbers plus the manuscript's DoS transposition (16,535 for a
# true 16,353) and would report three spurious mismatches here.
CURATED_COUNTS = {
    "Normal": 93_000, "Generic": 58_871, "Exploits": 44_525,
    "Fuzzers": 24_246, "DoS": 16_353, "Reconnaissance": 13_987,
    "Analysis": 2_677, "Backdoor": 2_329, "Shellcode": 1_511, "Worms": 174,
}

# CICIDS-2017: the columns an adversary cannot set to a fractional value.
#   destination port     a categorical identifier, not a quantity
#   *flag*               TCP flag counts and PSH/URG flags
#   header length        integer byte counts
#   init_win             advertised TCP window
#   act_data_pkt         count of packets carrying payload
#   min_seg_size         integer segment size
#   subflow              integer subflow packet and byte counts
CICIDS_FROZEN = ("destination port", "flag count", "psh flags", "urg flags",
                 "header length", "init_win", "act_data_pkt", "min_seg_size",
                 "subflow")

# Columns an adversary cannot set to an arbitrary fractional value. Frozen by
# name, because the one-hot expansion makes positions corpus-dependent.
#   proto_/service_/state_  one-hot indicators
#   sttl, dttl              IPv4 TTL, an integer an attacker sets but not by 0.03
#   swin, dwin              advertised TCP window
#   stcpb, dtcpb            TCP base sequence numbers
#   ct_*                    connection counts over the flow window
#   is_*                    boolean flags
#   trans_depth             integer depth into the connection
UNSW_FROZEN = ("proto_", "service_", "state_", "sttl", "dttl", "swin", "dwin",
               "stcpb", "dtcpb", "ct_", "is_ftp_login", "is_sm_ips_ports",
               "trans_depth")

# CIC-IoT-2023's frozen list is NOT written here. ciciot_loader_v7
# derives it from the header on disk and load_corpus returns it, because
# the previous version of this constant described a different release's
# columns and would have silently frozen nothing on this one.

TASKS_BY_CORPUS = {
    "unsw": ("multiclass10",),
    "cicids": ("binary", "multiclass8", "multiclass15"),
    "ciciot": ("binary", "multiclass8", "multiclass34"),
}

EPS_GRID = (0.01, 0.05, 0.10, 0.20)

ARMS = {
    "A": dict(gan=True, aug_frac=0.15, adv_frac=0.0,
              note="mask-conditioned generator only"),
    "B": dict(gan=False, aug_frac=0.0, adv_frac=0.5,
              note="PGD adversarial training only"),
    "C": dict(gan=True, aug_frac=0.15, adv_frac=0.5,
              note="both"),
    "D": dict(gan=False, aug_frac=0.0, adv_frac=0.0,
              note="neither -- the control that isolates each component"),
}


# --------------------------------------------------------------------------
# Adversarial training, added by wrapping fed.local_train
# --------------------------------------------------------------------------
_ADV = {"frac": 0.0, "eps": 0.1, "steps": 5, "spec": None}
DEV = torch.device("cpu")
_ORIG_LOCAL_TRAIN = fed.local_train


def _pgd_batch(model, xb, yb, eps, steps, pmask):
    """PGD on one batch, against the model as it currently stands."""
    was_training = model.training
    model.eval()
    x0 = xb.detach()
    alpha = max(eps / 4.0, 1e-6)
    x = (x0 + (torch.rand_like(x0) * 2 - 1) * eps)
    x = x0 + (x.clamp(0, 1) - x0) * pmask
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x), yb)
        g, = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * g.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0, 1)
        x = x0 + (x - x0) * pmask
    if was_training:
        model.train()
    return x.detach()


def local_train_with_adv(model, X, y, epochs=1, lr=1e-3, bs=512, gen=None,
                         mask=None, aug_frac=0.0, n_cls=10, max_batches=None):
    """fed.local_train plus optional PGD adversarial training.

    The generative-augmentation branch is kept byte-for-byte equivalent to the
    repository's own, so arm A reproduces its behaviour exactly and the only
    difference between the arms is the adversarial branch.
    """
    adv_frac = _ADV["frac"]
    if adv_frac <= 0:
        return _ORIG_LOCAL_TRAIN(model, X, y, epochs=epochs, lr=lr, bs=bs,
                                 gen=gen, mask=mask, aug_frac=aug_frac,
                                 n_cls=n_cls, max_batches=max_batches)

    spec = _ADV["spec"]
    dv = fed.dev_of(model)
    pmask = torch.from_numpy(
        spec.perturbable.astype(np.float32)).to(dv).unsqueeze(0)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    Xt, yt = torch.from_numpy(X).to(dv), torch.from_numpy(y).to(dv)
    for _ in range(epochs):
        perm = torch.randperm(n)
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            if xb.size(0) < 2:
                continue

            if gen is not None and aug_frac > 0:
                k = max(1, int(aug_frac * xb.size(0)))
                cnt = torch.bincount(yb, minlength=n_cls).float()
                cnt[cnt == 0] = float("inf")
                rare = torch.topk(-cnt, min(3, n_cls)).indices
                ys = rare[torch.randint(0, len(rare), (k,), device=dv)]
                gen.eval()
                with torch.no_grad():
                    xs = gen(k, ys, mask if mask is not None
                             else torch.ones(X.shape[1], device=dv))
                xb = torch.cat([xb, xs.to(dv)], 0)
                yb = torch.cat([yb, ys], 0)

            m = max(1, int(adv_frac * xb.size(0)))
            sel = torch.randperm(xb.size(0))[:m]
            x_adv = _pgd_batch(model, xb[sel], yb[sel],
                               _ADV["eps"], _ADV["steps"], pmask)
            xb = torch.cat([xb, x_adv], 0)
            yb = torch.cat([yb, yb[sel]], 0)

            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
            nb += 1
            if max_batches and nb >= max_batches:
                break
    return model


fed.local_train = local_train_with_adv


# --------------------------------------------------------------------------
def build_spec(feature_names, frozen) -> FeatureSpec:
    spec = FeatureSpec.from_names(feature_names, frozen, name="corpus")
    frozen = [c for c, p in zip(feature_names, spec.perturbable) if not p]
    print(f"  feature spec: {int(spec.perturbable.sum())}/{spec.d} perturbable")
    print(f"  frozen ({len(frozen)}): {', '.join(frozen[:12])}"
          + (" ..." if len(frozen) > 12 else ""))
    return spec


def g3_curated(df) -> None:          # superseded by load_corpus(); kept for reference
    got = df["target"].value_counts().to_dict()
    bad = [f"{k}: got {int(got.get(k, 0))}, spec {v}"
           for k, v in CURATED_COUNTS.items() if int(got.get(k, 0)) != v]
    if bad:
        raise G.GateFailure("G3: curated partition does not match its spec. "
                            + "; ".join(bad))
    print(f"  G3 ok: all ten counts match, {len(df)} rows pooled")


def load_corpus(args):
    """Load, then run G3 against that corpus's published spec."""
    if args.corpus == "unsw":
        df = ds.load_unsw(args.root, verbose=False)
        got = df["target"].value_counts().to_dict()
        bad = [f"{k}: got {int(got.get(k, 0))}, spec {v}"
               for k, v in CURATED_COUNTS.items() if int(got.get(k, 0)) != v]
        if bad:
            raise G.GateFailure("G3: curated UNSW partition does not match its "
                                "spec. " + "; ".join(bad))
        print(f"  G3 ok: all ten counts match, {len(df)} rows pooled")
        return df, UNSW_FROZEN

    if args.corpus == "cicids":
        rep = cic.check_counts(args.root)
        if rep["bad"] or rep["extra"] or rep["total"] != cic.TOTAL:
            raise G.GateFailure(
                "G3: CICIDS-2017 does not match the published release. "
                + "; ".join(rep["bad"][:6] or [f"total {rep['total']}"]))
        print(f"  G3 ok: all 15 raw counts and the total ({cic.TOTAL:,}) match")
        df = cic.load_cicids(args.root, task=args.task, verbose=False)
        print(f"  task={args.task}: {len(df):,} rows, {df.shape[1] - 1} "
              f"features, {df['target'].nunique()} classes")
        return df, CICIDS_FROZEN

    # ---- CIC-IoT-2023 ---------------------------------------------------
    # G3 runs on the CORPUS, not on the sample, and against Table 3, which
    # this release reproduces exactly. Cached, so it costs about a second
    # after the first time.
    rep = ciot.g3(args.root, cache=args.ciciot_cache, verbose=False)
    if not rep["ok"]:
        raise G.GateFailure(
            "G3: this CIC-IoT-2023 copy does not match Table 3 of Neto et "
            "al. (2023). " + "; ".join(rep["bad"][:6]))
    print(f"  G3 ok: all 34 class counts and the total ({rep['total']:,}) "
          "match Table 3 exactly")

    df = ciot.load_ciciot(args.root, task=args.task, rate=args.ciciot_rate,
                          seed=args.ciciot_seed, cache=args.ciciot_cache,
                          verbose=True)
    # The frozen list is derived from the header actually on disk, by
    # ciciot_loader_v7, not written out here. build_spec is given the
    # resolved names so FeatureSpec.from_names matches them exactly.
    frozen = tuple(c for c in df.columns
                   if c != "target" and ciot.is_frozen(c))
    return df, frozen


def g2_reference(args, classes):
    """The corpus distribution G2 asserts the test partition against."""
    if args.subsample:
        return None
    if args.corpus == "unsw":
        return {classes.index(k): v for k, v in CURATED_COUNTS.items()
                if k in classes}

    if args.corpus == "ciciot":
        # The release matches Table 3, so G2 asserts against the published
        # shares rather than against this copy's own.
        per = dict(ciot.COUNTS_34)
        if args.task == "multiclass8":
            counts = dict(ciot.CATEGORY_TOTALS)
        elif args.task == "binary":
            b = per["BenignTraffic"]
            counts = {"BenignTraffic": b, "Attack": sum(per.values()) - b}
        else:
            counts = per
        return {classes.index(k): v for k, v in counts.items() if k in classes}

    counts = dict(cic.COUNTS_15)
    if args.task == "binary":
        att = sum(v for k, v in counts.items() if k != "BENIGN")
        counts = {"BENIGN": counts["BENIGN"], "Attack": att}
    elif args.task == "multiclass8":
        g: dict[str, int] = {}
        for k, v in cic.COUNTS_15.items():
            g[cic.GROUP_8[k]] = g.get(cic.GROUP_8[k], 0) + v
        counts = g
    return {classes.index(k): v for k, v in counts.items() if k in classes}


def model_fingerprint(arm: str, fold, args, acfg: dict, dev) -> str:
    """Short hash of every setting that changes the trained model.

    Anything that does not change the model stays out, so a change of attack
    or of CW budget still reuses the cache.
    """
    parts = [
        f"v={PROTOCOL}", f"arm={arm}", f"fold={fold.fold}", f"seed={fold.seed}",
        f"rounds={args.rounds}", f"clients={args.clients}",
        f"alpha={args.alpha}", f"subsample={args.subsample}",
        f"folds={args.folds}", f"gan={acfg['gan']}",
        f"aug={acfg['aug_frac']}", f"advfrac={acfg['adv_frac']}",
        f"adveps={args.adv_eps}", f"advsteps={args.adv_steps}",
        f"maxb={args.max_batches}", f"dev={dev.type}",
        f"corpus={args.corpus}", f"task={args.task}",
        f"smotecap={args.smote_cap}",
        f"ciotrate={args.ciciot_rate}",
        # the TRAIN spec, and only when adversarial training uses it: arms
        # with adv_frac == 0 cannot have a spec-dependent model, so making
        # them depend on one retrains them for nothing.
        f"spec={TRAIN_FP if acfg['adv_frac'] > 0 else '-'}",
        f"ciotseed={args.ciciot_seed}",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def _norm_task(corpus, task):
    """UNSW has exactly one task. v11 and earlier left args.task at its
    argparse default and recorded UNSW rows as task='multiclass8'; v12 names
    it 'multiclass10'. Both are folded here so old UNSW rows still resume."""
    return "multiclass10" if corpus == "unsw" else task


def row_key(corpus, task, packaging, arm, fold, attack, eps):
    return (corpus, _norm_task(corpus, task),
            packaging or "-", arm, int(fold), attack, float(eps))


# set by main() once the feature specs are resolved; part of the resume key.
# SPEC_FP is the ATTACK spec, because that is what the attack numbers are
# conditional on. TRAIN_FP is what the model was hardened against. PAIR_FP is
# the resume key: equal to SPEC_FP when the two match, so every v16 row still
# resumes, and distinct when they do not.
SPEC_FP = "?"
TRAIN_FP = "?"
PAIR_FP = "?"
this_group = None


def pair_fp(train_fp: str, atk_fp: str) -> str:
    return atk_fp if train_fp == atk_fp else f"{train_fp}>{atk_fp}"


def run_key(args, arm, fold, attack, eps):
    return row_key(args.corpus, args.task, PAIR_FP, arm, fold, attack, eps)


def done_keys(path: Path) -> tuple[set, dict]:
    """Resume keys already in the file, and a count per corpus/task/packaging.

    v12 and earlier keyed resume on (arm, fold, attack, eps) alone. Those four
    say nothing about WHICH corpus a row came from, so a CIC-IoT run reading a
    file that already held UNSW rows treated arm A fold 0 fgsm as done and
    skipped every sweep for arms A, B and C -- silently, because the rows it
    did write were correct and the ones it skipped simply never appeared. The
    key now carries corpus, task and packaging, and the per-group tally is
    printed at startup so an inherited skip cannot pass unnoticed.
    """
    if not path.exists():
        return set(), {}
    out, groups = set(), {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:                                  # noqa: BLE001
            continue
        corpus = r.get("corpus", "?")
        task = r.get("task", "?")
        sfp = (r.get("spec_pair") or r.get("spec_fingerprint") or "-")
        out.add(row_key(corpus, task, sfp, r["arm"], r["fold"],
                        r["attack"], r["eps"]))
        g = (corpus, _norm_task(corpus, task), sfp)
        groups[g] = groups.get(g, 0) + 1
    return out, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="unsw",
                    choices=("unsw", "cicids", "ciciot"))
    ap.add_argument("--task", default="multiclass8",
                    choices=("binary", "multiclass8", "multiclass15",
                             "multiclass34"),
                    help="CICIDS and CIC-IoT only; UNSW is always its ten "
                         "classes")
    ap.add_argument("--ciciot-rate", type=float, default=ciot.DEFAULT_RATE,
                    help="declared uniform sampling rate; changing it is a "
                         "protocol change and lands in the fingerprint")
    ap.add_argument("--ciciot-seed", type=int, default=ciot.DEFAULT_SEED)
    ap.add_argument("--ciciot-cache", default=r"out/ciciot_cache")
    ap.add_argument("--frozen-mode", default="declared",
                    choices=("declared", "binary-only"),
                    help="which columns an adversary may NOT perturb. "
                         "'declared' uses the corpus's name-based list. "
                         "'binary-only' freezes just the strictly {0,1} "
                         "columns, which is all the data itself can settle. "
                         "The two give different robustness numbers; running "
                         "both measures how much that undeclared choice is "
                         "worth. The spec fingerprint keeps them apart. "
                         "In v17 this is the shorthand that sets both "
                         "--train-spec and --attack-spec; give those "
                         "separately for a threat-model mismatch.")
    ap.add_argument("--train-spec", default=None,
                    choices=("declared", "binary-only"),
                    help="the set adversarial training defends. Defaults to "
                         "--frozen-mode. Arms D and A do no adversarial "
                         "training, so this does not reach them.")
    ap.add_argument("--attack-spec", default=None,
                    choices=("declared", "binary-only"),
                    help="the set the evaluation attack may move. Defaults "
                         "to --frozen-mode. Setting this away from "
                         "--train-spec is the threat-model mismatch: a "
                         "defence built for one declared set, attacked under "
                         "another.")
    ap.add_argument("--smote-cap", type=int, default=20_000)
    ap.add_argument("--attack-rows", type=int, default=None,
                    help="evaluate the attacks on this many uniformly sampled "
                         "test rows (clean metrics stay on the full partition)")
    ap.add_argument("--root", default="data", help="corpus folder")
    ap.add_argument("--out", default="out/phase1.jsonl")
    ap.add_argument("--arms", nargs="+", default=["D", "A", "B", "C"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--subsample", type=int, default=None,
                    help="DO NOT use for headline numbers: make_folds caps "
                         "every large class at keep_whole_below=5000, which "
                         "returns a near-balanced corpus and violates G2")
    ap.add_argument("--adv-eps", type=float, default=0.1)
    ap.add_argument("--adv-steps", type=int, default=5)
    ap.add_argument("--cw-n", type=int, default=1000, help="rows for CW (it is slow)")
    ap.add_argument("--cw-steps", type=int, default=400)
    ap.add_argument("--cw-binary", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--cache-dir", default="out/models",
                    help="trained models are saved here and reloaded if present")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; sets fed.DEVICE and cfg['device']")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the cache and retrain")
    ap.add_argument("--cw-only", action="store_true",
                    help="skip the eps sweeps; re-run CW alone at the new budget")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 12k rows, 2 rounds -- for timing only")
    args = ap.parse_args()

    if args.corpus == "unsw":
        args.task = "multiclass10"
    elif args.task not in TASKS_BY_CORPUS[args.corpus]:
        raise SystemExit(
            f"--task {args.task} is not defined for --corpus {args.corpus}; "
            f"choose from {', '.join(TASKS_BY_CORPUS[args.corpus])}")

    if args.smoke:
        args.folds, args.rounds, args.subsample = 1, 2, 12_000
        args.cw_n, args.max_batches = 120, 6
        args.cw_steps, args.cw_binary = 120, 5
        print("*** SMOKE: --subsample 12000 makes the corpus near-balanced "
              "(G2 is reported, not asserted). Timing only; never a result.")

    global DEV
    DEV = fed.set_device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have, groups = done_keys(out_path)

    print(f"{VERSION}  protocol={PROTOCOL}")
    print(f"torch {torch.__version__}  threads="
          f"{os.environ.get('MC3_THREADS', '2')}  device={DEV}"
          + (f"  gpu={torch.cuda.get_device_name(0)}"
             if DEV.type == "cuda" and torch.cuda.is_available() else ""))
    if DEV.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda asked for, but torch.cuda is unavailable")
    if groups:
        print(f"\n[resume] {out_path} holds {sum(groups.values())} row(s)")
        print("  (groups are corpus / task / feature-spec fingerprint)")
        for g in sorted(groups, key=lambda t: (t[0], t[1], t[2])):
            print(f"  {g[0]:<8} {g[1]:<14} {g[2]:<9} {groups[g]:>5} rows")
        print("  This run's group is known only once the feature spec is")
        print("  resolved below; rows from another spec are NOT reused.")
        print("  Rows from another corpus, task or packaging are NOT reused.")

    print(f"\n[corpus] {args.corpus}  task={args.task}"
          + (f"  rate={args.ciciot_rate:g}  seed={args.ciciot_seed}"
             if args.corpus == "ciciot" else ""))
    df, frozen = load_corpus(args)

    print("\n[folds]")
    folds = ds.make_folds(df, n_clients=args.clients, n_folds=args.folds,
                          alpha=args.alpha, smote=True,
                          smote_cap=args.smote_cap,
                          subsample=args.subsample)
    del df
    print(f"  {len(folds)} folds, {args.clients} clients, "
          f"Dirichlet(alpha={args.alpha}), subsample={args.subsample}")

    declared_frozen = tuple(frozen)
    train_mode = args.train_spec or args.frozen_mode
    atk_mode = args.attack_spec or args.frozen_mode

    def frozen_for(mode: str) -> tuple:
        if mode == "declared":
            return declared_frozen
        X0 = folds[0].X_test
        bin_cols = []
        for i, name in enumerate(folds[0].feature_names):
            u = np.unique(X0[:, i])
            if u.size <= 2 and set(np.round(u, 6).tolist()) <= {0.0, 1.0}:
                bin_cols.append(name)
        return tuple(bin_cols)

    def spec_and_fp(mode: str, role: str):
        fz = frozen_for(mode)
        if mode == "binary-only":
            print(f"  {role} spec binary-only: {len(fz)} strictly "
                  f"{{0,1}} column(s) frozen, the rest perturbable")
            print("  (the permissive end of the choice, not a better answer "
                  "than the declared")
            print("   list -- it is the other end of the same assumption)")
        else:
            print(f"  {role} spec declared: the corpus's name-based list")
        sp = build_spec(folds[0].feature_names, fz)
        names = sorted(c for c, p in zip(folds[0].feature_names,
                                         sp.perturbable) if not p)
        # the hash input is unchanged from v16, so a matched run reproduces
        # v16's fingerprint exactly and its rows resume
        fp = hashlib.sha1(
            (mode + "||" + "|".join(names)).encode()).hexdigest()[:8]
        return sp, fp

    global SPEC_FP, TRAIN_FP, PAIR_FP
    train_spec, TRAIN_FP = spec_and_fp(train_mode, "train")
    if atk_mode == train_mode:
        spec, SPEC_FP = train_spec, TRAIN_FP
    else:
        spec, SPEC_FP = spec_and_fp(atk_mode, "attack")
    PAIR_FP = pair_fp(TRAIN_FP, SPEC_FP)
    mismatch = SPEC_FP != TRAIN_FP

    print(f"  train spec {TRAIN_FP} "
          f"({int(train_spec.perturbable.sum())}/{train_spec.d} perturbable)"
          f"   attack spec {SPEC_FP} "
          f"({int(spec.perturbable.sum())}/{spec.d} perturbable)")
    print(f"  resume group spec = {PAIR_FP}"
          + ("   THREAT-MODEL MISMATCH: hardened against one set, attacked "
             "under another" if mismatch else "   (matched)"))
    print("  These are declared assumptions, not properties of the data. "
          "Rows carry both;")
    print("  changing either changes the resume key, so nothing is silently "
          "reused.")

    if mismatch:
        # D and A do no adversarial training, so their models cannot depend
        # on the train spec: a mismatch run would recompute exactly the
        # matched attack-spec run under a different key and write rows that
        # look new and are not.
        no_adv = [a for a in args.arms if a in ("D", "A")]
        if no_adv:
            print(f"  refusing arms {no_adv} under a mismatch: they train "
                  "without adversarial\n  examples, so train spec cannot "
                  f"reach them and these rows would duplicate the\n  matched "
                  f"{atk_mode} run under a key that hides it.")
            args.arms = [a for a in args.arms if a not in ("D", "A")]
        if not args.arms:
            raise SystemExit("  nothing left to run: a mismatch is only "
                             "meaningful for an arm that\n  trains "
                             "adversarially (B or C).")

    global this_group
    this_group = (args.corpus, _norm_task(args.corpus, args.task), PAIR_FP)
    if this_group in groups:
        print(f"  resuming: {groups[this_group]} row(s) already match this "
              "corpus, task and spec pair")
    else:
        print("  no rows yet for this corpus, task and spec pair; every cell "
              "will be computed")
    _ADV["spec"] = train_spec
    _ADV["eps"] = args.adv_eps
    _ADV["steps"] = args.adv_steps

    ref = g2_reference(args, folds[0].classes)
    prev = G.g2_prevalence(folds[0].y_test, ref, tol=0.02)
    print(f"  G2 test prevalence"
          + (" (ASSERTED against the corpus)" if ref else " (reported only)")
          + ": " + ", ".join(f"{folds[0].classes[c]} {p:.4f}"
                             for c, p in sorted(prev['prevalence'].items())))

    commit = G.git_commit(str(HERE))
    summary = {}

    for arm in args.arms:
        acfg = ARMS[arm]
        print(f"\n{'=' * 66}\n  ARM {arm} -- {acfg['note']}\n{'=' * 66}")
        sweeps = {"fgsm": {}, "pgd": {}}   # atk -> fold -> {eps: f1}

        for fold in folds:
            _ADV["frac"] = acfg["adv_frac"]
            cfg = dict(rounds=args.rounds, gan=acfg["gan"],
                       aug_frac=acfg["aug_frac"], gan_steps=20 if acfg["gan"] else 0,
                       mask=True, keep_frac=0.6, trust=True, gm=True,
                       rule="mc3gram", max_batches=args.max_batches,
                       device=str(DEV))
            cache = Path(args.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            fp = model_fingerprint(arm, fold, args, acfg, DEV)
            ckpt = cache / f"{arm}_f{fold.fold}_{fp}.pt"
            meta = cache / f"{arm}_f{fold.fold}_{fp}.json"

            t0 = time.time()
            if ckpt.exists() and meta.exists() and not args.no_cache:
                model = MC3GRAM(fold.X_val.shape[1], len(fold.classes))
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model = model.to(DEV)
                model.eval()
                res = json.loads(meta.read_text(encoding="utf-8"))
                secs = time.time() - t0
                print(f"  fold {fold.fold}: cached [{fp}], clean macro-F1 "
                      f"{res['f1']:.4f}  acc {res['accuracy']:.4f}")
            else:
                model, res = fed.run_federated(fold, cfg, log=lambda *a: None)
                _ADV["frac"] = 0.0      # never attack-train during evaluation
                secs = time.time() - t0
                torch.save({k: v.cpu() for k, v in
                            model.state_dict().items()}, ckpt)
                meta.write_text(json.dumps(res), encoding="utf-8")
                print(f"  fold {fold.fold}: clean macro-F1 {res['f1']:.4f}  "
                      f"acc {res['accuracy']:.4f}  ({secs:.0f}s, cached [{fp}])")
            _ADV["frac"] = 0.0

            # the rows the attacks are evaluated on
            if args.attack_rows and args.attack_rows < len(fold.X_test):
                sel = np.random.default_rng(1000 + fold.fold).choice(
                    len(fold.X_test), args.attack_rows, replace=False)
                Xatk, yatk = fold.X_test[sel], fold.y_test[sel]
            else:
                Xatk, yatk = fold.X_test, fold.y_test
            sup = np.bincount(yatk, minlength=len(fold.classes))
            thin = [fold.classes[c] for c in range(len(sup)) if 0 < sup[c] < 30]
            if thin and fold.fold == folds[0].fold:
                print(f"    attacks on {len(Xatk):,} of {len(fold.X_test):,} "
                      f"test rows; thin support (<30): "
                      + ", ".join(f"{c}={sup[fold.classes.index(c)]}"
                                  for c in thin))

            base = dict(version=VERSION, protocol=PROTOCOL,
                        corpus=args.corpus, task=args.task,
                        ciciot_rate=(args.ciciot_rate
                                     if args.corpus == "ciciot" else None),
                        arm=arm, feature_set="full",
                        spec_fingerprint=SPEC_FP,
                        train_spec_fingerprint=TRAIN_FP,
                        spec_pair=PAIR_FP,
                        spec_mode=f"{train_mode}>{atk_mode}",
                        n_perturbable=int(spec.perturbable.sum()),
                        n_perturbable_train=int(train_spec.perturbable.sum()),
                        n_features=int(spec.d), fold=fold.fold,
                        seed=fold.seed, commit=commit, is_baseline=False,
                        n_clients=args.clients, rounds=args.rounds,
                        subsample=args.subsample, spec=spec.name,
                        device=str(DEV), fingerprint=fp,
                        test_rows=int(len(fold.X_test)),
                        attack_rows=int(len(Xatk)),
                        thin_support={c: int(sup[fold.classes.index(c)])
                                      for c in thin})

            if run_key(args, arm, fold.fold, "clean", 0.0) not in have:
                G.g1_append_row(str(out_path), dict(
                    base, attack="clean", eps=0.0, metrics=res,
                    prevalence=prev["prevalence"], secs=round(secs, 1)))
            for atk in ("fgsm", "pgd"):
                sweeps[atk].setdefault(fold.fold, {})[0.0] = res["f1"]

            for atk in (() if args.cw_only else ("fgsm", "pgd")):
                for eps in EPS_GRID:
                    if run_key(args, arm, fold.fold, atk, eps) in have:
                        continue
                    Xa, ya, st = build_eval_set(
                        Xatk, yatk, eps, model, spec, attack=atk)
                    m = fed.evaluate(model, Xa, ya)
                    G.g4_identities({"per_class": [], **m})
                    print(f"    {atk:5s} eps={eps:<5} macro-F1 {m['f1']:.4f}")
                    sweeps[atk][fold.fold][eps] = m["f1"]
                    G.g1_append_row(str(out_path), dict(
                        base, attack=atk, eps=eps, metrics=m, attack_detail=st))

            # eps=-1.0, matching what the CW row is WRITTEN with below. v11
            # and v12 checked 0.0 here, so the key never matched: CW was
            # recomputed on every run and a duplicate row appended each time.
            if run_key(args, arm, fold.fold, "cw", -1.0) not in have:
                # CW only on rows the model gets RIGHT on clean input. A row
                # already misclassified counts as fooled at zero distance and
                # reports the error rate as a margin.
                with torch.no_grad():
                    pred = model(torch.from_numpy(Xatk).to(DEV)
                                 ).argmax(1).cpu().numpy()
                correct = np.where(pred == yatk)[0]
                if len(correct) < 20:
                    print(f"    cw    skipped: only {len(correct)} correctly "
                          "classified rows to attack")
                    cw = {"skipped": True, "n_correct": int(len(correct))}
                else:
                    idx = np.random.default_rng(fold.seed).choice(
                        correct, min(args.cw_n, len(correct)), replace=False)
                    _, cw = cw_l2(model, Xatk[idx], yatk[idx],
                                  spec, steps=args.cw_steps,
                                  binary_steps=args.cw_binary)
                    cw["n_correct_pool"] = int(len(correct))
                    cw["n_flipped"] = int(round(cw["success_rate"] * len(idx)))
                    print(f"    cw    success {cw['success_rate']:.3f}  "
                          f"(on {len(idx)} correctly-classified rows)")
                    print("          within L2: " + "  ".join(
                        f"{k}:{v:.3f}" for k, v in cw["success_at_l2"].items()))
                G.g1_append_row(str(out_path), dict(
                    base, attack="cw", eps=-1.0, metrics={}, cw=cw))

        # ---- G5, per attack, pooled over folds ----------------------------
        print(f"\n  [G5] arm {arm}")
        for atk in ("fgsm", "pgd"):
            per_fold = sweeps[atk]
            if not per_fold:
                continue
            eps_list = sorted({e for d in per_fold.values() for e in d})
            mean_pts = [(e, float(np.mean([d[e] for d in per_fold.values()
                                           if e in d]))) for e in eps_list]
            # retention = F1(eps)/F1(0), per fold, then mean and spread.
            # This is the robustness quantity. A level difference that is the
            # same at eps=0 as at eps=0.2 is a clean-accuracy difference.
            ret = {}
            for e in eps_list:
                vals = [d[e] / d[0.0] for d in per_fold.values()
                        if e in d and d.get(0.0)]
                if vals:
                    ret[e] = (float(np.mean(vals)), float(np.std(vals)))
            rec = {"mean_pts": mean_pts, "retention": ret,
                   "n_folds": len(per_fold)}
            if len(eps_list) < 2:
                print(f"    {atk:5s} no sweep in this run "
                      f"({len(eps_list)} budget) -- G5 needs at least two")
                summary[(arm, atk)] = rec
                continue
            try:
                r = G.g5_monotone(mean_pts, arm=arm, corpus=args.corpus,
                                  attack=atk)
                rec.update(r)
                print(f"    {atk:5s} ok, degradation {r['degradation']:.4f}  "
                      + "  ".join(f"{e:g}:{f:.4f}" for e, f in r["points"]))
                e_max = eps_list[-1]
                if e_max in ret:
                    print(f"          retention at eps={e_max:g}: "
                          f"{ret[e_max][0]:.4f} +/- {ret[e_max][1]:.4f}")
            except G.GateFailure as e:
                rec["failed"] = str(e)
                print(f"    {atk:5s} *** {e}")
            summary[(arm, atk)] = rec

    # ---- the comparison Phase 1 exists for -------------------------------
    print(f"\n{'=' * 66}\n  ARM C vs ARM B -- the claim under test\n{'=' * 66}")
    for atk in ("fgsm", "pgd"):
        b, c = summary.get(("B", atk)), summary.get(("C", atk))
        if not (b and c) or "failed" in b or "failed" in c:
            print(f"  {atk}: not available")
            continue
        print(f"  {atk}")
        print(f"    {'eps':<6}{'B F1':>8}{'C F1':>8}{'C-B':>9}   "
              f"{'B keep':>8}{'C keep':>8}{'keep diff':>11}")
        for (e, fb), (_, fc) in zip(b["mean_pts"], c["mean_pts"]):
            rb, sb = b["retention"].get(e, (float("nan"), 0.0))
            rc, sc = c["retention"].get(e, (float("nan"), 0.0))
            pooled = (sb ** 2 + sc ** 2) ** 0.5
            flag = ""
            if e > 0 and abs(rc - rb) > pooled and pooled > 0:
                flag = "  *" if rc > rb else "  (B)"
            print(f"    {e:<6g}{fb:>8.4f}{fc:>8.4f}{fc - fb:>+9.4f}   "
                  f"{rb:>8.4f}{rc:>8.4f}{rc - rb:>+11.4f}{flag}")
        print(f"    B degradation {b.get('degradation', float('nan')):.4f}   "
              f"C degradation {c.get('degradation', float('nan')):.4f}")
        print("    The claim is about RETENTION, not level. A C-B level gap "
              "that is the same at")
        print("    eps=0 as at the top budget is clean accuracy carried "
              "through, not robustness.")
        print("    '*' marks a retention gap wider than the pooled fold "
              "spread; '(B)' means B held up better.")

    # ---- completeness: every cell this run should have produced -----------
    print(f"\n{'=' * 66}\n  COMPLETENESS -- {this_group[0]}/{this_group[1]}/"
          f"{this_group[2]}\n{'=' * 66}")
    after, _ = done_keys(out_path)
    want = []
    for arm in args.arms:
        for fold in folds:
            want.append((arm, fold.fold, "clean", 0.0))
            want.append((arm, fold.fold, "cw", -1.0))
            if not args.cw_only:
                for atk in ("fgsm", "pgd"):
                    for eps in EPS_GRID:
                        want.append((arm, fold.fold, atk, eps))
    missing = [w for w in want
               if run_key(args, w[0], w[1], w[2], w[3]) not in after]
    print(f"  {len(want) - len(missing)} of {len(want)} cells present")
    if missing:
        by_arm: dict[str, int] = {}
        for a, *_ in missing:
            by_arm[a] = by_arm.get(a, 0) + 1
        print("  MISSING: " + ", ".join(f"arm {a}: {n}"
                                        for a, n in sorted(by_arm.items())))
        for w in missing[:10]:
            print(f"    arm {w[0]} fold {w[1]} {w[2]} eps={w[3]:g}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print("  Re-run the same command; only the missing cells are computed.")
    else:
        print("  complete.")

    print(f"\nrows in {out_path}. Regenerate tables from that file alone.")


if __name__ == "__main__":
    main()
