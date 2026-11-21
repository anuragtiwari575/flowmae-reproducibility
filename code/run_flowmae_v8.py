#!/usr/bin/env python3
"""
run_flowmae_v8.py -- run FlowMAE through the project's existing protocol.

v8 CHANGES EXACTLY ONE THING: it imports flowmae_v4 instead of flowmae_v3.
flowmae_v4 is flowmae_v3 with inference chunked, which is what the CICIDS
out-of-memory failure actually needed -- the 3.34 GiB allocation was in
fed.evaluate at bs=4096, not in the attacks, so v7's --attack-bs could not
reach it. TRAINER_VERSION stays 2 on purpose: the training loop is byte for
byte the one v7 ran, so every cached UNSW checkpoint keeps its fingerprint
and is still valid. --attack-bs is still here and still useful.


WHAT THIS DOES AND WHY IT IS A WRAPPER
--------------------------------------
fed.run_federated builds MC3GRAM(d_in, n_cls, heads=heads).to(dev) at two
places and resolves that name as a module global at call time. Rebinding
fed.MC3GRAM therefore swaps the architecture while the federated loop, the
aggregation rules, the mask consensus, the attack suite, the five folds and
gates G1-G10 all run unchanged -- the same mechanism phase1_arms_v17 already
uses for fed.local_train. Nothing in the repository is edited, so every number
produced here is comparable, cell for cell, with the numbers already measured.

ARMS
----
Two new arms are registered rather than reusing D and B, because row_key is
(corpus, task, packaging, arm, fold, attack, eps) and does NOT carry the model:
reusing D and B would let a FlowMAE row and an MC3GRAM row collide on resume.

  F    FlowMAE, no adversarial training          -- the counterpart of arm D
  FB   FlowMAE + PGD adversarial training        -- the counterpart of arm B

The generator arms (A, C) are not offered. The generator was measured null on
clean detection at two round budgets and on three declaration pairs, so
carrying it into a new architecture would add a component this project has
already shown does nothing. Asking for A or C here fails loudly rather than
running something that looks like an answer.

THE TRAINING TERMS
------------------
Local training is reimplemented here rather than wrapped, because the two new
terms belong inside the optimiser step and a wrapper cannot reach it. The
adversarial branch calls phase1_arms_v17._pgd_batch, so it is the same PGD,
against the same declared perturbable set, as every number already in hand.

    total = CE(f(x_mixed), y)
          + mcp_lambda * masked-column reconstruction loss
          + gdr_lambda * (negative entropy of the input-gradient mass)

With mcp_lambda = gdr_lambda = col_dropout = 0, arm FB is plain PGD adversarial
training of FlowMAE and arm F is plain training of it, which is what makes the
architecture contrast clean.

THE TRAINER GATE
----------------
--no-flowmae keeps MC3GRAM and changes only the trainer. Run it with every new
term at zero and arms D/B, and the rows must reproduce the existing arm D/B
rows for the same spec. If they do not, this file's training loop differs from
the repository's and nothing downstream is interpretable. maskgate_v1.py
compares the two files; that is what it is for.

  python -u run_flowmae_v8.py --no-flowmae --corpus unsw --device cuda \
      --attack-rows 200000 --train-spec binary-only --attack-spec binary-only \
      --arms B --cache-dir out/models_trainergate \
      --out out/trainergate_unsw.jsonl
  python -u maskgate_v1.py --new out/trainergate_unsw.jsonl \
      --ref out/phase13_unsw_permissive.jsonl --new-arm B --ref-arm B

FINGERPRINTS
------------
Every FlowMAE hyperparameter and every loss weight goes into the model
fingerprint, so two configurations can never share a checkpoint. Arms outside
this file keep v17's fingerprint byte-for-byte and their caches still resume.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

import flowmae_v4 as fm

TRAINER_VERSION = 2
MODEL_VERSION = 2

FLOW_ARMS = {
    "F": dict(gan=False, aug_frac=0.0, adv_frac=0.0,
              note="FlowMAE, no adversarial training (counterpart of D)"),
    "FB": dict(gan=False, aug_frac=0.0, adv_frac=0.5,
               note="FlowMAE + PGD adversarial training (counterpart of B)"),
    # v3: masked-column pretraining gets its OWN arm names, with identical
    # switch settings to F and FB. The reason is the analyser, not the driver:
    # a row's arm is what analyze_v6 groups by, so a pretrained run written
    # under arm "F" would merge with a non-pretrained run under arm "F" into
    # one ten-fold pseudo-arm the moment both files are analysed together,
    # and the contrast would silently disappear into its own average. Distinct
    # names also give the two variants distinct model fingerprints, so their
    # checkpoints cannot share a cache entry.
    "FM": dict(gan=False, aug_frac=0.0, adv_frac=0.0,
               note="FlowMAE + masked-column pretraining, no adv. training"),
    "FMB": dict(gan=False, aug_frac=0.0, adv_frac=0.5,
                note="FlowMAE + masked-column pretraining + adv. training"),
}

MCP_ARMS = ("FM", "FMB")
PLAIN_ARMS = ("F", "FB")

CFG = {
    "flowmae": True,
    "d_model": 64, "n_layers": 3, "n_head": 4, "p_drop": 0.1,
    "col_dropout": 0.0, "gdr_lambda": 0.0,
    "mcp_lambda": 0.0, "mcp_frac": 0.3,
    "mcp_init": None,
}

_ORIG_LOCAL_TRAIN = None    # fed's own, captured before anything is rebound
_DRV = None                 # phase1_arms_v17, for its _pgd_batch and _ADV
_REPORTED: set = set()
_FOLD = {"n": None}         # set by the run_federated wrapper, read by the
                            # factory so a per-fold MCP checkpoint is loaded


# ------------------------------------------------------------ model factory
def _mcp_path() -> str | None:
    """The checkpoint for the fold now being trained.

    A pretrained encoder MUST be per fold. Pretraining once on fold 0's
    training rows and then evaluating on folds 1-4 leaks, because a row that
    is training data in fold 0 is test data in another fold. So --mcp-init
    takes a template containing {fold}; a plain path is accepted only when a
    single fold is being run, and says so loudly.
    """
    p = CFG["mcp_init"]
    if not p:
        return None
    if "{fold}" in p:
        if _FOLD["n"] is None:
            raise SystemExit("run_flowmae_v8: --mcp-init has {fold} but the "
                             "fold is unknown; the run_federated wrapper did "
                             "not take effect.")
        return p.format(fold=_FOLD["n"])
    if _FOLD["n"] not in (None, 0):
        raise SystemExit(
            f"run_flowmae_v8: --mcp-init is a single path but fold "
            f"{_FOLD['n']} is being trained. One checkpoint across folds "
            f"leaks test rows into pretraining. Pass a template such as "
            f"out/mcp_unsw_f{{fold}}.pt and pretrain each fold.")
    return p


def _make_model(d_in, n_cls, heads=("fpn", "mlp", "trf")):
    m = fm.FlowMAE(d_in, n_cls, heads=heads,
                   d_model=CFG["d_model"], n_layers=CFG["n_layers"],
                   n_head=CFG["n_head"], p_drop=CFG["p_drop"],
                   col_dropout=CFG["col_dropout"])
    path = _mcp_path()
    if path:
        sd = torch.load(path, map_location="cpu")
        enc = sd.get("encoder", sd)
        missing, unexpected = m.encoder.load_state_dict(enc, strict=False)
        if path not in _REPORTED:
            _REPORTED.add(path)
            print(f"  [mcp] encoder initialised from {path}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected key(s)",
                  flush=True)
            if missing:
                print(f"  [mcp] missing: {sorted(missing)[:6]}", flush=True)
            if len(missing) == len(m.encoder.state_dict()):
                raise SystemExit("run_flowmae_v8: the checkpoint matched NO "
                                 "encoder key. Its architecture differs from "
                                 "this run's --fm-* settings.")
    return m


# ----------------------------------------------------------- local training
def local_train_flowmae(model, X, y, epochs=1, lr=1e-3, bs=512, gen=None,
                        mask=None, aug_frac=0.0, n_cls=10, max_batches=None):
    """One client's local update: CE, plus the adversarial branch, plus the
    two new terms, in a single optimiser step."""
    if gen is not None and aug_frac > 0:
        raise SystemExit(
            "run_flowmae_v8: the generator arms (A, C) are not supported. "
            "The generator measured null on clean detection at two round "
            "budgets and on three declaration pairs; use arms F and FB.")

    drv = _DRV
    adv_frac = float(drv._ADV["frac"]) if drv is not None else 0.0
    spec = drv._ADV["spec"] if drv is not None else None

    dv = next(model.parameters()).device
    pmask = None
    if adv_frac > 0:
        if spec is None:
            raise SystemExit("run_flowmae_v8: adversarial training was asked "
                             "for but no perturbable-set spec is set.")
        pmask = torch.from_numpy(
            spec.perturbable.astype(np.float32)).to(dv).unsqueeze(0)

    gdr_l = float(CFG["gdr_lambda"])
    mcp_l = float(CFG["mcp_lambda"])
    is_flowmae = isinstance(model, fm.FlowMAE)
    if (mcp_l > 0 or CFG["col_dropout"] > 0) and not is_flowmae:
        raise SystemExit("run_flowmae_v8: --mcp-lambda and --col-dropout need "
                         "FlowMAE; they have no meaning for MC3GRAM.")

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    Xt = torch.from_numpy(X).to(dv)
    yt = torch.from_numpy(y).to(dv)

    for _ in range(epochs):
        perm = torch.randperm(n)
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            if xb.size(0) < 2:
                continue

            x_in, y_in = xb, yb
            if adv_frac > 0:
                m = max(1, int(adv_frac * xb.size(0)))
                # v2: CPU generator, not device=dv. v17 draws this on the CPU,
                # and drawing it on the GPU instead does not change the
                # algorithm but does change the random realisation twice over:
                # a different subset of rows is attacked, AND the draw is taken
                # from the CUDA stream rather than the CPU one, which shifts
                # every later draw in BOTH streams -- including the random
                # start inside _pgd_batch. The v1 trainer gate came back with a
                # mean signed difference of -0.0024 (no systematic effect) but
                # a worst cell of -0.0240, which is what two different
                # realisations of the same procedure look like. Matching v17
                # here removes the last difference between the two loops.
                sel = torch.randperm(xb.size(0))[:m]
                x_adv = drv._pgd_batch(model, xb[sel], yb[sel],
                                       drv._ADV["eps"], drv._ADV["steps"],
                                       pmask)
                x_in = torch.cat([xb, x_adv], 0)
                y_in = torch.cat([yb, yb[sel]], 0)

            opt.zero_grad()
            loss = F.cross_entropy(model(x_in), y_in)
            if mcp_l > 0:
                # the reconstruction target is the CLEAN batch: asking the
                # model to reconstruct an adversarial row would teach it to
                # predict the perturbation, which is not the objective.
                loss = loss + mcp_l * model.mcp_loss(xb, CFG["mcp_frac"])
            if gdr_l > 0:
                loss = loss + gdr_l * fm.gradient_dispersion_penalty(
                    model, xb, yb)
            loss.backward()
            opt.step()

            nb += 1
            if max_batches and nb >= max_batches:
                break
    return model


# --------------------------------------------------------------------- main
def main() -> int:
    global _ORIG_LOCAL_TRAIN, _DRV

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--no-flowmae", action="store_true",
                     help="keep MC3GRAM and change only the trainer. This is "
                          "the trainer gate; see the header.")
    pre.add_argument("--fm-dmodel", type=int, default=64)
    pre.add_argument("--fm-layers", type=int, default=3)
    pre.add_argument("--fm-nhead", type=int, default=4)
    pre.add_argument("--fm-dropout", type=float, default=0.1)
    pre.add_argument("--col-dropout", type=float, default=0.0,
                     help="per-column dropout during supervised training: the "
                          "cheap surrogate for GDR, no second backward.")
    pre.add_argument("--gdr-lambda", type=float, default=0.0,
                     help="weight on the gradient-dispersion penalty. Needs a "
                          "second backward; 0 costs nothing.")
    pre.add_argument("--mcp-lambda", type=float, default=0.0,
                     help="weight on masked-column reconstruction, trained "
                          "jointly with the classifier.")
    pre.add_argument("--mcp-frac", type=float, default=0.3,
                     help="fraction of columns masked per row for MCP.")
    pre.add_argument("--attack-bs", type=int, default=None,
                     help="rows per batch inside the attacks. The suite uses "
                          "4096, which on CICIDS needs more memory than an "
                          "8 GB card has once each column is a token. Lower "
                          "this rather than --attack-rows: the row count is "
                          "part of what makes a comparison matched, the batch "
                          "size is not.")
    pre.add_argument("--mcp-init", default=None,
                     help="checkpoint from pretrain_mcp_v1.py to initialise "
                          "the encoder from.")
    mine, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    CFG.update(flowmae=not mine.no_flowmae,
               d_model=mine.fm_dmodel, n_layers=mine.fm_layers,
               n_head=mine.fm_nhead, p_drop=mine.fm_dropout,
               col_dropout=mine.col_dropout, gdr_lambda=mine.gdr_lambda,
               mcp_lambda=mine.mcp_lambda, mcp_frac=mine.mcp_frac,
               mcp_init=mine.mcp_init)

    if mine.attack_bs:
        # Patch the attack module itself, not the driver's imported names:
        # build_eval_set calls pgd and fgsm through attack_suite_v4's own
        # namespace, so rebinding them in the driver would have no effect.
        import functools
        import attack_suite_v4 as asuite
        for _n in ("fgsm", "pgd", "cw_l2"):
            if hasattr(asuite, _n):
                setattr(asuite, _n,
                        functools.partial(getattr(asuite, _n),
                                          bs=mine.attack_bs))
        print(f"  attack batch size forced to {mine.attack_bs}", flush=True)

    import fed
    _ORIG_LOCAL_TRAIN = fed.local_train

    import phase1_arms_v17 as drv
    _DRV = drv

    if not hasattr(drv, "_pgd_batch") or not hasattr(drv, "_ADV"):
        raise SystemExit("run_flowmae_v8: phase1_arms_v17 does not expose "
                         "_pgd_batch/_ADV; the driver changed and this "
                         "wrapper is stale.")

    drv.ARMS.update(FLOW_ARMS)

    if CFG["flowmae"]:
        fed.MC3GRAM = _make_model
        if hasattr(drv, "MC3GRAM"):
            drv.MC3GRAM = _make_model

    # replaces v17's local_train_with_adv: the adversarial branch is
    # reproduced here by calling v17's own _pgd_batch, and the two new terms
    # are added in the same optimiser step.
    fed.local_train = local_train_flowmae

    # The model factory needs the fold number to pick the right per-fold MCP
    # checkpoint, and only run_federated knows it. Wrapping it is the smallest
    # place to record it and changes nothing else.
    _orig_run = fed.run_federated

    def run_federated(fold, cfg, log=print):
        _FOLD["n"] = int(fold.fold)
        try:
            return _orig_run(fold, cfg, log=log)
        finally:
            _FOLD["n"] = None

    fed.run_federated = run_federated
    if getattr(drv, "fed", None) is not None and drv.fed is not fed:
        raise SystemExit("run_flowmae_v8: the driver imported a different fed "
                         "module; the rebinds would not take effect.")

    _orig_fp = drv.model_fingerprint

    def model_fingerprint(arm, fold, args, acfg, dev):
        # v5: record the fold HERE, not only in the run_federated wrapper.
        # A fold whose checkpoint is already cached never goes through
        # run_federated at all -- the driver builds the model outside it to
        # load the state dict into -- so the wrapper left _FOLD unset and the
        # {fold} template had nothing to fill. This hook runs on both paths,
        # because the fingerprint is what names the cache file and so must be
        # computed before any model exists.
        _FOLD["n"] = int(getattr(fold, "fold", fold))

        # v3: the arm name and the pretraining flags must agree. This hook is
        # the first place the arm is known, and it is called for every arm and
        # fold, so a disagreement is caught before any training happens. An
        # FM/FMB row with no pretrained encoder, or an F/FB row WITH one,
        # would be mislabelled in the output file and no later check would
        # notice: the numbers would be internally consistent and simply
        # attributed to the wrong condition.
        mcp_on = bool(CFG["mcp_init"]) or CFG["mcp_lambda"] > 0
        if arm in MCP_ARMS and not mcp_on:
            raise SystemExit(
                f"run_flowmae_v8: arm {arm} means masked-column pretraining, "
                f"but neither --mcp-init nor --mcp-lambda is set. Use arm "
                f"{'F' if arm == 'FM' else 'FB'} for a run without it.")
        if arm in PLAIN_ARMS and mcp_on:
            raise SystemExit(
                f"run_flowmae_v8: arm {arm} means NO pretraining, but "
                f"--mcp-init/--mcp-lambda is set. Use arm "
                f"{'FM' if arm == 'F' else 'FMB'} instead, so the rows carry "
                f"the condition they were produced under.")

        base = _orig_fp(arm, fold, args, acfg, dev)
        # TRAINER_VERSION is in the hash because this file's training loop
        # decides what a checkpoint contains. v1 drew the adversarial row
        # selection on the CUDA generator and v2 draws it on the CPU one, as
        # v17 does; without this token a v1 checkpoint would be reused under
        # v2 and the correction would have no visible effect. Bump it with
        # any change to local_train_flowmae.
        # MODEL_VERSION enters the hash ONLY when the masked-token path is
        # actually used. flowmae_v2 changed that path (a positional embedding
        # on masked slots, a per-column readout) and nothing else, so with no
        # masking the forward pass is byte-identical to v1 and the arm F / FB
        # checkpoints already on disk are still the right ones. Adding the
        # token unconditionally would retrain them for no reason; leaving it
        # out when masking IS used would reuse a v1 checkpoint under v2.
        mv = (f"|mv={MODEL_VERSION}"
              if (mcp_on or CFG["col_dropout"] > 0) else "")
        s = (f"{base}|tv={TRAINER_VERSION}{mv}"
             f"|flowmae={CFG['flowmae']}|dm={CFG['d_model']}"
             f"|L={CFG['n_layers']}|H={CFG['n_head']}|pd={CFG['p_drop']}"
             f"|cd={CFG['col_dropout']}|gdr={CFG['gdr_lambda']}"
             f"|mcp={CFG['mcp_lambda']}|mcpf={CFG['mcp_frac']}"
             f"|init={CFG['mcp_init']}")
        return hashlib.sha1(s.encode()).hexdigest()[:10]

    drv.model_fingerprint = model_fingerprint

    print("=" * 70)
    print("FlowMAE  (run_flowmae_v8 over phase1_arms_v17)")
    print(f"  architecture : "
          + ("FlowMAE per-column transformer" if CFG["flowmae"]
             else "MC3GRAM (trainer gate: architecture unchanged)"))
    if CFG["flowmae"]:
        print(f"  d_model={CFG['d_model']} layers={CFG['n_layers']} "
              f"heads={CFG['n_head']} dropout={CFG['p_drop']}")
    print(f"  col_dropout  : {CFG['col_dropout']}")
    print(f"  gdr_lambda   : {CFG['gdr_lambda']}"
          + ("  (second backward ON)" if CFG["gdr_lambda"] > 0 else ""))
    print(f"  mcp_lambda   : {CFG['mcp_lambda']}  mask_frac={CFG['mcp_frac']}")
    print(f"  mcp_init     : {CFG['mcp_init']}")
    print("  arms F and FB added; the generator arms A and C are refused")
    print("  EVERY setting above is in the model fingerprint")
    print("=" * 70, flush=True)

    return drv.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
