#!/usr/bin/env python3
"""
seedrun_v1.py -- repeat a run under a different training seed, with the
folds and client partitions held fixed.

WHY THIS IS A WRAPPER AND NOT A FLAG
------------------------------------
The review's point is that five folds are not five independent
replications: they are partitions of one corpus with overlapping
training sets, so a paired t-test over them is optimistic. The fix is
repetition over seeds with the partitions FIXED, which separates
run-to-run variance from fold variance.

phase1_arms_v17 has no --seed option, and adding one to it is not
something this wrapper should do silently, because of the second
problem: the model cache is keyed by a fingerprint, and if the seed is
not in that fingerprint then the second seed loads the first seed's
checkpoints and reports a variance of exactly zero. That failure is
silent and would look like a result.

So this wrapper does two things and nothing else:

  1. seeds torch, numpy and the standard library before the run, which
     changes initialisation, batch order and the adversarial-example
     sampling, while leaving make_folds untouched -- it derives its
     partitions from fold-level seeds, so the folds and the client
     splits are identical across seeds by construction;
  2. appends the seed to the driver's model_fingerprint, so a run under
     a new seed cannot collide with a cached one.

Everything else is the driver's, unchanged.

USAGE
  python -u seedrun_v1.py --seed 1 -- \
      python_args_for_the_driver...

  e.g.
  python -u seedrun_v1.py --seed 1 --driver phase1_arms_v17 -- \
      --corpus unsw --device cuda --arms B --attack-rows 200000 \
      --train-spec binary-only --attack-spec binary-only \
      --cache-dir out/models_seed1 --out out/seed1_unsw_B.jsonl

  For FlowMAE arms use --driver run_flowmae_v8.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--driver", default="phase1_arms_v17",
                    help="phase1_arms_v17 or run_flowmae_v8")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="everything after -- goes to the driver")
    a = ap.parse_args()

    rest = a.rest
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        raise SystemExit("nothing to pass to the driver; put the "
                         "driver's arguments after --")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    drv = importlib.import_module(a.driver)
    base_fp = drv.model_fingerprint

    def fingerprint(*args, **kw):
        return f"{base_fp(*args, **kw)}s{a.seed}"

    drv.model_fingerprint = fingerprint

    # run_flowmae_v8 rebinds the driver's fingerprint itself, so patch
    # there too when that is the entry point
    if a.driver != "phase1_arms_v17":
        import phase1_arms_v17 as inner
        inner_fp = inner.model_fingerprint

        def inner_fingerprint(*args, **kw):
            out = inner_fp(*args, **kw)
            return out if out.endswith(f"s{a.seed}") else f"{out}s{a.seed}"

        inner.model_fingerprint = inner_fingerprint

    print("=" * 74)
    print(f"seedrun_v1: seed {a.seed}, driver {a.driver}")
    print("  folds and client partitions are UNCHANGED -- make_folds")
    print("  derives them from fold-level seeds, not from this one.")
    print(f"  model fingerprints carry the suffix s{a.seed}, so cached")
    print("  checkpoints from another seed cannot be reused.")
    print("=" * 74)

    sys.argv = [f"{a.driver}.py"] + rest
    return int(drv.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
