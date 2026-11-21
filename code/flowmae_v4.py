#!/usr/bin/env python3
"""
flowmae_v4.py -- flowmae_v3, with inference chunked so CICIDS fits in 8 GB.

WHAT WENT WRONG
---------------
The CICIDS run died twice with

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.34 GiB.
    GPU 0 has a total capacity of 7.93 GiB

run_flowmae_v7 answered that with --attack-bs, which caps the batch inside
attack_suite_v4. The second failure has the same 3.34 GiB in it, and its
traceback ends in

    torch._transformer_encoder_layer_fwd

That fused kernel is only taken when the module is in eval mode AND grad is
off, so the allocation is not in the attacks at all and not in training. It
is in fed.evaluate, which runs the clean test set at bs=4096 (fed.py:63) and
is called for the test set, the balanced test set and every round's
validation. On UNSW, 60 columns at that batch fits. On CICIDS it does not:
attention is quadratic in the number of columns, and 4096 rows of a wider
table is where the 3.34 GiB comes from.

WHAT THIS CHANGES
-----------------
FlowMAE.forward processes at most EVAL_CHUNK rows at a time WHEN GRAD IS
OFF, and is untouched otherwise.

Why that is free of consequences for the numbers: the model has no operation
that mixes rows. Normalisation is LayerNorm, which is per row -- this is the
same property that made the architecture safe under FedAvg in the first place
-- attention runs over the column tokens within a row, and the classification
head is per row. Splitting a batch therefore returns exactly what the whole
batch returns, bit for bit on the same hardware. Nothing about the comparison
with the CICIDS MC3GRAM baseline moves, which is the point: --attack-rows
stays at 200000, and only how many rows go through the GPU at once changes.

Chunking is deliberately NOT applied when grad is on. Under autograd every
chunk's activations are kept alive until the backward pass, so chunking there
saves no memory while making the graph larger. Training and the attacks keep
their own batch sizes.

USAGE
  Nothing to call. run_flowmae_v8.py imports this instead of flowmae_v3.
  The chunk is 512 rows and can be changed without editing anything:

      set FLOWMAE_EVAL_CHUNK=256
"""

from __future__ import annotations

import os
import sys

import torch

import flowmae_v3 as _v3

# re-export flowmae_v3 wholesale, private helpers included, so that this
# module is a drop-in for it
_self = sys.modules[__name__]
for _k, _v in vars(_v3).items():
    if not _k.startswith("__"):
        setattr(_self, _k, _v)

EVAL_CHUNK = int(os.environ.get("FLOWMAE_EVAL_CHUNK", "512"))

_ORIG_FORWARD = _v3.FlowMAE.forward
_PATCH_MARK = "_flowmae_v4_chunked"


def _cat_like(parts):
    """Concatenate chunk outputs, whatever shape forward returned."""
    first = parts[0]
    if torch.is_tensor(first):
        return torch.cat(parts, dim=0)
    if isinstance(first, (tuple, list)):
        out = [ _cat_like([p[i] for p in parts]) for i in range(len(first)) ]
        return type(first)(out)
    if isinstance(first, dict):
        return {k: _cat_like([p[k] for p in parts]) for k in first}
    raise TypeError(f"forward returned {type(first)!r}, which this cannot "
                    f"split; set FLOWMAE_EVAL_CHUNK=0 to disable chunking")


def _chunked_forward(self, x, *a, **kw):
    n = EVAL_CHUNK
    if (n <= 0 or torch.is_grad_enabled() or not torch.is_tensor(x)
            or x.shape[0] <= n):
        return _ORIG_FORWARD(self, x, *a, **kw)

    # Any other positional or keyword tensor with the same leading dimension
    # is a per-row argument (a column mask, say) and has to be cut with x.
    rows = x.shape[0]

    def _slice(v, lo, hi):
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == rows:
            return v[lo:hi]
        return v

    parts = []
    for lo in range(0, rows, n):
        hi = min(lo + n, rows)
        aa = tuple(_slice(v, lo, hi) for v in a)
        kk = {k: _slice(v, lo, hi) for k, v in kw.items()}
        parts.append(_ORIG_FORWARD(self, x[lo:hi], *aa, **kk))
    return _cat_like(parts)


if not getattr(_v3.FlowMAE.forward, _PATCH_MARK, False):
    setattr(_chunked_forward, _PATCH_MARK, True)
    _v3.FlowMAE.forward = _chunked_forward
    FlowMAE = _v3.FlowMAE

print(f"  flowmae_v4: inference chunked at {EVAL_CHUNK} rows "
      f"(training and attacks untouched)", flush=True)
