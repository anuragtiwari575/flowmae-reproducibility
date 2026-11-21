#!/usr/bin/env python3
"""
flowmae_v1.py -- FlowMAE: a per-column tokenised transformer for flow-based
intrusion detection, with masked-column pretraining and a gradient-dispersion
regulariser.

WHY THIS ARCHITECTURE, AND WHAT EACH PIECE IS FOR
-------------------------------------------------
Two measurements from this project motivate every component here, so none of
them is decoration.

  (a) Holding one trained detector's weights bit-identical and changing only
      which columns the threat model declares perturbable moves macro-F1
      retention at eps=0.2 by 0.384 to 0.866. Retraining the identical
      configuration moves it by 0.010.

  (b) Training against the worst k-sparse declaration at k=20 of a 38-column
      pool reproduces training against all 38 to within 0.007 retention and
      0.001 clean macro-F1. The adversarial loss is therefore carried by a
      minority of columns: a small set of coordinates holds nearly all of the
      gradient mass, and that concentration is what makes the reported figure
      a property of the declaration rather than of the model.

The response to (b) is not to fight the concentration at attack time -- that
was measured and it bought nothing -- but to stop it forming during training.
Three pieces do that.

  1. PER-COLUMN TOKENISATION. A flow's feature vector is not a sequence and
     not an image. The CNN-LSTM and ResNet-LSTM detectors in this literature
     reshape the vector and apply convolution or recurrence along an axis that
     carries no order, which is why their inductive bias is wrong for tabular
     heterogeneous features. Here each column gets its own embedding and
     attention runs between columns, so the model represents which columns
     inform which, and that relation is what masked pretraining can exploit.

  2. MASKED-COLUMN PRETRAINING (MCP). Self-supervised: mask a random subset of
     columns -- by replacing their tokens with a learned [MASK] embedding, not
     by zeroing the value, which a real zero would be confused with -- and
     reconstruct them from the rest. A representation that can infer any column
     from the others is redundant by construction, which is the property (a)
     and (b) say is missing. This differs from the autoencoders used in this
     literature, which compress the whole vector; nothing in a compression
     objective requires a column to be recoverable when it is absent.

  3. GRADIENT-DISPERSION REGULARISER (GDR). A penalty on the concentration of
     input-gradient mass: with p_j = |d loss / d x_j| normalised over columns,
     the regulariser maximises the entropy H(p). It targets (b) directly and
     is measured in its own right by the lambda sweep. It requires a
     double backward, so it is OFF by default and costs nothing at lambda 0.
     A cheap surrogate with no second backward -- per-column dropout during
     supervised training -- is also provided, and the two are separable in the
     ablation.

DROP-IN COMPATIBILITY
---------------------
fed.run_federated builds MC3GRAM(d_in, n_cls, heads=heads).to(dev) at two
places and resolves the name as a module global at call time, so a driver can
rebind fed.MC3GRAM to FlowMAE and the federated loop, the aggregation rules,
the attack suite and every gate run unchanged -- the same mechanism v17 already
uses for fed.local_train. FlowMAE therefore matches MC3GRAM's interface:

    FlowMAE(d_in, n_cls, heads=<tuple>)   same positional signature
    forward(x) -> logits                  same
    .feature_mask                         buffer of length d_in, applied to
                                          the input, copied into by the GMA
    .attribution(x) -> {name: (d_in,)}    per-view attribution for
                                          mask_consensus

`heads` is accepted for signature compatibility. FlowMAE has one trunk, so the
names are not used as separate heads; the number of entries sets how many
ablation VIEWS attribution() returns, which is what mask_consensus votes over.

WHAT IS NOT CLAIMED
-------------------
No temporal modelling. Attention here is over columns of one flow record, not
over time, and the paper must say so. A detector that reshapes a feature vector
and calls the result a sequence is making a claim it cannot support, and this
file does not repeat it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- tokenisation
class ColumnTokeniser(nn.Module):
    """One embedding per column: token_j = x_j * W_j + b_j.

    This is the numerical branch of FT-Transformer's feature tokeniser. Every
    column here is already min-max scaled to [0,1] by the loaders, and the
    one-hot expansions are columns in their own right, so a single numerical
    branch covers the whole vector and no categorical lookup is needed.

    A learned [MASK] embedding replaces a column's token when that column is
    masked. Masking the TOKEN rather than setting the value to zero matters:
    zero is a legal value for most of these columns, so a zeroed input says
    "this column is zero", while a [MASK] token says "this column is absent".
    Only the second is the pretraining task we want.

    v2 ADDS A POSITIONAL EMBEDDING TO THE MASKED SLOTS, AND IT IS NOT OPTIONAL
    -------------------------------------------------------------------------
    v1 replaced a masked column's token with the shared mask_token and nothing
    else. An unmasked column carries its identity in its own weight and bias,
    but a masked one then carried none: every masked slot in a row held the
    SAME vector, and attention is permutation-equivariant, so the encoder's
    output at every masked position of a row was identical and the shared
    readout gave them all one number. The model was being asked to predict the
    value of an unnamed column.

    Measured, on synthetic data with four latent factors driving twenty
    columns, against the column-mean predictor on the same masked positions:

        v1  (shared mask token, shared head)   1.011 x  -- LOSES to the mean
        + positional embedding on masked slots 0.816 x
        + per-column readout as well           0.694 x

    and with five columns masked in one row, the spread of v1's five
    predictions was 0.000000 -- exactly zero, to the printed precision.

    This is also what the UNSW pretraining run reported: best held-out MSE
    0.0757-0.0781 against a mean-predictor MSE of 0.0470-0.0483, a ratio of
    1.60 on all five folds. Not a training failure; the objective was not
    learnable as posed.

    The embedding is added ONLY where a column is masked. With no col_mask the
    forward pass is byte-identical to v1, so the arm F and arm FB results
    already measured remain valid as the no-pretraining baseline and the
    masked-column contrast stays clean. Adding it to every token instead
    reconstructs better still (0.572 x) but changes the classifier, which
    would mean re-running both baselines to keep the comparison honest.
    """

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.d_in, self.d_model = d_in, d_model
        self.weight = nn.Parameter(torch.empty(d_in, d_model))
        self.bias = nn.Parameter(torch.zeros(d_in, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        # "which column is absent" -- see the class docstring
        self.mask_pos = nn.Parameter(torch.zeros(d_in, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        bound = 1.0 / math.sqrt(d_model)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.mask_token, -bound, bound)
        nn.init.uniform_(self.mask_pos, -bound, bound)
        nn.init.uniform_(self.cls_token, -bound, bound)

    def forward(self, x: torch.Tensor,
                col_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, d_in) -> (B, 1 + d_in, d_model), CLS token first.

        col_mask: (B, d_in) or (1, d_in), 1 where the column is MASKED.
        """
        t = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if col_mask is not None:
            m = col_mask.unsqueeze(-1).to(t.dtype)
            mt = self.mask_token.view(1, 1, -1) + self.mask_pos.unsqueeze(0)
            t = t * (1.0 - m) + mt * m
        cls = self.cls_token.expand(t.size(0), -1, -1)
        return torch.cat([cls, t], dim=1)


# ------------------------------------------------------------------- encoder
class FlowEncoder(nn.Module):
    """Pre-norm transformer over column tokens.

    LayerNorm throughout and no BatchNorm anywhere, for the reason the
    repository's own Stem gives: under a Dirichlet(0.5) label skew each client
    sees a different class mix, so BatchNorm's running statistics diverge
    between clients and averaging them destroys the aggregate. LayerNorm
    carries no cross-sample statistics and aggregates cleanly under FedAvg.
    """

    def __init__(self, d_in: int, d_model: int = 64, n_layers: int = 3,
                 n_head: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.tok = ColumnTokeniser(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2,
            dropout=p_drop, activation="gelu", batch_first=True,
            norm_first=True)
        # enable_nested_tensor=False only silences a warning: with
        # norm_first=True PyTorch has already disabled that fast path, which
        # is what the warning says. No numerical effect, so checkpoints from
        # a build that emitted the warning stay valid.
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers,
                                         enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, col_mask=None):
        h = self.enc(self.tok(x, col_mask))
        return self.norm(h)          # (B, 1 + d_in, d_model)


# ------------------------------------------------------------- the full model
class FlowMAE(nn.Module):
    """Encoder + classification head + masked-column reconstruction head.

    Signature-compatible with model.MC3GRAM so that fed.run_federated can build
    it unchanged.
    """

    def __init__(self, d_in: int, n_cls: int,
                 heads: tuple[str, ...] = ("fpn", "mlp", "trf"),
                 d_model: int = 64, n_layers: int = 3, n_head: int = 4,
                 p_drop: float = 0.1, col_dropout: float = 0.0):
        super().__init__()
        self.d_in, self.n_cls = d_in, n_cls
        self.heads_used = tuple(heads)          # kept for interface parity
        self.n_views = max(1, len(heads))
        self.col_dropout = float(col_dropout)

        self.encoder = FlowEncoder(d_in, d_model, n_layers, n_head, p_drop)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_cls))
        # One scalar per column, read off that column's own output token --
        # but with a PER-COLUMN readout, not one shared linear layer. v1
        # shared it, which forced every column's reconstruction through the
        # same map and cost 0.694 -> 0.816 of the mean-predictor ratio in the
        # synthetic check. The readout touches only the pretraining path, so
        # the classifier is unaffected and the existing F/FB rows stand.
        self.recon_w = nn.Parameter(torch.empty(d_in, d_model))
        self.recon_b = nn.Parameter(torch.zeros(d_in))
        nn.init.uniform_(self.recon_w, -1.0 / math.sqrt(d_model),
                         1.0 / math.sqrt(d_model))

        # The GMA writes into this; MC3GRAM applies it to the input and so
        # does FlowMAE, so the mask-consensus path behaves identically.
        self.register_buffer("feature_mask", torch.ones(d_in))

    # ------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor,
                col_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x * self.feature_mask
        if col_mask is None and self.training and self.col_dropout > 0:
            # the cheap surrogate for GDR: a different random subset of
            # columns is absent in every batch, so no small set of columns can
            # be relied on. Costs nothing; separable from GDR in the ablation.
            col_mask = (torch.rand(x.size(0), self.d_in, device=x.device)
                        < self.col_dropout).float()
        h = self.encoder(x, col_mask)
        return self.cls_head(h[:, 0])           # CLS token

    # --------------------------------------------- masked-column objective
    def reconstruct(self, x: torch.Tensor, col_mask: torch.Tensor):
        """Predict the masked columns' values from the unmasked ones.

        Returns (prediction, target, mask) with the per-column scalars, so the
        caller takes the MSE over masked positions only -- a loss over all
        positions would be dominated by copying the visible columns through.
        """
        h = self.encoder(x * self.feature_mask, col_mask)
        pred = ((h[:, 1:] * self.recon_w.unsqueeze(0)).sum(-1)
                + self.recon_b.unsqueeze(0))              # (B, d_in)
        return pred, x, col_mask

    def mcp_loss(self, x: torch.Tensor, mask_frac: float = 0.3):
        m = (torch.rand(x.size(0), self.d_in, device=x.device)
             < mask_frac).float()
        # a row with nothing masked contributes no gradient; force one column
        empty = m.sum(1) == 0
        if empty.any():
            j = torch.randint(0, self.d_in, (int(empty.sum()),), device=x.device)
            m[empty, j] = 1.0
        pred, target, m = self.reconstruct(x, m)
        return (((pred - target) ** 2) * m).sum() / m.sum().clamp_min(1.0)

    # ------------------------------------------------- attribution, for GMA
    @torch.enable_grad()
    def attribution(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-view attribution: mean |d max-logit / d x| over a batch.

        MC3GRAM has three heads and mask_consensus votes over them. FlowMAE has
        one trunk, so the views are made by ablation instead: view i masks
        every column with (j mod n_views) == i, which is deterministic, covers
        the columns evenly and gives genuinely different attributions for the
        consensus to vote over. Returning a single view would reduce the GMA to
        a plain top-k and quietly change what that component means.
        """
        self.eval()
        dev = x.device
        idx = torch.arange(self.d_in, device=dev)
        out: dict[str, torch.Tensor] = {}
        for i in range(self.n_views):
            cm = (idx % self.n_views == i).float().unsqueeze(0)
            xi = x.clone().requires_grad_(True)
            logit = self.forward(xi, col_mask=cm.expand(x.size(0), -1))
            s = logit.max(dim=1).values.sum()
            g, = torch.autograd.grad(s, xi, retain_graph=False)
            out[f"view{i}"] = g.abs().mean(0).detach()
        return out


# ----------------------------------------------------- gradient dispersion
# PyTorch's fused scaled-dot-product-attention kernels have no second-order
# derivative: a double backward through them raises
#   derivative for aten::_scaled_dot_product_efficient_attention_backward
#   is not implemented
# and the gradient-dispersion penalty needs exactly that, because the penalty
# is a function of a gradient. The math backend is the unfused implementation
# and does support it. The choice is recorded in the autograd graph at FORWARD
# time, so wrapping the forward and the first grad call is enough -- the second
# backward, which happens later in the training loop, then runs on math nodes.
#
# The maths is identical; only the kernel differs. It is slower, which is part
# of why this penalty is the expensive option and col_dropout is the cheap one.
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend

    def _math_attention():
        return _sdpa_kernel(SDPBackend.MATH)
except Exception:                                    # older torch
    import contextlib

    @contextlib.contextmanager
    def _math_attention():
        try:
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False,
                    enable_math=True):
                yield
        except Exception:
            yield


def gradient_dispersion_penalty(model: nn.Module, xb: torch.Tensor,
                                yb: torch.Tensor, eps: float = 1e-12
                                ) -> torch.Tensor:
    """Negative entropy of the normalised input-gradient mass.

    With g = d CE(f(x), y) / d x and p_j = |g_j| / sum_k |g_k| per row, this
    returns the mean of -H(p). Adding it to the loss therefore MAXIMISES the
    entropy, which spreads the gradient mass across columns.

    create_graph=True is required -- the penalty is a function of a gradient,
    so a second backward is needed for it to reach the parameters. That is the
    cost of this term and it is why lambda defaults to 0. The cheap surrogate
    is FlowMAE(col_dropout=...), which needs no second backward.
    """
    xb = xb.detach().requires_grad_(True)
    with _math_attention():
        loss = F.cross_entropy(model(xb), yb)
        g, = torch.autograd.grad(loss, xb, create_graph=True)
    a = g.abs()
    p = a / a.sum(dim=1, keepdim=True).clamp_min(eps)
    H = -(p * (p + eps).log()).sum(dim=1)
    return -H.mean()


def uniform_entropy(d: int) -> float:
    """log d -- the entropy of a perfectly dispersed gradient, for reporting.

    The penalty's value on its own is not interpretable; H / log d is, and it
    is what the lambda sweep should report."""
    return math.log(max(d, 2))
