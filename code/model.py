"""
MC^3-GRAM model components, following the manuscript's equation numbering.

    Eqs 5-10    stem encoder
    Eqs 11-13   FPN head
    Eqs 14-15   residual MLP head
    Eqs 16-19   lightweight Transformer head
    Eq  20      Jacobian feature attribution -> per-head mask
    Eqs 21-25   accuracy gating, head aggregation, prediction
    Eqs 26-27   global accuracy, head-level mask vote (GMA)
    Eqs 28-33   conditional generator, dual-head critic, Wasserstein hinge

Everything is deliberately small: the study runs on CPU, and the questions
being asked are about whether each component contributes, not about squeezing
the last point of accuracy out of the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- stem, Eqs 5-10
class Stem(nn.Module):
    """Normalisation here is LayerNorm rather than BatchNorm.

    Under a Dirichlet(0.5) label skew each client sees a different class mix,
    so BatchNorm's running statistics diverge between clients and averaging
    them destroys the aggregate -- the standard failure mode of BatchNorm in
    federated training.  LayerNorm carries no cross-sample statistics and so
    aggregates cleanly.
    """

    def __init__(self, d_in: int, d: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------ FPN head, Eqs 11-13
class FPNHead(nn.Module):
    """Multi-scale 1-D pyramid over the latent vector with top-down fusion.

    The latent is treated as a length-d single-channel sequence; three
    successively coarser levels are built by strided convolution and fused
    top-down, which is the 1-D analogue of the feature pyramid in Eq. 11.
    """

    def __init__(self, d: int, n_cls: int, ch: int = 16):
        super().__init__()
        self.c1 = nn.Conv1d(1, ch, 5, stride=2, padding=2)
        self.c2 = nn.Conv1d(ch, ch, 5, stride=2, padding=2)
        self.c3 = nn.Conv1d(ch, ch, 5, stride=2, padding=2)
        self.l3 = nn.Conv1d(ch, ch, 1)
        self.l2 = nn.Conv1d(ch, ch, 1)
        self.l1 = nn.Conv1d(ch, ch, 1)
        self.out = nn.Linear(ch * 3, n_cls)

    def forward(self, z):
        x = z.unsqueeze(1)
        p1 = F.relu(self.c1(x))
        p2 = F.relu(self.c2(p1))
        p3 = F.relu(self.c3(p2))
        t3 = self.l3(p3)
        t2 = self.l2(p2) + F.interpolate(t3, size=p2.shape[-1], mode="nearest")
        t1 = self.l1(p1) + F.interpolate(t2, size=p1.shape[-1], mode="nearest")
        g = torch.cat([t1.mean(-1), t2.mean(-1), t3.mean(-1)], dim=1)
        return self.out(g)


# --------------------------------------------------- residual MLP head, Eqs 14-15
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(),
                               nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x):
        return F.relu(x + self.f(x))


class MLPHead(nn.Module):
    def __init__(self, d: int, n_cls: int):
        super().__init__()
        self.b1, self.b2 = ResBlock(d), ResBlock(d)
        self.out = nn.Linear(d, n_cls)

    def forward(self, z):
        return self.out(self.b2(self.b1(z)))


# ------------------------------------------------- Transformer head, Eqs 16-19
class TransformerHead(nn.Module):
    """Latent reshaped into tokens, one self-attention encoder layer, mean-pool."""

    def __init__(self, d: int, n_cls: int, n_tok: int = 8, n_head: int = 4):
        super().__init__()
        assert d % n_tok == 0
        self.n_tok, self.d_tok = n_tok, d // n_tok
        self.pos = nn.Parameter(torch.zeros(1, n_tok, self.d_tok))
        self.attn = nn.MultiheadAttention(self.d_tok, n_head, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(self.d_tok), nn.LayerNorm(self.d_tok)
        self.ff = nn.Sequential(nn.Linear(self.d_tok, self.d_tok * 2), nn.ReLU(),
                                nn.Linear(self.d_tok * 2, self.d_tok))
        self.out = nn.Linear(self.d_tok, n_cls)

    def forward(self, z):
        t = z.view(z.size(0), self.n_tok, self.d_tok) + self.pos
        a, _ = self.attn(t, t, t, need_weights=False)
        t = self.ln1(t + a)
        t = self.ln2(t + self.ff(t))
        return self.out(t.mean(1))


# ------------------------------------------------------------------ full model
HEADS = ("fpn", "mlp", "trf")


class MC3GRAM(nn.Module):
    """Stem + the selected heads + accuracy-gated aggregation (Eqs 21-25).

    `heads` selects which of the three are built, which is what the per-head
    ablations vary.  `feature_mask` is the consensus mask from the GMA, applied
    to the input as in Eq. 25; it is a buffer so it travels with the state dict.
    """

    def __init__(self, d_in: int, n_cls: int, d: int = 128,
                 heads: tuple[str, ...] = HEADS):
        super().__init__()
        self.heads_used = tuple(heads)
        self.stem = Stem(d_in, d)
        mods = {}
        if "fpn" in heads:
            mods["fpn"] = FPNHead(d, n_cls)
        if "mlp" in heads:
            mods["mlp"] = MLPHead(d, n_cls)
        if "trf" in heads:
            mods["trf"] = TransformerHead(d, n_cls)
        self.heads = nn.ModuleDict(mods)
        # gate logits over the heads, Eq. 22
        self.gate = nn.Parameter(torch.zeros(len(mods)))
        self.register_buffer("feature_mask", torch.ones(d_in))

    def head_logits(self, x):
        z = self.stem(x * self.feature_mask)
        return {k: h(z) for k, h in self.heads.items()}

    def forward(self, x):
        hl = self.head_logits(x)
        w = torch.softmax(self.gate, 0)
        keys = list(self.heads.keys())
        out = sum(w[i] * hl[k] for i, k in enumerate(keys))
        return out

    # ----------------------------------------------- Eq. 20, Jacobian attribution
    @torch.enable_grad()
    def attribution(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-head feature attribution: mean |d logit_max / d x| over a batch."""
        self.eval()
        out = {}
        for k, h in self.heads.items():
            xi = x.clone().requires_grad_(True)
            z = self.stem(xi * self.feature_mask)
            logit = h(z)
            s = logit.max(dim=1).values.sum()
            g, = torch.autograd.grad(s, xi, retain_graph=False)
            out[k] = g.abs().mean(0).detach()
        return out


def mask_consensus(attr: dict[str, torch.Tensor], keep_frac: float = 0.5,
                   soft: bool = False) -> tuple[torch.Tensor, float]:
    """Global Mask Aggregator, Eqs 26-27.

    Each head votes for the features in its own top-`keep_frac`; a feature
    enters the consensus mask when a majority of heads vote for it.  Returns
    the mask and the mean pairwise agreement between head votes, which is the
    quantity the imbalance sweep tracks.
    """
    keys = list(attr.keys())
    d = attr[keys[0]].numel()
    dev = attr[keys[0]].device          # GPU port: votes follow the attributions
    k = max(1, int(round(keep_frac * d)))
    votes = []
    for key in keys:
        idx = torch.topk(attr[key], k).indices
        v = torch.zeros(d, dtype=torch.float32, device=dev)
        v[idx] = 1.0
        votes.append(v)
    V = torch.stack(votes)
    if soft:
        mask = V.mean(0)
    else:
        mask = (V.sum(0) > len(keys) / 2).float()
        # never mask everything out
        if mask.sum() == 0:
            mask = torch.ones(d, device=dev)

    # mean pairwise Jaccard agreement between head votes
    agrees = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = (V[i] * V[j]).sum().item()
            union = ((V[i] + V[j]) > 0).float().sum().item()
            agrees.append(inter / union if union else 1.0)
    agreement = float(sum(agrees) / len(agrees)) if agrees else 1.0
    return mask, agreement


# -------------------------------------------- conditional GAN, Eqs 28-33
class Generator(nn.Module):
    """Mask-conditioned conditional generator, Eq. 28-29.

    Conditioned on a class embedding and on the consensus mask, so the
    synthetic samples are generated in the sub-space the heads agree matters.
    """

    def __init__(self, d_in: int, n_cls: int, d_noise: int = 32, d_h: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_cls, 16)
        self.d_noise = d_noise
        self.net = nn.Sequential(
            nn.Linear(d_noise + 16 + d_in, d_h), nn.LayerNorm(d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.LayerNorm(d_h), nn.ReLU(),
            nn.Linear(d_h, d_in), nn.Sigmoid(),
        )

    def forward(self, n: int, y: torch.Tensor, mask: torch.Tensor):
        # GPU port: the noise was built on the CPU while the module may live on
        # the GPU. Derive the device from the module's own weights and bring the
        # labels and the mask along, so a caller cannot get this wrong.
        dev = self.emb.weight.device
        zn = torch.randn(n, self.d_noise, device=dev)
        m = mask.to(dev).unsqueeze(0).expand(n, -1)
        return self.net(torch.cat([zn, self.emb(y.to(dev)), m], dim=1))


class Critic(nn.Module):
    """Dual-head critic, Eq. 30: a Wasserstein score and an auxiliary class head."""

    def __init__(self, d_in: int, n_cls: int, d_h: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, d_h), nn.LeakyReLU(0.2),
            nn.Linear(d_h, d_h), nn.LeakyReLU(0.2),
        )
        self.w = nn.Linear(d_h, 1)
        self.c = nn.Linear(d_h, n_cls)

    def forward(self, x):
        h = self.body(x)
        return self.w(h).squeeze(1), self.c(h)


def critic_clip(critic: nn.Module, c: float = 0.01):
    """Weight clipping to keep the critic 1-Lipschitz (Eq. 32)."""
    with torch.no_grad():
        for p in critic.parameters():
            p.clamp_(-c, c)
