"""
attack_suite_v1.py -- the real evaluation attacks for MC3-GRAM.

Drop this into `ablation/` beside perturb.py. It does not modify perturb.py;
it replaces its role in the EVALUATION path only. `generator_perturbation`
stays where it is, as a training-time augmenter, which is what it actually is.

Why this file exists
--------------------
perturb.generator_perturbation computes

    prop  = gen(xb.size(0), yb, mask)          # a fresh class-conditional sample
    delta = (prop - xb).clamp(-eps, eps) * mask

The generator never sees `xb`. `prop` is therefore an arbitrary realistic
member of class y, and the displacement points from this sample toward a
typical one. That is prototype attraction: it DENOISES. The signature is that
macro-F1 rises with the budget, which no attack can do.

What this file adds
-------------------
  * FGSM, PGD (strong defaults, random start), Carlini-Wagner L2, and a
    transfer setting.
  * A declared split of each corpus's columns into perturbable and frozen.
    Perturbing a flag count or a TTL by 0.03 is not an attack an adversary
    can mount; leaving such columns free inflates every attack's effect.
  * An evaluation set built by REPLACING the test rows with their adversarial
    counterparts, which is the standard protocol, instead of injecting copies
    beside the clean ones.

Conventions kept from perturb.py
--------------------------------
  * X is float32 [N, d] in [0,1]; y is int64 [N].
  * `model(xb)` returns logits.
  * `mask` is a torch tensor [d] (the consensus mask), or None.
  * Rates are derived from one confusion matrix, by perturb.rates_from_confusion.

Self-check:  python attack_suite_v1.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

VERSION = "attack_suite_v3"

# v3: CW now reports attack success as a function of the L2 budget.
#
# mean_l2_successful is conditioned on success, so it is NOT comparable across
# models whose success rates differ. In the three-fold Phase 1 run the arm that
# was hardest to attack showed the SMALLEST mean L2 (success 0.041, mean L2
# 0.254) and the easiest showed the largest (success 0.736, mean L2 0.428),
# because averaging over only the flipped rows averages over whichever rows sat
# nearest the boundary. Read as a margin, that table reverses the ranking.
#
# success_at_l2 censors the failures instead -- the fraction of attacked rows
# that can be flipped within a given L2 -- and is comparable across arms.

# v2 fixes, both in the CW path and its self-check:
#   * c_init raised from 1e-2 to 1.0 and capped, with early exit once every
#     sample is fooled. At 1e-2 and three binary steps the penalty never grew
#     large enough to cross a confident model's margin, so CW reported an
#     18.5% success rate that was an optimiser budget artifact, not a property
#     of the model.
#   * the self-check's synthetic task was too easy (sigma=0.05, clean accuracy
#     exactly 1.0000, enormous margins). Now sigma=0.15 with weight decay, so
#     clean accuracy lands near 0.90 and the attack numbers mean something.


# --------------------------------------------------------------- device
def _dev(module) -> torch.device:
    """The device a module's parameters live on.

    Same reason as in perturb.py: inputs are built from numpy and start on the
    CPU, and a device mismatch surfaces only when the offending line is first
    reached -- often after the training is already spent.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ------------------------------------------------------- feature spec
@dataclass
class FeatureSpec:
    """Which columns an attacker may move.

    perturbable : bool array [d]. True = continuous quantity an adversary can
                  shape by changing traffic (rates, durations, byte and packet
                  length statistics, inter-arrival statistics).
                  False = flag counts, protocol/service identifiers, TTL
                  fields, and any integer count that has no fractional value.

    Declare one per corpus and record it in the manuscript. A robustness
    number measured against an attacker allowed to set "SYN Flag Count" to
    0.37 does not describe any real adversary.
    """

    perturbable: np.ndarray
    name: str = "unnamed"

    def __post_init__(self):
        self.perturbable = np.asarray(self.perturbable, dtype=bool)
        if self.perturbable.ndim != 1:
            raise ValueError("perturbable must be 1-D [d]")
        if not self.perturbable.any():
            raise ValueError("no perturbable column: every attack would be a no-op")

    @property
    def d(self) -> int:
        return int(self.perturbable.size)

    def torch_mask(self, dev) -> torch.Tensor:
        return torch.from_numpy(self.perturbable.astype(np.float32)).to(dev)

    @classmethod
    def all_perturbable(cls, d: int, name: str = "all") -> "FeatureSpec":
        """Fallback only. Use a real split; this one overstates every attack."""
        return cls(np.ones(d, dtype=bool), name=name)

    @classmethod
    def from_names(cls, columns, frozen_substrings, name: str = "by-name") -> "FeatureSpec":
        """Freeze any column whose name contains one of the substrings.

        Example for CICIDS-2017 / CIC-IoT-2023 style headers:
            frozen_substrings = ["flag", "protocol", "ttl", "_count",
                                 "proto", "service", "state", "syn", "ack",
                                 "fin", "rst", "psh", "urg", "cwe", "ece"]
        and for UNSW-NB15:
            ["proto", "service", "state", "sttl", "dttl", "swin", "dwin",
             "trans_depth", "ct_", "is_", "stcpb", "dtcpb"]
        """
        cols = [str(c).lower() for c in columns]
        subs = [s.lower() for s in frozen_substrings]
        pert = np.array([not any(s in c for s in subs) for c in cols], dtype=bool)
        return cls(pert, name=name)


def _apply_box(x_adv, x0, eps, pmask, p="inf"):
    """Project into the eps-ball around x0, into [0,1], and freeze columns."""
    if p == "inf":
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    return x0 + (x_adv - x0) * pmask


# ----------------------------------------------------------------- FGSM
def fgsm(model, X, y, eps, spec: FeatureSpec, bs: int = 4096) -> np.ndarray:
    """One signed-gradient step of size eps, on perturbable columns only."""
    model.eval()
    dev = _dev(model)
    pmask = spec.torch_mask(dev).unsqueeze(0)
    out = []
    for i in range(0, len(X), bs):
        x0 = torch.from_numpy(X[i:i + bs]).to(dev)
        yb = torch.from_numpy(y[i:i + bs]).to(dev)
        xb = x0.clone().requires_grad_(True)
        loss = F.cross_entropy(model(xb), yb)
        g, = torch.autograd.grad(loss, xb)
        x_adv = x0 + eps * g.sign()
        out.append(_apply_box(x_adv, x0, eps, pmask).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------ PGD
def pgd(model, X, y, eps, spec: FeatureSpec, steps: int = 40,
        alpha: float | None = None, random_start: bool = True,
        seed: int = 0, bs: int = 4096) -> np.ndarray:
    """L-infinity PGD with the usual settings.

    steps=40 and alpha=eps/10 with a random start, against the 10 steps and
    alpha=eps/4 with no random start in perturb.pgd_perturbation. A weak PGD
    understates the attack and flatters the defence.
    """
    model.eval()
    dev = _dev(model)
    pmask = spec.torch_mask(dev).unsqueeze(0)
    if alpha is None:
        alpha = max(eps / 10.0, 1e-6)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i in range(0, len(X), bs):
        x0 = torch.from_numpy(X[i:i + bs]).to(dev)
        yb = torch.from_numpy(y[i:i + bs]).to(dev)
        if random_start:
            noise = (torch.rand(x0.shape, generator=gen) * 2 - 1).to(dev) * eps
            xb = _apply_box(x0 + noise, x0, eps, pmask)
        else:
            xb = x0.clone()
        for _ in range(steps):
            xb = xb.detach().requires_grad_(True)
            loss = F.cross_entropy(model(xb), yb)
            g, = torch.autograd.grad(loss, xb)
            xb = _apply_box(xb.detach() + alpha * g.sign(), x0, eps, pmask)
        out.append(xb.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------- Carlini-Wagner L2
def cw_l2(model, X, y, spec: FeatureSpec, steps: int = 200,
          lr: float = 0.01, kappa: float = 0.0, c_init: float = 1.0,
          binary_steps: int = 6, c_max: float = 1e6,
          bs: int = 1024) -> tuple[np.ndarray, dict]:
    """Smallest-L2 untargeted CW, with frozen columns held fixed.

    Returns (X_adv, stats). For a tabular detector this is the informative
    attack: not how much damage a given budget does, but how little movement
    the decision boundary requires.

    Optimises w in tanh space on the perturbable columns only, so the box
    constraint holds by construction and frozen columns cannot move.
    """
    model.eval()
    dev = _dev(model)
    pm = spec.torch_mask(dev).unsqueeze(0)

    adv_all, l2_all, ok_all = [], [], []
    for i in range(0, len(X), bs):
        x0 = torch.from_numpy(X[i:i + bs]).to(dev)
        yb = torch.from_numpy(y[i:i + bs]).to(dev)
        n = x0.size(0)

        x0c = x0.clamp(1e-6, 1 - 1e-6)
        w0 = torch.log(x0c / (1 - x0c))               # logit of x0

        lo = torch.zeros(n, device=dev)
        hi = torch.full((n,), 1e10, device=dev)
        c = torch.full((n,), c_init, device=dev)

        best_adv = x0.clone()
        best_l2 = torch.full((n,), float("inf"), device=dev)

        for _bs_step in range(binary_steps):
            if torch.isfinite(best_l2).all() and _bs_step > 0 and (hi < 1e9).all():
                break           # every sample fooled and bracketed; nothing left to search
            w = w0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([w], lr=lr)
            for _ in range(steps):
                x_adv = torch.sigmoid(w)
                x_adv = x0 + (x_adv - x0) * pm      # freeze non-perturbable
                diff = x_adv - x0
                l2sq = (diff * diff).sum(1)

                z = model(x_adv)
                onehot = F.one_hot(yb, z.size(1)).bool()
                z_true = z[onehot]
                z_other = z.masked_fill(onehot, float("-inf")).max(1).values
                # untargeted: push true logit below the best other logit
                f = torch.clamp(z_true - z_other + kappa, min=0.0)

                loss = (l2sq + c * f).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    fooled = z.argmax(1) != yb
                    better = fooled & (l2sq < best_l2)
                    if better.any():
                        best_l2 = torch.where(better, l2sq, best_l2)
                        best_adv[better] = x_adv[better].detach()

            with torch.no_grad():
                succ = torch.isfinite(best_l2)
                hi = torch.where(succ, torch.min(hi, c), hi)
                lo = torch.where(~succ, torch.max(lo, c), lo)
                c = torch.where(hi < 1e9, (lo + hi) / 2, c * 10).clamp(max=c_max)

        adv_all.append(best_adv.detach().cpu().numpy())
        l2_all.append(best_l2.sqrt().detach().cpu().numpy())
        ok_all.append(torch.isfinite(best_l2).cpu().numpy())

    l2 = np.concatenate(l2_all)          # inf where the attack never succeeded
    ok = np.concatenate(ok_all)

    # The comparable quantity: of the rows attacked, what fraction can be
    # flipped inside a given L2 budget. Failures are censored at infinity
    # rather than dropped, so arms with very different success rates can be
    # read against each other.
    grid = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)
    success_at = {f"{g:g}": float((l2 <= g).mean()) for g in grid}

    def q(p):
        # l2 carries +inf where the attack never succeeded. When the quantile
        # falls between two infinities numpy evaluates inf - inf, returns NaN
        # and warns. The None is the right answer and was already returned;
        # only the warning was noise, and a warning that fires on an ordinary
        # outcome trains people to ignore warnings.
        if not np.isfinite(l2).any():
            return None
        with np.errstate(invalid="ignore"):
            v = float(np.quantile(l2, p))
        return v if np.isfinite(v) else None

    stats = {
        "attack": "cw_l2",
        "success_rate": float(ok.mean()),
        "success_at_l2": success_at,
        "l2_p01": q(0.01), "l2_p02": q(0.02), "l2_p05": q(0.05),
        "n": int(len(l2)),
        # conditioned on success: keep for continuity, never compare across arms
        "mean_l2_successful": float(l2[ok].mean()) if ok.any() else float("nan"),
        "median_l2_successful": (float(np.median(l2[ok])) if ok.any()
                                 else float("nan")),
        "_warning": ("mean/median_l2_successful are conditioned on success and "
                     "are not comparable across models with different success "
                     "rates; use success_at_l2"),
    }
    return np.concatenate(adv_all).astype(np.float32), stats


# ------------------------------------------------------------- transfer
def transfer(surrogate, X, y, eps, spec: FeatureSpec, **kw) -> np.ndarray:
    """PGD computed on a surrogate; evaluate the result on the target model.

    The black-box setting. An attacker is not constrained to the defender's
    own generator, and is not usually handed the defender's gradients either.
    """
    return pgd(surrogate, X, y, eps, spec, **kw)


# -------------------------------------------------- evaluation set builder
def build_eval_set(X, y, eps, model, spec: FeatureSpec, attack: str = "pgd",
                   mode: str = "replace", seed: int = 0,
                   surrogate=None, **kw):
    """The test set at a given attack strength.

    mode="replace" (default, and the headline protocol): every test row is
        replaced by its adversarial counterpart. This is what "accuracy under
        PGD at eps" means everywhere else in the literature.

    mode="inject": adversarial copies are added beside the clean rows, as
        perturb.build_evaluation_set did. Kept only so the old numbers can be
        reproduced for the protocol-dependence table. It dilutes the attack by
        construction, because half the evaluation set is clean, and it
        conflates attack strength with injection ratio.

    eps is attack strength and nothing else. It is never also an injection
    ratio.
    """
    if eps <= 0 and attack != "cw":
        return X, y, {"attack": "clean", "eps": 0.0}

    if attack == "fgsm":
        Xa, st = fgsm(model, X, y, eps, spec, **kw), {"attack": "fgsm"}
    elif attack == "pgd":
        Xa, st = pgd(model, X, y, eps, spec, seed=seed, **kw), {"attack": "pgd"}
    elif attack == "cw":
        Xa, st = cw_l2(model, X, y, spec, **kw)
    elif attack == "transfer":
        if surrogate is None:
            raise ValueError("attack='transfer' needs surrogate=<model>")
        Xa = transfer(surrogate, X, y, eps, spec, seed=seed, **kw)
        st = {"attack": "transfer"}
    else:
        raise ValueError(f"unknown attack {attack!r}; "
                         "generator_perturbation is not an attack")

    st.update({"eps": float(eps), "mode": mode, "spec": spec.name,
               "n_perturbable": int(spec.perturbable.sum()), "d": spec.d})

    if mode == "replace":
        return Xa.astype(np.float32), y.astype(np.int64), st
    if mode == "inject":
        return (np.concatenate([X, Xa]).astype(np.float32),
                np.concatenate([y, y]).astype(np.int64), st)
    raise ValueError(f"unknown mode {mode!r}")


# ------------------------------------------------------------- self-check
def _selfcheck():
    """Synthetic, CPU, a few seconds. Asserts the attacks actually attack."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, d, k = 1200, 12, 3

    # sigma=0.15, not 0.05: overlapping clusters and a model with ordinary
    # margins. At sigma=0.05 clean accuracy was exactly 1.0000 and the attack
    # numbers described a toy nothing resembles.
    centres = rng.normal(0.5, 0.18, size=(k, d))
    y = rng.integers(0, k, size=n).astype(np.int64)
    X = np.clip(centres[y] + rng.normal(0, 0.15, size=(n, d)), 0, 1).astype(np.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, k))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(model(Xt), yt).backward()
        opt.step()

    def acc(Xe, ye):
        with torch.no_grad():
            p = model(torch.from_numpy(Xe)).argmax(1).numpy()
        return float((p == ye).mean())

    clean = acc(X, y)
    print(f"clean accuracy            {clean:.4f}")
    assert 0.70 < clean < 0.999, (
        f"self-check task is wrong: clean accuracy {clean:.4f}. Below 0.70 the "
        "model did not train; at 1.0 the margins are so wide that the attack "
        "numbers mean nothing.")

    frozen = np.zeros(d, dtype=bool)
    frozen[:4] = True                       # pretend the first 4 are flags
    spec = FeatureSpec(~frozen, name="selfcheck")

    prev = clean
    for eps in (0.02, 0.05, 0.10, 0.20):
        Xa, ya, st = build_eval_set(X, y, eps, model, spec, attack="pgd")
        a = acc(Xa, ya)
        moved = np.abs(Xa - X).max(0)
        print(f"PGD   eps={eps:<5} accuracy {a:.4f}   "
              f"frozen cols moved: {moved[frozen].max():.2e}")
        assert a <= prev + 1e-9, (
            f"accuracy ROSE with eps ({prev:.4f} -> {a:.4f}). "
            "That is the denoising signature. This is not an attack.")
        assert moved[frozen].max() < 1e-6, "a frozen column moved"
        prev = a

    Xa, ya, st = build_eval_set(X, y, 0.05, model, spec, attack="fgsm")
    print(f"FGSM  eps=0.05  accuracy {acc(Xa, ya):.4f}")

    Xa, stats = cw_l2(model, X[:200], y[:200], spec, steps=200, binary_steps=6)
    print(f"CW    success {stats['success_rate']:.3f}   "
          f"mean L2 {stats['mean_l2_successful']:.4f} (not cross-comparable)")
    print("      success within L2 budget: "
          + "  ".join(f"{k}:{v:.3f}" for k, v in stats["success_at_l2"].items()))
    assert stats["success_rate"] > 0.8, (
        f"CW success only {stats['success_rate']:.3f}. CW is unbounded, so a low "
        "rate means the optimiser never crossed the margin -- raise steps, "
        "binary_steps or c_init rather than reporting this as robustness.")

    Xi, yi, _ = build_eval_set(X, y, 0.10, model, spec, attack="pgd", mode="inject")
    print(f"inject mode: {len(X)} -> {len(Xi)} rows, accuracy {acc(Xi, yi):.4f} "
          "(diluted by the clean half -- secondary only)")

    print("\nOK -- every attack reduces accuracy, frozen columns held.")


if __name__ == "__main__":
    print(VERSION)
    _selfcheck()
