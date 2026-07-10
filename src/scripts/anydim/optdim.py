"""OptDim: adaptive training-dimension selection for any-dimensional models.

Implements Algorithm 1 ("OptDim") from the any-dimensional optimization paper
(``any_dim_opt.pdf``). In our setting the "dimension" ``n`` is the *training
graph size* (number of subsampled nodes) and ``N`` is the full training graph,
used as the ``n -> infinity`` proxy for the gradient ``grad l^N``.

Each epoch the scheduler:
    1. computes the current-subgraph gradient ``grad l^n`` and the full-graph
       gradient ``grad l^N``,
    2. estimates the max Hessian eigenvalue ``L_t`` (smoothness) via power
       iteration with Hessian-vector products,
    3. computes the target dimension

           n* = n * ((1 - eta*L)/(1 - eta*L/2))^(1/a)
                  * (1 - <grad l^n, grad l^N> / ||grad l^N||^2)^(1/a)
                  * (1 + a/b)^(1/a)

    4. snaps ``n*`` to the nearest element of ``nList`` and only ever
       *increases* the current dimension (monotone).

For GNNs the paper fixes ``b = 2`` and treats ``a > 1/2`` as a heuristic to be
supplied. Both are configurable.

The module is intentionally self-contained (it only depends on torch) so that
importing it cannot perturb the baseline experiment arms.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Gradient / Hessian utilities
# --------------------------------------------------------------------------- #
def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def grad_vector(model, loader, use_edges, device, max_batches=None):
    """Flat mean-loss gradient ``grad l`` accumulated over a NeighborLoader.

    The cross-entropy loss is a mean over labelled (masked) nodes. Because the
    gradient is linear in the per-node losses, we accumulate the *sum*-reduced
    loss gradient over all batches via ``backward`` (memory-bounded: one batch
    in the graph at a time) and divide by the total labelled-node count. This
    yields the exact mean-loss gradient without ever materialising the whole
    graph's computation graph -- important for the large datasets.

    Returns a 1-D tensor (detached) on ``device``.
    """
    model.zero_grad(set_to_none=True)
    total = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device)
        m = batch.mask.bool()
        if m.sum() == 0:
            continue
        out = model(batch.x, batch.edge_index, batch.edge_weight) if use_edges else model(batch.x)
        loss = F.cross_entropy(out[m], batch.y[m], reduction="sum")
        loss.backward()
        total += int(m.sum().item())

    params = _trainable_params(model)
    g = torch.cat([
        (p.grad.detach().reshape(-1) if p.grad is not None
         else torch.zeros(p.numel(), device=device))
        for p in params
    ])
    model.zero_grad(set_to_none=True)
    if total > 0:
        g = g / total
    return g


def max_hessian_eig(model, batches, use_edges, device, n_iter=10, tol=1e-4):
    """Estimate the largest Hessian eigenvalue ``L_t`` via power iteration.

    Uses Hessian-vector products (double backprop). To stay memory-bounded on
    large graphs, the Hessian operator is estimated over a small fixed list of
    ``batches`` (a stochastic estimate of the full-graph Hessian); the same
    batches are reused across power-iteration steps so the operator is
    consistent. The paper notes this quantity can be expensive and that one may
    instead supply a known ``L`` in the small ``eta*L`` regime -- see the
    ``fixed_L`` escape hatch in :class:`OptDimScheduler`.

    Returns a non-negative float estimate of the spectral norm of the Hessian
    (we take ``|lambda|`` since the dominant eigenvalue may be returned with
    either sign by power iteration).
    """
    params = _trainable_params(model)
    # Move the (few) batches to device once and reuse.
    dev_batches = [b.to(device) for b in batches]

    v = [torch.randn_like(p) for p in params]
    _normalize_(v)

    eig = 0.0
    for _ in range(n_iter):
        hv = [torch.zeros_like(p) for p in params]
        count = 0
        for batch in dev_batches:
            m = batch.mask.bool()
            if m.sum() == 0:
                continue
            out = model(batch.x, batch.edge_index, batch.edge_weight) if use_edges else model(batch.x)
            loss = F.cross_entropy(out[m], batch.y[m])
            # allow_unused: some params (e.g. an unused conv layer) may not be
            # in the graph; treat their gradient as zero.
            grads = torch.autograd.grad(loss, params, create_graph=True,
                                        allow_unused=True)
            gv = sum((g * vi).sum() for g, vi in zip(grads, v) if g is not None)
            hv_part = torch.autograd.grad(gv, params, retain_graph=False,
                                          allow_unused=True)
            for acc, h in zip(hv, hv_part):
                if h is not None:
                    acc.add_(h.detach())
            count += 1
        if count > 1:
            hv = [h / count for h in hv]

        new_eig = _global_dot(hv, v)            # Rayleigh quotient (v is unit norm)
        _normalize_(hv)
        v = hv
        if abs(new_eig - eig) <= tol * (abs(new_eig) + 1e-12):
            eig = new_eig
            break
        eig = new_eig

    return float(abs(eig))


def _global_dot(xs, ys):
    return float(sum((x * y).sum() for x, y in zip(xs, ys)).item())


def _normalize_(vs):
    norm = (sum((v * v).sum() for v in vs)) ** 0.5
    norm = float(norm.item()) if torch.is_tensor(norm) else float(norm)
    if norm > 0:
        for v in vs:
            v.div_(norm)
    return vs


# --------------------------------------------------------------------------- #
# n* computation (Algorithm 1, inner update)
# --------------------------------------------------------------------------- #
def compute_n_star(n, eta, L, grad_n, grad_N, a, b, eps=1e-8):
    """Compute the OptDim target dimension ``n*``.

        n* = n * ((1 - eta*L)/(1 - eta*L/2))^(1/a)
               * (1 - <grad l^n, grad l^N> / ||grad l^N||^2)^(1/a)
               * (1 + a/b)^(1/a)

    The factors are clamped to remain in the valid (positive) regime: the
    theory assumes ``eta < 1/L`` (so ``1 - eta*L > 0``) and a non-negative
    dimensional-efficiency term. Outside that regime the corresponding factor
    is floored at ``eps`` so ``n*`` stays finite and positive.

    Returns ``(n_star, info)`` where ``info`` records the intermediate
    quantities for logging/debugging.
    """
    inv_a = 1.0 / a

    one_minus = 1.0 - eta * L
    one_minus_half = 1.0 - 0.5 * eta * L
    hess_ratio = one_minus / one_minus_half if one_minus_half != 0 else eps
    hess_ratio = max(hess_ratio, eps)               # eta < 1/L regime

    gN_sq = float((grad_N * grad_N).sum().item())
    dot_nN = float((grad_n * grad_N).sum().item())
    eff_term = 1.0 - (dot_nN / gN_sq if gN_sq > 0 else 0.0)
    eff_term = max(eff_term, eps)                   # gamma_n >= 0 regime

    ab_term = 1.0 + a / b

    factor = (hess_ratio ** inv_a) * (eff_term ** inv_a) * (ab_term ** inv_a)
    n_star = n * factor

    info = {
        "L": L,
        "eta": eta,
        "hess_ratio": hess_ratio,
        "grad_N_sq": gN_sq,
        "dot_nN": dot_nN,
        "eff_term": eff_term,
        "ab_term": ab_term,
        "factor": factor,
        "n_star_raw": n_star,
    }
    return n_star, info


def snap_to_list(n_star, n_list):
    """Return the element of ``n_list`` minimizing ``|n_k - n_star|``."""
    return min(n_list, key=lambda nk: abs(nk - n_star))


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
@dataclass
class OptDimScheduler:
    """Stateful OptDim dimension scheduler (Algorithm 1).

    Holds the candidate dimension grid ``n_list`` and the current dimension
    ``n``. Call :meth:`step` once per epoch (after ``trainEpoch``) with the
    model, the current-size and full-graph loaders, and the learning rate; it
    returns the (possibly increased) dimension to train at next, applying the
    monotone-increase rule ``if n_k > n: n <- n_k``.

    If ``fixed_L`` is provided, the (expensive) Hessian power iteration is
    skipped and that value is used for ``L_t`` -- matching the paper's
    suggestion to supply a known smoothness constant in the small ``eta*L``
    regime.
    """

    n_list: List[int]
    a: float = 1.0
    b: float = 2.0
    fixed_L: Optional[float] = None
    hvp_n_batches: int = 1
    hvp_n_iter: int = 10
    grad_max_batches: Optional[int] = None
    n: int = field(init=False)
    history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.n_list = sorted(int(x) for x in self.n_list)
        self.n = self.n_list[0]

    def step(self, model, cur_loader, full_loader, eta, use_edges, device):
        """One OptDim update. Returns the dimension to use for the next epoch."""
        grad_n = grad_vector(model, cur_loader, use_edges, device,
                             max_batches=self.grad_max_batches)
        grad_N = grad_vector(model, full_loader, use_edges, device,
                             max_batches=self.grad_max_batches)

        if self.fixed_L is not None:
            L = float(self.fixed_L)
        else:
            hvp_batches = []
            for i, batch in enumerate(cur_loader):
                if i >= self.hvp_n_batches:
                    break
                hvp_batches.append(batch)
            L = max_hessian_eig(model, hvp_batches, use_edges, device,
                                n_iter=self.hvp_n_iter)

        n_star, info = compute_n_star(self.n, eta, L, grad_n, grad_N,
                                      self.a, self.b)
        n_k = snap_to_list(n_star, self.n_list)

        prev = self.n
        if n_k > self.n:
            self.n = n_k

        info.update({"n_prev": prev, "n_star": n_star, "n_snapped": n_k,
                     "n_next": self.n})
        self.history.append(info)
        return self.n
