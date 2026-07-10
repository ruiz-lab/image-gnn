# AnyDim — Adaptive growing-graph training via any-dimensional optimization

This package implements the **OptDim** adaptive dimension scheduler from the
any-dimensional optimization paper (`any_dim_opt.pdf`) and applies it to the
*growing-graph* GNN training strategy studied in this repo.

## Relation to the paper

The any-dimensional optimization framework treats the training "dimension" $n$
as a quantity that can be *grown during training* instead of fixed up front. The
paper's **OptDim** algorithm chooses, at each step, the dimension $n^\star$ that
best trades off gradient quality against per-step compute cost.

In our setting we make the following identification:

| Paper concept                     | Here                                                        |
| --------------------------------- | ----------------------------------------------------------- |
| dimension $n$                     | **training graph size** (number of subsampled nodes)        |
| the $n \to \infty$ limit          | the **full training graph** (used as the $\nabla l^N$ proxy) |
| smoothness constant $L_t$         | max Hessian eigenvalue (power iteration on HVPs)            |
| cost exponent $b$                 | fixed to $b = 2$ for GNNs (paper)                           |
| gradient-error exponent $a$       | heuristic, $a > 1/2$ (supplied by the user)                 |

Each epoch the scheduler computes the target dimension

$$
n^\star = n \times \left(\frac{1-\eta L}{1-\eta L/2}\right)^{1/a} \times \left(1 - \frac{\langle \nabla l^n, \nabla l^N\rangle}{\|\nabla l^N\|^2}\right)^{1/a} \times \left(1 + \frac{a}{b}\right)^{1/a}
$$

snaps $n^\star$ to the nearest element of the candidate grid `n_list`, and only
ever *increases* the current size (monotone growth). This replaces the
previously used **linear** growing-subgraph schedule with an adaptive one, over
the exact same candidate sizes — so the only difference between the two arms is
adaptive-vs-linear selection.

## Modules

| File                  | Purpose                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `core.py`             | Model defs (`GNN`, `MLP`) and train/eval/subsample helpers, lifted verbatim from `neurips_exps.py` so baseline arms reproduce prior results. |
| `optdim.py`           | The OptDim algorithm: full-graph gradient, max-Hessian-eigenvalue power iteration, $n^\star$ computation, and the `OptDimScheduler`. |
| `anydim_exps.py`      | Per-dataset driver running the four experiment arms (concurrently, one per GPU).            |
| `run_anydim_exps.py`  | CLI orchestrator that streams datasets one-at-a-time through the working directory.         |

## Experiment arms

`anydim_exps.py` runs four arms per dataset and compares their generalization:

1. **`mlp`** — MLP baseline (no edges).
2. **`fixed`** — GNN trained at each fixed graph size in `fixed_sizes`.
3. **`linear`** — GNN on the original *linear* growing-subgraph schedule (the
   previously reported method).
4. **`optdim`** — GNN on the *adaptive* growing-subgraph schedule chosen by
   OptDim (the new contribution). Reuses the same `n_list` as `linear`.

## Running experiments

The orchestrator stages each dataset's kNN-graph `.pkl` files from cold storage
into the working `data/` dir, runs the arms (one per GPU), then deletes the
working copy before moving to the next dataset.

```bash
python src/scripts/anydim/run_anydim_exps.py \
    --datasets MNIST FER2013 FMNIST PathMNIST CIFAR10 CELEBA \
    --epochs 6 --a 1.0 --b 2 --gpus 0 1 2 3
```

Useful flags:

- `--datasets` — subset/order of datasets to run (default: small→large).
- `--source`   — read-only cold-storage root holding `<DS>Graph/` folders.
- `--work-root` / `--out` — staging dir and results output root.
- `--gpus`     — GPU indices (capped at 4 by convention).
- `--arms`     — subset of `{mlp, fixed, linear, optdim}` to run.
- `--keep-data`— keep the staged working copy instead of deleting it.

## Tuning hyperparameters

**Optimization / training:**

- `--epochs` — epochs for the MLP and fixed-size GNN arms.
- `--lr`, `--hidden`, `--batch-size`, `--seed` — standard training knobs.

**Size grids** (auto-derived from the graph size $N$ if omitted):

- `--fixed-sizes` — the fixed-size GNN sweep sizes.
- `--nlist`       — the shared candidate-size grid used by both the `linear` and
  `optdim` arms.

**OptDim-specific:**

- `--a` — gradient-error scaling exponent $a$. Heuristic $a > 1/2$; larger $a$
  dampens the growth factor (smaller $n^\star$ jumps).
- `--b` — per-step compute-cost exponent (fixed to $b = 2$ for GNNs per the
  paper).
- `--fixed-L` — supply a known smoothness constant $L$ instead of estimating it
  by Hessian power iteration. **Recommended for large graphs / the small
  $\eta L$ regime**, where the power iteration is expensive.
- `--hvp-n-batches` — number of batches used to estimate the Hessian operator.
- `--hvp-n-iter` — power-iteration steps for the max Hessian eigenvalue.
- `--grad-max-batches` — cap batches when estimating $\nabla l^n$ / $\nabla l^N$
  (`None` = full pass; useful for huge graphs).
- `--optdim-epochs` — epochs for the OptDim arm (default: `len(n_list)`).

To sweep a hyperparameter, invoke `run_anydim_exps.py` repeatedly with different
values (e.g. varying `--a` over $\{0.75,\ 1.0,\ 1.5\}$) and compare the per-arm
train/test accuracy and generalization gap printed in the run summary.
