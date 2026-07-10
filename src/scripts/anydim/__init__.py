"""Any-dimensional optimization experiments (OptDim) for growing-graph GNN training.

This package extends the NeurIPS growing-subgraphs experiments
(``src/scripts/neurips_exps.py``) with the OptDim adaptive dimension scheduler
from the any-dimensional optimization paper (``any_dim_opt.pdf``).

Modules:
    core        -- model definitions, train/eval/subsample helpers (lifted
                   verbatim from ``neurips_exps.py`` so the baseline arms
                   reproduce the previously reported results).
    optdim      -- the OptDim algorithm (full-graph gradient, max-Hessian-
                   eigenvalue power iteration, ``n*`` computation, scheduler).
    anydim_exps -- per-dataset experiment driver running all four arms.
    run_anydim_exps -- CLI orchestrator that streams datasets through the
                   working directory one at a time.
"""
