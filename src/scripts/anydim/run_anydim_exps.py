"""CLI orchestrator for the any-dimensional (OptDim) growing-graph experiments.

For each requested dataset, *sequentially*:
    1. copy the non-PCA knn-graph .pkl files from the cold-storage disk
       (``--source``, default /data/cnetto1/image-gnn-data-2) into the working
       directory ``data/<DS>Graph/``,
    2. run all four experiment arms (parallelized one-per-GPU),
    3. delete the *working-directory copy* (the cold-storage originals are
       never touched),
    4. move on to the next dataset.

This keeps at most one dataset's graphs resident in the working directory at a
time, so the working disk is never flooded.

Example:
    python src/scripts/anydim/run_anydim_exps.py \
        --datasets MNIST FER2013 FMNIST PathMNIST CIFAR10 CELEBA \
        --epochs 6 --a 1.0 --b 2 --gpus 0 1 2 3
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Make sibling modules importable whether run as a script or a module, matching
# the repo convention of running from src/scripts with that dir on sys.path.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent          # src/scripts
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/scripts/anydim

REPO_ROOT = SCRIPTS_DIR.parent.parent                          # repo root


# Dataset registry: dir/file stems + non-PCA file preference (base, then -2).
# WANGraph is intentionally excluded.
DATASETS = {
    "MNIST":     {"dir": "MNISTGraph",     "stem": "mnist"},
    "FER2013":   {"dir": "FER2013Graph",   "stem": "fer2013"},
    "FMNIST":    {"dir": "FMNISTGraph",    "stem": "fmnist"},
    "PathMNIST": {"dir": "PathMNISTGraph", "stem": "pathmnist"},
    "CIFAR10":   {"dir": "CIFAR10Graph",   "stem": "cifar10"},
    "CELEBA":    {"dir": "CELEBAGraph",    "stem": "celeba"},
}

# Order: small -> large (ascending by file size), so cheap runs validate first.
DEFAULT_ORDER = ["MNIST", "FER2013", "FMNIST", "PathMNIST", "CIFAR10", "CELEBA"]


def pick_pkl(src_dir: Path, stem: str, split: str) -> Path:
    """Select the non-PCA knn-graph pkl for a split, preferring base over -2.

    'pca' files are always excluded. Preference order:
      <stem>_<split>_knn_graph.pkl  (base)  ->  <stem>_<split>_knn_graph-2.pkl
      ->  any other non-pca <stem>_<split>_knn_graph*.pkl
    """
    base = src_dir / f"{stem}_{split}_knn_graph.pkl"
    if base.exists():
        return base
    dash2 = src_dir / f"{stem}_{split}_knn_graph-2.pkl"
    if dash2.exists():
        return dash2
    candidates = sorted(
        p for p in src_dir.glob(f"{stem}_{split}_knn_graph*.pkl")
        if "pca" not in p.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No non-PCA {split} graph for stem '{stem}' in {src_dir}")
    return candidates[0]


def stage_dataset(ds_name, source_root, work_root, verbose=True):
    """Copy this dataset's train/test pkls into the working dir. Returns paths."""
    meta = DATASETS[ds_name]
    src_dir = Path(source_root) / meta["dir"]
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    train_src = pick_pkl(src_dir, meta["stem"], "train")
    test_src = pick_pkl(src_dir, meta["stem"], "test")

    work_dir = Path(work_root) / meta["dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    train_dst = work_dir / train_src.name
    test_dst = work_dir / test_src.name

    for src, dst in [(train_src, train_dst), (test_src, test_dst)]:
        if verbose:
            size_gb = src.stat().st_size / 1e9
            print(f"[{ds_name}] copying {src.name} ({size_gb:.2f} GB) -> {dst}")
        shutil.copy2(src, dst)

    return work_dir, train_dst, test_dst


def cleanup_dataset(work_dir: Path, ds_name, verbose=True):
    """Delete the working-directory copy only. Cold storage is untouched."""
    work_dir = Path(work_dir)
    # Safety: only ever delete inside the working data dir.
    if "image-gnn-data-2" in str(work_dir) or work_dir == work_dir.anchor:
        raise RuntimeError(f"Refusing to delete cold-storage path: {work_dir}")
    if work_dir.is_dir():
        if verbose:
            print(f"[{ds_name}] removing working copy {work_dir}")
        shutil.rmtree(work_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_ORDER,
                   choices=list(DATASETS.keys()),
                   help="Datasets to run, in order (WANGraph excluded).")
    p.add_argument("--source", default="/data/cnetto1/image-gnn-data-2",
                   help="Cold-storage root holding <DS>Graph/ folders (read-only).")
    p.add_argument("--work-root", default=str(REPO_ROOT / "data"),
                   help="Working data dir to stage into and delete from.")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "anydim"),
                   help="Output root for per-dataset results/plots.")
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3],
                   help="GPU indices to use (capped at 4 by convention).")

    # Training hyperparameters.
    p.add_argument("--epochs", type=int, default=6,
                   help="Epochs for the MLP and fixed-size GNN arms.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)

    # Size grids (auto-derived from graph size if omitted).
    p.add_argument("--fixed-sizes", nargs="+", type=int, default=None,
                   help="Fixed-size GNN sweep sizes (default: auto from N).")
    p.add_argument("--nlist", nargs="+", type=int, default=None, dest="n_list",
                   help="Shared candidate-size grid for linear & OptDim arms "
                        "(default: auto from N).")

    # OptDim hyperparameters.
    p.add_argument("--a", type=float, default=1.0,
                   help="GNN gradient-error scaling exponent (a > 1/2 heuristic).")
    p.add_argument("--b", type=float, default=2.0,
                   help="Per-step compute-cost exponent (b=2 for GNNs).")
    p.add_argument("--fixed-L", type=float, default=None,
                   help="Use a known smoothness L instead of Hessian power "
                        "iteration (recommended for the small eta*L regime / "
                        "large graphs).")
    p.add_argument("--hvp-n-batches", type=int, default=1,
                   help="Batches used to estimate the Hessian operator.")
    p.add_argument("--hvp-n-iter", type=int, default=10,
                   help="Power-iteration steps for the max Hessian eigenvalue.")
    p.add_argument("--grad-max-batches", type=int, default=None,
                   help="Cap batches when estimating grad l^n / grad l^N "
                        "(None = full pass; useful for huge graphs).")
    p.add_argument("--optdim-epochs", type=int, default=None,
                   help="Epochs for the OptDim arm (default: len(n_list)).")

    p.add_argument("--arms", nargs="+",
                   default=["mlp", "fixed", "linear", "optdim"],
                   choices=["mlp", "fixed", "linear", "optdim"],
                   help="Subset of arms to run.")
    p.add_argument("--keep-data", action="store_true",
                   help="Do not delete the staged working copy after each run.")
    return p.parse_args()


def build_cfg(args):
    return {
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": args.hidden,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "fixed_sizes": args.fixed_sizes,
        "n_list": args.n_list,
        "a": args.a,
        "b": args.b,
        "fixed_L": args.fixed_L,
        "hvp_n_batches": args.hvp_n_batches,
        "hvp_n_iter": args.hvp_n_iter,
        "grad_max_batches": args.grad_max_batches,
        "optdim_epochs": args.optdim_epochs,
        "arms": args.arms,
    }


def main():
    args = parse_args()

    if len(args.gpus) > 4:
        print(f"WARNING: {len(args.gpus)} GPUs requested; convention is <=4. "
              f"Using first 4: {args.gpus[:4]}")
        args.gpus = args.gpus[:4]
    # Restrict visibility so we never touch GPUs outside the allowed set.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)
    # After masking, devices are re-indexed 0..k-1.
    gpus = list(range(len(args.gpus)))

    # Import after setting CUDA_VISIBLE_DEVICES so child processes inherit it.
    from anydim_exps import run_dataset

    cfg = build_cfg(args)
    print(f"Datasets (sequential): {args.datasets}")
    print(f"Source (read-only): {args.source}")
    print(f"Working dir (stage+delete): {args.work_root}")
    print(f"Visible GPUs: {args.gpus} -> reindexed {gpus}")

    summary = {}
    for ds_name in args.datasets:
        print(f"\n===== {ds_name} =====")
        work_dir = None
        try:
            work_dir, train_path, test_path = stage_dataset(
                ds_name, args.source, args.work_root)
            results = run_dataset(
                ds_name, str(train_path), str(test_path), cfg, gpus, args.out)
            summary[ds_name] = {
                arm: {k: v for k, v in r.items() if k != "history"}
                for arm, r in results.items()
            }
        except Exception as e:
            import traceback
            print(f"[{ds_name}] FAILED: {e}\n{traceback.format_exc()}")
            summary[ds_name] = {"error": str(e)}
        finally:
            if work_dir is not None and not args.keep_data:
                try:
                    cleanup_dataset(work_dir, ds_name)
                except Exception as e:
                    print(f"[{ds_name}] cleanup failed: {e}")

    print("\n===== SUMMARY =====")
    for ds_name, arms in summary.items():
        print(f"\n## {ds_name}")
        if "error" in arms:
            print(f"  ERROR: {arms['error']}")
            continue
        for arm, r in arms.items():
            if "error" in r:
                print(f"  {arm:8s} ERROR: {r['error']}")
            elif arm == "mlp":
                print(f"  {arm:8s} train={r['train_acc']:.4f} "
                      f"test={r['test_acc']:.4f} gap={r['gap']:.4f}")
            elif arm == "fixed":
                best = max(zip(r['sizes'], r['test_accs']), key=lambda t: t[1])
                print(f"  {arm:8s} best test={best[1]:.4f} @ size {best[0]}")
            else:
                print(f"  {arm:8s} final train={r['final_train_acc']:.4f} "
                      f"test={r['final_test_acc']:.4f} gap={r['gap']:.4f}")


if __name__ == "__main__":
    main()
