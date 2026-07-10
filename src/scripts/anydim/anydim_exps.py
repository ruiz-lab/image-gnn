"""Per-dataset driver for the any-dimensional (OptDim) growing-graph experiments.

Runs four experiment *arms* per dataset and compares their generalization:

    1. ``mlp``     -- MLP baseline (no edges).
    2. ``fixed``   -- GNN trained at each fixed graph size in ``fixed_sizes``.
    3. ``linear``  -- GNN on a *linear* growing-subgraph schedule (the original
                      paper strategy).
    4. ``optdim``  -- GNN on an *adaptive* growing-subgraph schedule chosen by
                      the OptDim algorithm (the new contribution).

Arms 1-3 are behaviourally identical to ``src/scripts/neurips_exps.py`` so that
running without ``optdim`` reproduces the previously reported results. Arm 4
reuses the *same* candidate-size grid (``n_list``) as the linear arm, so the
only difference is adaptive-vs-linear selection over the same sizes.

The four arms run concurrently, one per GPU, via ``torch.multiprocessing``;
datasets are processed sequentially by the orchestrator (run_anydim_exps.py).
"""

import json
import os
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

from core import (
    GNN, MLP,
    train, evaluate, subsample, make_loader, set_seed, load_graph,
)
from optdim import OptDimScheduler


# --------------------------------------------------------------------------- #
# Size-grid helpers
# --------------------------------------------------------------------------- #
def default_grid(num_nodes):
    """Derive (fixed_sizes, schedule/n_list) grids from the graph size.

    Mirrors the spirit of the original FMNIST experiment
    (``sizes = [6000, 12000, 24000, 48000, 96000]`` with a growing schedule
    ``[750, 1500, 3000, 6000, 12000, 24000]``) but scales to any dataset by
    doubling from a small base up to (roughly) the full node count.
    """
    grid = []
    s = 750
    while s < num_nodes:
        grid.append(s)
        s *= 2
    grid.append(num_nodes)
    # fixed-size sweep: the upper half of the grid (larger graphs)
    fixed = [g for g in grid if g >= grid[len(grid) // 2]]
    return fixed, grid


# --------------------------------------------------------------------------- #
# Arms (each runs in its own process / GPU)
# --------------------------------------------------------------------------- #
def _resolve_device(device_str):
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def arm_mlp(train_path, test_path, device_str, cfg):
    device = _resolve_device(device_str)
    set_seed(cfg["seed"])
    train_data = load_graph(train_path)
    test_data = load_graph(test_path)
    num_classes = int(train_data.y.unique().numel())
    criterion = torch.nn.CrossEntropyLoss()

    model = MLP(train_data.num_node_features, cfg["hidden"], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    train_loader = make_loader(train_data, [-1], cfg["batch_size"])
    test_loader = make_loader(test_data, [-1], 512)

    tr_acc = te_acc = tr_loss = te_loss = 0.0
    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train(model, train_loader, False, optimizer, criterion, device)
        if epoch == cfg["epochs"] - 1:
            te_loss, te_acc = evaluate(model, test_loader, False, criterion, device)
    return {
        "arm": "mlp",
        "train_acc": tr_acc, "test_acc": te_acc,
        "train_loss": tr_loss, "test_loss": te_loss,
        "gap": tr_acc - te_acc, "loss_gap": tr_loss - te_loss,
    }


def arm_fixed(train_path, test_path, device_str, cfg):
    device = _resolve_device(device_str)
    set_seed(cfg["seed"])
    train_data = load_graph(train_path)
    test_data = load_graph(test_path)
    num_classes = int(train_data.y.unique().numel())
    criterion = torch.nn.CrossEntropyLoss()
    test_loader = make_loader(test_data, [-1], 512)

    sizes, train_accs, test_accs, gaps = [], [], [], []
    train_losses, test_losses = [], []
    for sz in cfg["fixed_sizes"]:
        sub = subsample(train_data, sz)
        loader = make_loader(sub, [10, 10], cfg["batch_size"], shuffle=True)
        model = GNN(train_data.num_node_features, cfg["hidden"], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        tr_acc = te_acc = tr_loss = te_loss = 0.0
        for epoch in range(cfg["epochs"]):
            tr_loss, tr_acc = train(model, loader, True, optimizer, criterion, device)
            if epoch == cfg["epochs"] - 1:
                te_loss, te_acc = evaluate(model, test_loader, True, criterion, device)
        sizes.append(sz)
        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_losses.append(tr_loss); test_losses.append(te_loss)
        gaps.append(tr_acc - te_acc)
    return {
        "arm": "fixed", "sizes": sizes,
        "train_accs": train_accs, "test_accs": test_accs,
        "train_losses": train_losses, "test_losses": test_losses,
        "gaps": gaps,
    }


def arm_linear(train_path, test_path, device_str, cfg):
    """GNN on the original linear growing-subgraph schedule."""
    device = _resolve_device(device_str)
    set_seed(cfg["seed"])
    train_data = load_graph(train_path)
    test_data = load_graph(test_path)
    num_classes = int(train_data.y.unique().numel())
    criterion = torch.nn.CrossEntropyLoss()
    test_loader = make_loader(test_data, [-1], 512)

    schedule = cfg["n_list"]
    cached = {sz: subsample(train_data, sz) for sz in set(schedule)}

    model = GNN(train_data.num_node_features, cfg["hidden"], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    tr_accs, te_accs, tr_losses, te_losses, used_sizes = [], [], [], [], []
    for sz in schedule:
        sub = cached[sz]
        loader = make_loader(sub, [10, 10], cfg["batch_size"], shuffle=True)
        tr_loss, tr_acc = train(model, loader, True, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, True, criterion, device)
        used_sizes.append(sz)
        tr_accs.append(tr_acc); te_accs.append(te_acc)
        tr_losses.append(tr_loss); te_losses.append(te_loss)
    return {
        "arm": "linear", "schedule": schedule, "used_sizes": used_sizes,
        "train_accs": tr_accs, "test_accs": te_accs,
        "train_losses": tr_losses, "test_losses": te_losses,
        "final_train_acc": tr_accs[-1], "final_test_acc": te_accs[-1],
        "gap": tr_accs[-1] - te_accs[-1],
        "loss_gap": tr_losses[-1] - te_losses[-1],
    }


def arm_optdim(train_path, test_path, device_str, cfg):
    """GNN on the OptDim adaptive growing-subgraph schedule (new contribution).

    Shares the same candidate grid (``n_list``) as the linear arm; OptDim picks
    which size to grow to at each epoch instead of following the fixed linear
    schedule. Runs for the same number of epochs (= len(schedule)) for a fair
    compute-matched comparison.
    """
    device = _resolve_device(device_str)
    set_seed(cfg["seed"])
    train_data = load_graph(train_path)
    test_data = load_graph(test_path)
    num_classes = int(train_data.y.unique().numel())
    criterion = torch.nn.CrossEntropyLoss()
    test_loader = make_loader(test_data, [-1], 512)

    n_list = cfg["n_list"]
    cached = {sz: subsample(train_data, sz) for sz in set(n_list)}
    # Full-graph loader (the n -> infinity proxy) over labelled nodes.
    full_loader = make_loader(train_data, [10, 10], cfg["batch_size"])

    sched = OptDimScheduler(
        n_list=n_list, a=cfg["a"], b=cfg["b"],
        fixed_L=cfg.get("fixed_L"),
        hvp_n_batches=cfg.get("hvp_n_batches", 1),
        hvp_n_iter=cfg.get("hvp_n_iter", 10),
        grad_max_batches=cfg.get("grad_max_batches"),
    )

    model = GNN(train_data.num_node_features, cfg["hidden"], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    n_epochs = cfg.get("optdim_epochs") or len(n_list)
    tr_accs, te_accs, tr_losses, te_losses, chosen_sizes = [], [], [], [], []
    n = sched.n
    for epoch in range(n_epochs):
        sub = cached[n]
        loader = make_loader(sub, [10, 10], cfg["batch_size"], shuffle=True)
        tr_loss, tr_acc = train(model, loader, True, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, True, criterion, device)
        chosen_sizes.append(n)
        tr_accs.append(tr_acc); te_accs.append(te_acc)
        tr_losses.append(tr_loss); te_losses.append(te_loss)
        # OptDim update: choose the dimension for the next epoch.
        cur_loader = make_loader(sub, [10, 10], cfg["batch_size"])
        n = sched.step(model, cur_loader, full_loader, cfg["lr"], True, device)

    return {
        "arm": "optdim", "n_list": n_list, "chosen_sizes": chosen_sizes,
        "a": cfg["a"], "b": cfg["b"],
        "train_accs": tr_accs, "test_accs": te_accs,
        "train_losses": tr_losses, "test_losses": te_losses,
        "final_train_acc": tr_accs[-1], "final_test_acc": te_accs[-1],
        "gap": tr_accs[-1] - te_accs[-1],
        "loss_gap": tr_losses[-1] - te_losses[-1],
        "history": sched.history,
    }


ARMS = {
    "mlp": arm_mlp,
    "fixed": arm_fixed,
    "linear": arm_linear,
    "optdim": arm_optdim,
}


def _run_arm_worker(arm_name, train_path, test_path, device_str, cfg, q):
    try:
        result = ARMS[arm_name](train_path, test_path, device_str, cfg)
        q.put((arm_name, result))
    except Exception as e:  # surface the failure to the parent without hanging
        import traceback
        q.put((arm_name, {"arm": arm_name, "error": str(e),
                          "traceback": traceback.format_exc()}))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_plots(ds_name, results, out_dir):
    fixed = results.get("fixed", {})
    mlp = results.get("mlp", {})
    linear = results.get("linear", {})
    optdim = results.get("optdim", {})

    # ---- Generalization gap vs size ----
    if fixed.get("sizes"):
        plt.figure(figsize=(8, 5))
        plt.plot(fixed["sizes"], fixed["gaps"], marker="o", label="GNN (fixed) Gen Gap")
        lo, hi = fixed["sizes"][0], fixed["sizes"][-1]
        if "gap" in mlp:
            plt.hlines(mlp["gap"], lo, hi, colors="r", linestyles="dashed", label="MLP Gen Gap")
        if "gap" in linear:
            plt.hlines(linear["gap"], lo, hi, colors="g", linestyles="dashdot", label="Linear Growing GNN Gen Gap")
        if "gap" in optdim:
            plt.hlines(optdim["gap"], lo, hi, colors="m", linestyles="dotted", label="OptDim Growing GNN Gen Gap")
        plt.xlabel("Training Graph Size")
        plt.ylabel("Generalization Gap (Train Acc - Test Acc)")
        plt.title(f"{ds_name}: GNN vs MLP Generalization Gap")
        plt.legend(); plt.grid(True)
        plt.savefig(out_dir / f"{ds_name}_gen_gap.png", bbox_inches="tight")
        plt.close()

    # ---- Test accuracy curves for the growing arms ----
    plt.figure(figsize=(8, 5))
    if linear.get("test_accs"):
        plt.plot(range(1, len(linear["test_accs"]) + 1), linear["test_accs"],
                 marker="o", label="Linear schedule (test acc)")
    if optdim.get("test_accs"):
        plt.plot(range(1, len(optdim["test_accs"]) + 1), optdim["test_accs"],
                 marker="s", label="OptDim schedule (test acc)")
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.title(f"{ds_name}: Linear vs OptDim growing-graph test accuracy")
    plt.legend(); plt.grid(True)
    plt.savefig(out_dir / f"{ds_name}_growing_test_acc.png", bbox_inches="tight")
    plt.close()

    # ---- Chosen dimension vs epoch (linear vs OptDim) ----
    plt.figure(figsize=(8, 5))
    if linear.get("used_sizes"):
        plt.plot(range(1, len(linear["used_sizes"]) + 1), linear["used_sizes"],
                 marker="o", label="Linear schedule size")
    if optdim.get("chosen_sizes"):
        plt.plot(range(1, len(optdim["chosen_sizes"]) + 1), optdim["chosen_sizes"],
                 marker="s", label="OptDim chosen size")
    plt.xlabel("Epoch"); plt.ylabel("Training Graph Size (dimension n)")
    plt.title(f"{ds_name}: dimension schedule (linear vs OptDim)")
    plt.legend(); plt.grid(True)
    plt.savefig(out_dir / f"{ds_name}_dimension_schedule.png", bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_dataset(ds_name, train_path, test_path, cfg, gpus, out_root):
    """Run all four arms for one dataset, parallelized one-arm-per-GPU.

    Returns the results dict and writes ``<out_root>/<ds_name>/results.json``
    plus comparison plots.
    """
    out_dir = Path(out_root) / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the size grids now (needs num_nodes) unless supplied explicitly.
    if cfg.get("fixed_sizes") is None or cfg.get("n_list") is None:
        head = load_graph(train_path)
        num_nodes = head.num_nodes
        del head
        auto_fixed, auto_grid = default_grid(num_nodes)
        cfg = dict(cfg)
        cfg.setdefault("fixed_sizes", None)
        cfg.setdefault("n_list", None)
        if cfg["fixed_sizes"] is None:
            cfg["fixed_sizes"] = auto_fixed
        if cfg["n_list"] is None:
            cfg["n_list"] = auto_grid

    arm_names = cfg.get("arms", ["mlp", "fixed", "linear", "optdim"])

    print(f"[{ds_name}] arms={arm_names} | fixed_sizes={cfg['fixed_sizes']} | "
          f"n_list={cfg['n_list']} | a={cfg['a']} b={cfg['b']} | gpus={gpus}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for i, arm_name in enumerate(arm_names):
        device_str = f"cuda:{gpus[i % len(gpus)]}" if gpus else "cpu"
        p = ctx.Process(
            target=_run_arm_worker,
            args=(arm_name, train_path, test_path, device_str, cfg, q),
        )
        p.start()
        procs.append(p)

    results = {}
    for _ in arm_names:
        arm_name, res = q.get()
        results[arm_name] = res
        if "error" in res:
            print(f"[{ds_name}] arm '{arm_name}' FAILED:\n{res['traceback']}")
        else:
            print(f"[{ds_name}] arm '{arm_name}' done.")
    for p in procs:
        p.join()

    with open(out_dir / "results.json", "w") as f:
        json.dump({"dataset": ds_name, "config": _jsonable(cfg),
                   "results": results}, f, indent=2)

    try:
        make_plots(ds_name, results, out_dir)
    except Exception as e:
        print(f"[{ds_name}] plotting failed: {e}")

    return results


def _jsonable(cfg):
    return {k: v for k, v in cfg.items()}
