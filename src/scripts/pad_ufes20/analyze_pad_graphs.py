"""
Analyze kNN data-manifold graphs for PAD-UFES-20.

Reports, per k: node/edge counts, degree distribution, edge-weight stats,
threshold sparsity (weights zeroed by the 0.75 cut), edge homophily
(same-class fraction) on the full graph and on the weight>0 subgraph, connected
components, and class balance. Saves a summary plot to recon_images/.
"""
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRAPH_DIR = Path("data/PadUfes20Graph")
K_VALUES = [10, 15, 30]
LABEL_NAMES = {0: "BCC", 1: "SCC", 2: "ACK", 3: "SEK", 4: "MEL", 5: "NEV"}


def components(num_nodes, edge_index):
    """Connected components over the undirected edge set (union-find)."""
    parent = list(range(num_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for a, b in zip(src, dst):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(num_nodes)]
    _, counts = np.unique(roots, return_counts=True)
    return counts


def homophily(edge_index, y, w=None):
    src, dst = edge_index[0], edge_index[1]
    same = (y[src] == y[dst])
    if w is None:
        return same.float().mean().item()
    if w.sum() == 0:
        return float("nan")
    return (same.float() * w).sum().item() / w.sum().item()


def analyze(k):
    g = torch.load(GRAPH_DIR / f"pad_train_knn_graph-{k}.pkl")
    N = g.x.shape[0]
    ei = g.edge_index
    ew = g.edge_weight.view(-1)
    y = g.y
    E = ei.shape[1]

    # degree (undirected edge list already symmetrized)
    deg = torch.bincount(ei[0], minlength=N).float()

    # weight stats
    nz = ew > 0
    n_zero = int((~nz).sum())

    # homophily
    homo_all = homophily(ei, y)
    homo_w = homophily(ei, y, ew)
    # homophily restricted to surviving (nonzero-weight) edges
    if nz.any():
        homo_survive = homophily(ei[:, nz], y)
    else:
        homo_survive = float("nan")

    # components on full edge set and on surviving subgraph
    comp_all = components(N, ei)
    comp_survive = components(N, ei[:, nz]) if nz.any() else np.array([N])

    # random-edge homophily baseline = sum_c p_c^2
    _, cnt = np.unique(y.numpy(), return_counts=True)
    p = cnt / cnt.sum()
    baseline = float((p ** 2).sum())

    return {
        "k": k, "N": N, "E": E,
        "deg_mean": deg.mean().item(), "deg_med": deg.median().item(),
        "deg_min": deg.min().item(), "deg_max": deg.max().item(),
        "w_mean_all": ew.mean().item(),
        "w_mean_nz": ew[nz].mean().item() if nz.any() else float("nan"),
        "n_zero": n_zero, "frac_zero": n_zero / E,
        "homo_all": homo_all, "homo_w": homo_w, "homo_survive": homo_survive,
        "baseline": baseline,
        "n_comp_all": len(comp_all), "largest_comp_all": int(comp_all.max()),
        "n_comp_survive": len(comp_survive), "largest_comp_survive": int(comp_survive.max()),
        "deg": deg.numpy(), "ew": ew.numpy(),
    }


def main():
    results = [analyze(k) for k in K_VALUES]

    print("=" * 78)
    print("PAD-UFES-20 kNN GRAPH ANALYSIS  (train graph; 2054 nodes total, 4 classes)")
    print("=" * 78)
    for r in results:
        print(f"\n--- k = {r['k']} ---")
        print(f"  nodes: {r['N']}   directed-symmetrized edges: {r['E']}")
        print(f"  degree: mean={r['deg_mean']:.1f}  median={r['deg_med']:.0f}  min={r['deg_min']:.0f}  max={r['deg_max']:.0f}")
        print(f"  edge weight: mean(all)={r['w_mean_all']:.4f}  mean(nonzero)={r['w_mean_nz']:.4f}")
        print(f"  thresholding (<0.75 -> 0): {r['n_zero']}/{r['E']} zeroed  ({100*r['frac_zero']:.1f}%)  "
              f"-> {r['E']-r['n_zero']} surviving edges")
        print(f"  homophily (same-class edge fraction):")
        print(f"      all edges (unweighted) : {r['homo_all']:.3f}")
        print(f"      all edges (weighted)   : {r['homo_w']:.3f}")
        print(f"      surviving edges only   : {r['homo_survive']:.3f}")
        print(f"      random baseline        : {r['baseline']:.3f}")
        print(f"  connected components:")
        print(f"      full graph     : {r['n_comp_all']} comps, largest={r['largest_comp_all']} "
              f"({100*r['largest_comp_all']/r['N']:.1f}% of nodes)")
        print(f"      surviving-only : {r['n_comp_survive']} comps, largest={r['largest_comp_survive']} "
              f"({100*r['largest_comp_survive']/r['N']:.1f}% of nodes)")

    # summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r in results:
        axes[0, 0].hist(r["deg"], bins=40, alpha=0.5, label=f"k={r['k']}")
        axes[0, 1].hist(r["ew"][r["ew"] > 0], bins=40, alpha=0.5, label=f"k={r['k']}")
    axes[0, 0].set_title("Degree distribution"); axes[0, 0].set_xlabel("degree"); axes[0, 0].legend()
    axes[0, 1].set_title("Nonzero edge weights"); axes[0, 1].set_xlabel("weight"); axes[0, 1].legend()

    ks = [r["k"] for r in results]
    axes[0, 2].plot(ks, [r["frac_zero"] for r in results], "o-")
    axes[0, 2].set_title("Fraction of edges zeroed by threshold"); axes[0, 2].set_xlabel("k"); axes[0, 2].set_ylim(0, 1)

    axes[1, 0].plot(ks, [r["homo_all"] for r in results], "o-", label="all (unweighted)")
    axes[1, 0].plot(ks, [r["homo_survive"] for r in results], "s-", label="surviving")
    axes[1, 0].plot(ks, [r["baseline"] for r in results], "k--", label="random baseline")
    axes[1, 0].set_title("Edge homophily"); axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylim(0, 1); axes[1, 0].legend()

    axes[1, 1].plot(ks, [r["largest_comp_all"] / r["N"] for r in results], "o-", label="full")
    axes[1, 1].plot(ks, [r["largest_comp_survive"] / r["N"] for r in results], "s-", label="surviving")
    axes[1, 1].set_title("Largest component (fraction of nodes)"); axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylim(0, 1); axes[1, 1].legend()

    axes[1, 2].plot(ks, [r["deg_mean"] for r in results], "o-")
    axes[1, 2].set_title("Mean degree"); axes[1, 2].set_xlabel("k")

    fig.suptitle("PAD-UFES-20 kNN graph statistics")
    fig.tight_layout()
    out = Path("recon_images") / "pad_graph_stats.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsummary figure -> {out}")


if __name__ == "__main__":
    main()
