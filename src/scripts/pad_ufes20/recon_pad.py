"""
Generate original-vs-reconstruction figures for the best PAD-UFES-20 CNNVAE.
Saves N side-by-side panels to recon_images/.
"""
import sys
import yaml
import argparse

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.models import CNNVAEModel
from data_preproc.datasets import build_datasets


def parse_args(args):
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset_config", default="config/cnnvae_pad_dataset_config.yaml")
    p.add_argument("-m", "--model_config", default="config/best_cnnvae_pad_model_config.yaml")
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-o", "--out_dir", default="recon_images")
    p.add_argument("-n", "--num", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(args)


def denorm(t):
    # images were normalized with mean=std=0.5 -> undo to [0,1]
    return (t * 0.5 + 0.5).clamp(0, 1)


def main(sys_args):
    args = parse_args(sys_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.dataset_config) as f:
        dcfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        mcfg = yaml.safe_load(f)

    # use the held-out test split for reconstructions
    _, test_ds = build_datasets(dcfg)

    model = CNNVAEModel(**CNNVAEModel.pre_init(mcfg["args"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(test_ds), size=args.num, replace=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = {0: "BCC", 1: "SCC", 2: "ACK", 3: "SEK", 4: "MEL", 5: "NEV"}

    per_recon_mse = []
    with torch.no_grad():
        for j, idx in enumerate(idxs):
            data = test_ds[int(idx)]
            x = data.x.to(device)              # (1, 3, 224, 224)
            recon, mu, logvar, z = model(_wrap(x))
            mse = torch.mean((recon - x) ** 2).item()
            per_recon_mse.append(mse)

            orig = denorm(x[0]).cpu().permute(1, 2, 0).numpy()
            rec = denorm(recon[0]).cpu().permute(1, 2, 0).numpy()
            lbl = int(test_ds.targets[int(idx)])

            fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
            axes[0].imshow(orig); axes[0].set_title("original"); axes[0].axis("off")
            axes[1].imshow(rec); axes[1].set_title(f"reconstruction\nMSE={mse:.4f}"); axes[1].axis("off")
            fig.suptitle(f"test idx {int(idx)}  |  class {lbl} ({label_names.get(lbl, '?')})")
            fig.tight_layout()
            out_path = out_dir / f"recon_{j:02d}_idx{int(idx)}_cls{lbl}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {out_path}  (MSE={mse:.4f})")

    print(f"mean recon MSE over {args.num} images: {np.mean(per_recon_mse):.4f}")


class _wrap:
    """Minimal object exposing `.x` so CNNVAEModel.forward(batch) works."""
    def __init__(self, x):
        self.x = x


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
