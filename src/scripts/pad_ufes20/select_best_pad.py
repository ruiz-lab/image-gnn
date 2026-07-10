"""
Select the best PAD-UFES-20 CNNVAE sweep run (min test_loss) and write its
architecture to a model config file for the checkpointed retrain.
"""
import sys
import yaml
import argparse

from pathlib import Path

import wandb


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GNN-image-VAE_train-PADUFES20")
    parser.add_argument("--entity", type=str, default="caiodeberaldini")
    parser.add_argument("--out", type=str, default="config/best_cnnvae_pad_model_config.yaml")
    return parser.parse_args(args)


def main(sys_args):
    args = parse_args(sys_args)
    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))

    finished = [
        r for r in runs
        if r.state == "finished" and r.summary.get("test_loss") is not None
    ]
    if not finished:
        print("ERROR: no finished runs with a test_loss found.")
        return 1

    best = min(finished, key=lambda r: r.summary["test_loss"])
    cfg = best.config

    blocks = cfg.get("blocks")
    if blocks is None:
        blocks = [cfg["block_1"], cfg["block_2"], cfg["block_3"]]
    blocks = [int(b) for b in blocks]

    model_config = {
        "model": "CNNVAEModel",
        "args": {
            "in_channels": int(cfg.get("in_channels", 3)),
            "img_size": int(cfg.get("img_size", 224)),
            "latent_size": int(cfg["latent_size"]),
            "blocks": blocks,
        },
    }

    with open(Path(args.out), "w") as f:
        yaml.safe_dump(model_config, f, default_flow_style=False, sort_keys=False)

    print(f"BEST RUN: {best.name}  test_loss={best.summary['test_loss']:.4f}")
    print(f"  latent_size={model_config['args']['latent_size']} blocks={blocks} img_size={model_config['args']['img_size']}")
    print(f"  wrote -> {args.out}")
    print(f"  finished runs considered: {len(finished)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
