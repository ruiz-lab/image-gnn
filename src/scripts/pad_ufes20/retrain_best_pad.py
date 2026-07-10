"""
Retrain the best PAD-UFES-20 CNNVAE config with checkpoint saving.

The sweep uses Trainer.train() which does NOT save weights; this uses
Trainer.train_eval() (which checkpoints to src/scripts/checkpoints/). Prints the
resulting checkpoint path on the last line as: CHECKPOINT=<path>
"""
import sys
import yaml
import argparse

from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import Trainer


CKPT_DIR = Path("src/scripts/checkpoints")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_config", type=str, required=True)
    parser.add_argument("-t", "--training_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    return parser.parse_args(args)


def load(p):
    with open(Path(p), "r") as f:
        return yaml.safe_load(f)


def main(sys_args):
    args = parse_args(sys_args)
    wandb.init(mode="disabled")

    config = {
        "dataset_config": load(args.dataset_config),
        "training_config": load(args.training_config),
        "model_config": load(args.model_config),
    }
    config["training_config"]["save_model"] = True

    before = set(CKPT_DIR.glob("*.pt"))
    trainer = Trainer(**config)
    trainer.train_eval()
    after = set(CKPT_DIR.glob("*.pt"))

    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new:
        print(f"CHECKPOINT={new[-1]}")
    else:
        # fall back to newest checkpoint overall
        newest = max(after, key=lambda p: p.stat().st_mtime)
        print(f"CHECKPOINT={newest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
