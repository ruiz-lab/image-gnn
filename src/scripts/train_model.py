import sys
import yaml
import wandb
import argparse

from pathlib import Path

from training.train import Trainer

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", 
        "--dataset_config", 
        type=str, 
        help="Training configuration file", 
        required=True
    )

    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        help="Dataset setup configuration file", 
        required=True
    )

    parser.add_argument(
        "-t", 
        "--training_config", 
        type=str, 
        help="Model configuration file", 
        required=True
    )
 
    return parser.parse_args(args)

def main():
    wandb.init()
    args = parse_args(sys.argv[1:])

    config_paths = {
        "dataset_config": args.dataset_config,
        "training_config": args.training_config,
        "model_config": args.model_config
    }

    config = {}
    for config_name, config_file_path in config_paths.items():
        with open(Path(config_file_path), "r") as f:
            config[config_name] = yaml.safe_load(f)

    config["model_config"]["args"] = {
        k: v for k, v in wandb.config.items()
    }

    trainer = Trainer(**config)
    trainer.train()

if __name__ == "__main__":
    sweep_id = "qk3zqvni"
    wandb.agent(
        sweep_id=sweep_id,
        function=main,
        project="GNN-image-VAE_train"
    )