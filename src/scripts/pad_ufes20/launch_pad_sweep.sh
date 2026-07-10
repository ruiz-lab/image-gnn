#!/usr/bin/env bash
# Launch parallel wandb sweep agents for the PAD-UFES-20 CNNVAE grid search.
# 8 agents total: 2 per GPU across GPUs 0-3 (never more than 4 GPUs).
set -u

GENV=/home/cnetto1/miniconda3/envs/graph_env/bin/python
REPO=/data/home/cnetto1/image-gnn
LOGDIR="$REPO/logs/pad_sweep"
mkdir -p "$LOGDIR"

export WANDB_SWEEP_ID="r50ublc2"
export WANDB_PROJECT="GNN-image-VAE_train-PADUFES20"

TRAIN_CFG="config/cnnvae_pad_training_config.yaml"
DATA_CFG="config/cnnvae_pad_dataset_config.yaml"
MODEL_CFG="config/cnnvae_pad_model_config.yaml"

cd "$REPO" || exit 1

GPUS=(0 1 2 3)
AGENTS_PER_GPU=2

pids=()
for gpu in "${GPUS[@]}"; do
    for a in $(seq 1 "$AGENTS_PER_GPU"); do
        log="$LOGDIR/agent_gpu${gpu}_${a}.log"
        CUDA_VISIBLE_DEVICES="$gpu" "$GENV" src/scripts/train_model.py \
            -t "$TRAIN_CFG" -d "$DATA_CFG" -m "$MODEL_CFG" \
            > "$log" 2>&1 &
        pids+=($!)
        echo "launched agent on GPU $gpu (instance $a) pid $! -> $log"
    done
done

echo "all ${#pids[@]} agents launched; waiting for sweep to finish..."
wait
echo "SWEEP COMPLETE"
