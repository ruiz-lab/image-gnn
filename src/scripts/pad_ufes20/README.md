# PAD-UFES-20 — CNNVAE embeddings & kNN graph pipeline

Scripts for processing the **PAD-UFES-20** skin-lesion dataset (2054 clinical
smartphone images, RGB 224×224) into the CNNVAE latent embeddings and kNN
"data-manifold" graphs consumed by the downstream GNN.

The dataset is loaded via `PadUfes20Dataset` in
[`src/scripts/data_preproc/datasets.py`](../data_preproc/datasets.py); the
CNNVAE encoder/decoder gained an `img_size` argument so the same architecture
handles the native 224×224 images (not just the 28×28 datasets).

## Configs

The pipeline is config-driven. The relevant files in `config/`:

| File                                | Purpose                                              |
| ----------------------------------- | ---------------------------------------------------- |
| `cnnvae_pad_dataset_config.yaml`    | `PadUfes20Dataset` train/test split (80/20, seed 42) |
| `cnnvae_pad_model_config.yaml`      | CNNVAE architecture for the sweep (`img_size: 224`)  |
| `cnnvae_pad_training_config.yaml`   | learning rate / batch size / epochs / loss           |
| `wandb_cnnvae_pad_sweep_config.yaml`| wandb grid over `latent_size` × `blocks`             |
| `best_cnnvae_pad_model_config.yaml` | best architecture selected from the sweep (written by `select_best_pad.py`) |

## Scripts

| Script                   | Purpose                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| `launch_pad_sweep.sh`    | Launch parallel wandb sweep agents (CNNVAE grid search, 2 agents/GPU).         |
| `select_best_pad.py`     | Pick the min-`test_loss` sweep run and write its arch to a best-model config.  |
| `retrain_best_pad.py`    | Retrain the best config **with checkpointing** (the sweep itself doesn't save).|
| `gen_pad_embedding.py`   | Run the encoder over the split, save `[latent..., label]` `.npy` embeddings.   |
| `parse_pad_embedding.py` | Append the SmSL train/test tag column the graph builder expects.               |
| `gen_pad_knn_graph.py`   | Build approximate-kNN Gaussian-kernel graphs, threshold + symmetrize.          |
| `recon_pad.py`           | Original-vs-reconstruction figures for the trained CNNVAE (QA).                |
| `analyze_pad_graphs.py`  | Per-`k` graph diagnostics: degree/weight stats, homophily, components.         |
| `run_pad_pipeline.sh`    | End-to-end orchestrator chaining the steps below.                              |

## Running the full pipeline

### 1. Sweep the CNNVAE architecture

Create the wandb sweep from `config/wandb_cnnvae_pad_sweep_config.yaml`, then set
`WANDB_SWEEP_ID` (and `WANDB_PROJECT`) at the top of `launch_pad_sweep.sh` and
launch the agents:

```bash
bash src/scripts/pad_ufes20/launch_pad_sweep.sh
```

Agents run `src/scripts/train_model.py`, which reads `WANDB_SWEEP_ID` /
`WANDB_PROJECT` from the environment.

### 2. Select → retrain → embed → parse → build graphs

Once the sweep grid is exhausted, the orchestrator runs the rest end-to-end
(it first waits for the sweep agents to finish):

```bash
bash src/scripts/pad_ufes20/run_pad_pipeline.sh
```

which performs:

1. `select_best_pad.py` — best config → `best_cnnvae_pad_model_config.yaml`
2. `retrain_best_pad.py` — retrain best config, save a checkpoint
3. `gen_pad_embedding.py` — embed all images → `data/PadUfes20Embeddings/`
4. `parse_pad_embedding.py` — add SmSL tags → `.../smsl_embeddings/`
5. `gen_pad_knn_graph.py` — build kNN graphs for `k ∈ {10, 15, 30}` → `data/PadUfes20Graph/`

### Running steps individually

Each script takes explicit `--`/`-` flags, so any step can be run on its own —
see the usage docstring at the top of each file. Example (build a graph for a
single `k`):

```bash
python src/scripts/pad_ufes20/gen_pad_knn_graph.py \
    --train data/PadUfes20Embeddings/smsl_embeddings/pad_smsl_train_embeddings.npy \
    --test  data/PadUfes20Embeddings/smsl_embeddings/pad_smsl_test_embeddings.npy \
    --out_dir data/PadUfes20Graph -k 15
```

## Tuning hyperparameters

- **CNNVAE architecture** — edit the grid in `wandb_cnnvae_pad_sweep_config.yaml`
  (`latent_size`, `blocks`). `select_best_pad.py` picks the min-`test_loss` run.
- **Training** — `learning_rate`, `batch_size`, `num_epochs` in
  `cnnvae_pad_training_config.yaml`.
- **Graph construction** — `gen_pad_knn_graph.py` flags: `-k` (neighbors), the
  Gaussian-kernel width (auto median-distance heuristic by default), and the
  edge-weight threshold. Use `analyze_pad_graphs.py` to inspect the resulting
  degree distribution, sparsity, and class homophily per `k`.

## Diagnostics

- `recon_pad.py` — sanity-check reconstruction quality of the trained CNNVAE.
- `analyze_pad_graphs.py` — per-`k` graph statistics (degrees, edge-weight
  distribution, threshold sparsity, edge homophily, connected components, class
  balance). Figures are written to `recon_images/`.
