#!/usr/bin/env bash
# End-to-end PAD-UFES-20 pipeline orchestrator (detached).
# Waits for the running sweep agents to finish, then:
#   1. select best config       2. retrain with checkpoint
#   3. embed all images          4. parse (SmSL tags)
#   5. build kNN graphs (k in K_VALUES)
#
# Designed to survive session/terminal close (launched via setsid/nohup).
set -u

GENV=/home/cnetto1/miniconda3/envs/graph_env/bin/python
REPO=/data/home/cnetto1/image-gnn
cd "$REPO" || exit 1

LOGDIR="$REPO/logs/pad_pipeline"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/pipeline.log"

PROJECT="GNN-image-VAE_train-PADUFES20"
ENTITY="caiodeberaldini"

TRAIN_CFG="config/cnnvae_pad_training_config.yaml"
DATA_CFG="config/cnnvae_pad_dataset_config.yaml"
BEST_MODEL_CFG="config/best_cnnvae_pad_model_config.yaml"

EMB_DIR="data/PadUfes20Embeddings"
SMSL_DIR="$EMB_DIR/smsl_embeddings"
GRAPH_DIR="data/PadUfes20Graph"
K_VALUES=(10 15 30)

# retrain runs single-GPU; pin to GPU 0
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

log "==================== PAD-UFES-20 pipeline start ===================="

# ---- 1. wait for sweep agents to exhaust the grid --------------------------
log "STEP 0: waiting for sweep agents (train_model.py) to finish..."
WAIT_MAX=$((6 * 3600))   # 6h safety cap
WAITED=0
while pgrep -f "train_model.py" > /dev/null 2>&1; do
    sleep 30
    WAITED=$((WAITED + 30))
    if [ "$WAITED" -ge "$WAIT_MAX" ]; then
        log "WARNING: sweep wait exceeded ${WAIT_MAX}s; proceeding anyway."
        break
    fi
    if [ $((WAITED % 300)) -eq 0 ]; then
        log "  ...still waiting (${WAITED}s elapsed)"
    fi
done
log "STEP 0 done: no sweep agents running."

# ---- 2. select best config -------------------------------------------------
log "STEP 1: selecting best config..."
$GENV src/scripts/pad_ufes20/select_best_pad.py \
    --project "$PROJECT" --entity "$ENTITY" --out "$BEST_MODEL_CFG" \
    >> "$LOG" 2>&1 || die "select_best_pad failed"
[ -f "$BEST_MODEL_CFG" ] || die "best model config not written"
log "STEP 1 done: $BEST_MODEL_CFG"

# ---- 3. retrain best with checkpoint --------------------------------------
log "STEP 2: retraining best config with checkpoint saving..."
RETRAIN_OUT="$LOGDIR/retrain.log"
$GENV src/scripts/pad_ufes20/retrain_best_pad.py \
    -d "$DATA_CFG" -t "$TRAIN_CFG" -m "$BEST_MODEL_CFG" \
    > "$RETRAIN_OUT" 2>&1 || die "retrain failed (see $RETRAIN_OUT)"
CKPT=$(grep '^CHECKPOINT=' "$RETRAIN_OUT" | tail -1 | cut -d= -f2-)
[ -n "$CKPT" ] && [ -f "$CKPT" ] || die "no checkpoint produced (see $RETRAIN_OUT)"
log "STEP 2 done: checkpoint = $CKPT"

# ---- 4. embed all images ---------------------------------------------------
log "STEP 3: embedding all images..."
$GENV src/scripts/pad_ufes20/gen_pad_embedding.py \
    -d "$DATA_CFG" -m "$BEST_MODEL_CFG" -c "$CKPT" -o "$EMB_DIR" \
    >> "$LOG" 2>&1 || die "embedding failed"
log "STEP 3 done: embeddings in $EMB_DIR"

# ---- 5. parse (SmSL tags) --------------------------------------------------
log "STEP 4: parsing embeddings (SmSL tags)..."
$GENV src/scripts/pad_ufes20/parse_pad_embedding.py \
    --in_dir "$EMB_DIR" --out_dir "$SMSL_DIR" --prefix pad \
    >> "$LOG" 2>&1 || die "parse failed"
log "STEP 4 done: smsl embeddings in $SMSL_DIR"

# ---- 6. build kNN graphs ---------------------------------------------------
for k in "${K_VALUES[@]}"; do
    log "STEP 5: building kNN graph (k=$k)..."
    $GENV src/scripts/pad_ufes20/gen_pad_knn_graph.py \
        --train "$SMSL_DIR/pad_smsl_train_embeddings.npy" \
        --test  "$SMSL_DIR/pad_smsl_test_embeddings.npy" \
        --out_dir "$GRAPH_DIR" -k "$k" \
        >> "$LOG" 2>&1 || die "knn graph (k=$k) failed"
    log "STEP 5 done (k=$k): graphs in $GRAPH_DIR"
done

log "==================== PAD-UFES-20 pipeline COMPLETE ===================="
log "Artifacts:"
log "  best config : $BEST_MODEL_CFG"
log "  checkpoint  : $CKPT"
log "  embeddings  : $EMB_DIR/pad_{train,test}_embeddings.npy"
log "  smsl        : $SMSL_DIR/"
log "  knn graphs  : $GRAPH_DIR/ (k = ${K_VALUES[*]})"
