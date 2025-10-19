#!/bin/bash
#SBATCH --job-name=dino_nucleus
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-normal

# ============================================================================
# DINO Training Script for Large Nucleus Graphs
# Trains student-teacher transformer with self-supervised learning
# ============================================================================

echo "================================================"
echo "DINO Training Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "================================================"

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Paths
DATA_DIR="/shares/sinha/rding/graph-transformers/outputs"
OUTPUT_DIR="/shares/sinha/rding/graph-transformers/model_checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"

# Model Architecture
HIDDEN_DIM=256
NUM_HEADS=8
NUM_LAYERS=4
OUTPUT_DIM=1280

# DINO Parameters
DINO_OUT_DIM=65536
TEACHER_TEMP=0.04
STUDENT_TEMP=0.1

# Subgraph Sampling
SAMPLING_STRATEGY="spatial"  # spatial, khop, or random_walk
SUBGRAPH_SIZE=1000
SUBGRAPHS_PER_GRAPH=4
BATCH_SIZE=8

# Training
EPOCHS=100
LR=0.0001
WEIGHT_DECAY=0.04
WARMUP_EPOCHS=10

# System
DEVICE="cuda"
SEED=42

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

source /opt/apps/anaconda3/bin/activate
conda activate histograph

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p logs

# ============================================================================
# COUNT AVAILABLE GRAPHS
# ============================================================================

echo ""
echo "Checking data..."
N_GRAPHS=$(find "$DATA_DIR" -name "nucleus_graph.pkl" | wc -l)
echo "  Data directory: $DATA_DIR"
echo "  Found graphs: $N_GRAPHS"

if [ $N_GRAPHS -eq 0 ]; then
    echo "ERROR: No graphs found!"
    exit 1
fi

# ============================================================================
# TRAINING PARAMETERS SUMMARY
# ============================================================================

echo ""
echo "Training Configuration:"
echo "  Model:"
echo "    Hidden dim: $HIDDEN_DIM"
echo "    Attention heads: $NUM_HEADS"
echo "    Transformer layers: $NUM_LAYERS"
echo "    Output dim: $OUTPUT_DIM"
echo ""
echo "  DINO:"
echo "    Projection dim: $DINO_OUT_DIM"
echo "    Teacher temp: $TEACHER_TEMP"
echo "    Student temp: $STUDENT_TEMP"
echo ""
echo "  Sampling:"
echo "    Strategy: $SAMPLING_STRATEGY"
echo "    Subgraph size: $SUBGRAPH_SIZE nodes"
echo "    Subgraphs/graph: $SUBGRAPHS_PER_GRAPH"
echo "    Batch size: $BATCH_SIZE subgraphs"
echo ""
echo "  Training:"
echo "    Epochs: $EPOCHS"
echo "    Learning rate: $LR"
echo "    Weight decay: $WEIGHT_DECAY"
echo "    Warmup epochs: $WARMUP_EPOCHS"
echo ""

# Estimate training stats
TOTAL_SUBGRAPHS=$((N_GRAPHS * SUBGRAPHS_PER_GRAPH))
BATCHES_PER_EPOCH=$((TOTAL_SUBGRAPHS / BATCH_SIZE))
TOTAL_BATCHES=$((BATCHES_PER_EPOCH * EPOCHS))

echo "  Estimated stats:"
echo "    Subgraphs per epoch: $TOTAL_SUBGRAPHS"
echo "    Batches per epoch: $BATCHES_PER_EPOCH"
echo "    Total batches: $TOTAL_BATCHES"
echo "    Est. time: ~$((BATCHES_PER_EPOCH * EPOCHS / 2 / 3600)) hours (0.5s/batch)"
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "================================================"
echo "Starting DINO Training"
echo "================================================"
echo ""

python train_dino_subgraphs.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_dim $HIDDEN_DIM \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --output_dim $OUTPUT_DIM \
    --dino_out_dim $DINO_OUT_DIM \
    --teacher_temp $TEACHER_TEMP \
    --student_temp $STUDENT_TEMP \
    --sampling_strategy $SAMPLING_STRATEGY \
    --subgraph_size $SUBGRAPH_SIZE \
    --subgraphs_per_graph $SUBGRAPHS_PER_GRAPH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --device $DEVICE \
    --seed $SEED

TRAIN_EXIT_CODE=$?

echo ""
echo "================================================"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at: $(date)"
    echo "================================================"
    
    # ========================================================================
    # GENERATE EMBEDDINGS FOR ALL SLIDES
    # ========================================================================
    
    echo ""
    echo "Generating embeddings for all slides..."
    
    CHECKPOINT="$OUTPUT_DIR/checkpoint_epoch_${EPOCHS}.pt"
    EMBEDDINGS_FILE="$OUTPUT_DIR/slide_embeddings.npz"
    
    if [ -f "$CHECKPOINT" ]; then
        python infer_embeddings.py \
            --checkpoint "$CHECKPOINT" \
            --graph_dir "$DATA_DIR" \
            --output_file "$EMBEDDINGS_FILE" \
            --num_samples 10 \
            --aggregation mean \
            --device $DEVICE
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Embeddings saved to: $EMBEDDINGS_FILE"
        else
            echo "⚠ Embedding generation failed"
        fi
    else
        echo "⚠ Checkpoint not found: $CHECKPOINT"
    fi
    
else
    echo "Training failed at: $(date)"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "================================================"
    exit 1
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================"
echo "COMPLETE - Summary"
echo "================================================"
echo "  Checkpoints: $OUTPUT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Embeddings: $EMBEDDINGS_FILE"
echo ""
echo "View tensorboard with:"
echo "  tensorboard --logdir $LOG_DIR"
echo ""
echo "================================================"