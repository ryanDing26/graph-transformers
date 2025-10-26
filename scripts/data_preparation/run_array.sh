#!/bin/bash
#SBATCH --job-name=wsi_nucleus_array
#SBATCH --output=logs/nucleus_array_%A_%a.out
#SBATCH --error=logs/nucleus_array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-normal
#SBATCH --array=0-249


# ============================================================================
# SLURM Array Job Script for Parallel WSI Processing
# Each array task processes a subset of .svs files in parallel
# 
# To determine --array range:
#   - Count .svs files: ls /path/to/slides/*.svs | wc -l
#   - If you have 50 files and want 5 files per job: --array=0-9 (10 jobs)
#   - If you have 50 files and want 1 file per job: --array=0-49 (50 jobs)
# ============================================================================

echo "================================================"
echo "Array Job started at: $(date)"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "================================================"

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Input/Output directories
INPUT_DIR="/shares/sinha/anamikay/GTEX_ovary_histology/"          # Directory containing .svs files
OUTPUT_DIR="/shares/sinha/rding/graph-transformers/outputs"      # Where to save results

# Array job configuration
FILES_PER_JOB=1
                  # Set to 1 for maximum parallelism
                  # Set to 5-10 if you have limited GPU resources

# Processing parameters
TILE_SIZE=512
MPP=0.5
BATCH_SIZE=32
K_NEIGHBORS=10
MAX_DISTANCE=500
USE_PE=""  # Add "--use_pe" to enable Laplacian positional encodings

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Activate conda/virtual environment
source /opt/apps/anaconda3/bin/activate
conda activate histograph

# Print environment info
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"

# ============================================================================
# RUN PROCESSING FOR THIS ARRAY TASK
# ============================================================================

echo ""
echo "Array task $SLURM_ARRAY_TASK_ID processing..."
echo "Files per job: $FILES_PER_JOB"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create logs directory
mkdir -p logs

# Run the batch processing script with array indexing
python batch_process_wsi.py \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    --tile_size $TILE_SIZE \
    --mpp $MPP \
    --batch_size $BATCH_SIZE \
    --k_neighbors $K_NEIGHBORS \
    --max_distance $MAX_DISTANCE \
    --array_idx $SLURM_ARRAY_TASK_ID \
    --files_per_job $FILES_PER_JOB \
    $USE_PE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Array task $SLURM_ARRAY_TASK_ID completed at: $(date)"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "Array task $SLURM_ARRAY_TASK_ID failed at: $(date)"
    echo "================================================"
    exit 1
fi