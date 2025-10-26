#!/bin/bash
# Complete end-to-end pipeline example
# From WSI slides to spatial embedding visualizations

set -e  # Exit on error

echo "================================================================"
echo "COMPLETE NUCLEUS GRAPH + DINO PIPELINE"
echo "================================================================"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
WSI_DIR="/shares/sinha/anamikay/GTEX_ovary_histology"
OUTPUT_DIR="/shares/sinha/rding/graph-transformers/outputs"
CHECKPOINT_DIR="/shares/sinha/rding/graph-transformers/dino_checkpoints"
VIZ_DIR="/shares/sinha/rding/graph-transformers/visualizations"
EMBEDDING_DIR="/shares/sinha/rding/graph-transformers/embeddings"

# ============================================================================
# STEP 1: Process WSI to Graphs (if not already done)
# ============================================================================

echo "Step 1: WSI Processing → Nucleus Graphs"
echo "=========================================="
echo ""

# Check if graphs already exist
N_GRAPHS=$(find "$OUTPUT_DIR" -name "nucleus_graph.pkl" 2>/dev/null | wc -l)

if [ $N_GRAPHS -gt 0 ]; then
    echo "✓ Found $N_GRAPHS existing graphs in $OUTPUT_DIR"
    echo "  Skipping WSI processing..."
else
    echo "No existing graphs found. Processing WSIs..."
    echo ""
    
    # Submit array job to process all WSIs in parallel
    echo "Submitting SLURM array job for WSI processing..."
    JOB_ID=$(sbatch run_array.sh | awk '{print $4}')
    echo "  Job ID: $JOB_ID"
    echo "  Monitor with: squeue -j $JOB_ID"
    echo ""
    
    echo "Waiting for WSI processing to complete..."
    echo "(This may take several hours)"
    
    # Wait for job to complete
    while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
        sleep 60
        N_COMPLETE=$(find "$OUTPUT_DIR" -name "nucleus_graph.pkl" 2>/dev/null | wc -l)
        echo "  Progress: $N_COMPLETE graphs completed..."
    done
    
    N_GRAPHS=$(find "$OUTPUT_DIR" -name "nucleus_graph.pkl" 2>/dev/null | wc -l)
    echo ""
    echo "✓ WSI processing complete: $N_GRAPHS graphs generated"
fi

echo ""

# ============================================================================
# STEP 2: Train DINO Model
# ============================================================================

echo "Step 2: Training DINO Model"
echo "============================"
echo ""

CHECKPOINT="$CHECKPOINT_DIR/checkpoint_epoch_100.pt"

if [ -f "$CHECKPOINT" ]; then
    echo "✓ Found existing checkpoint: $CHECKPOINT"
    echo "  Skipping training..."
else
    echo "No checkpoint found. Starting DINO training..."
    echo ""
    
    # Submit training job
    echo "Submitting SLURM job for DINO training..."
    JOB_ID=$(sbatch run_dino_training.sh | awk '{print $4}')
    echo "  Job ID: $JOB_ID"
    echo "  Monitor with: squeue -j $JOB_ID"
    echo "  Logs: tail -f logs/dino_${JOB_ID}.out"
    echo ""
    
    echo "Training in progress..."
    echo "(This may take 12-24 hours)"
    echo ""
    echo "You can monitor progress with tensorboard:"
    echo "  tensorboard --logdir $CHECKPOINT_DIR/logs"
    echo ""
    
    # Wait for training to complete
    while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
        sleep 300  # Check every 5 minutes
    done
    
    if [ -f "$CHECKPOINT" ]; then
        echo "✓ Training complete: $CHECKPOINT"
    else
        echo "✗ Training failed - checkpoint not found"
        exit 1
    fi
fi

echo ""

# ============================================================================
# STEP 3: Generate Slide-Level Embeddings
# ============================================================================

echo "Step 3: Generating Slide-Level Embeddings"
echo "=========================================="
echo ""

EMBEDDINGS_FILE="$EMBEDDING_DIR/slide_embeddings.npz"

if [ -f "$EMBEDDINGS_FILE" ]; then
    echo "✓ Found existing embeddings: $EMBEDDINGS_FILE"
    echo "  Skipping embedding generation..."
else
    echo "Generating embeddings for all slides..."
    mkdir -p "$EMBEDDING_DIR"
    
    python infer_embeddings.py \
        --checkpoint "$CHECKPOINT" \
        --graph_dir "$OUTPUT_DIR" \
        --output_file "$EMBEDDINGS_FILE" \
        --num_samples 10 \
        --aggregation mean \
        --device cuda
    
    if [ -f "$EMBEDDINGS_FILE" ]; then
        echo "✓ Embeddings generated: $EMBEDDINGS_FILE"
    else
        echo "✗ Embedding generation failed"
        exit 1
    fi
fi

echo ""

# ============================================================================
# STEP 4: Generate Spatial Embedding Maps
# ============================================================================

echo "Step 4: Creating Spatial Embedding Maps"
echo "========================================"
echo ""

# Check if visualizations exist
N_VIZ=$(find "$VIZ_DIR" -name "*_embedding_map.png" 2>/dev/null | wc -l)

if [ $N_VIZ -gt 0 ]; then
    echo "✓ Found $N_VIZ existing visualizations in $VIZ_DIR"
    read -p "Regenerate? (y/n): " REGEN
    if [ "$REGEN" != "y" ]; then
        echo "  Skipping visualization..."
        echo ""
        echo "================================================================"
        echo "PIPELINE COMPLETE"
        echo "================================================================"
        exit 0
    fi
fi

echo "Generating spatial embedding maps for all slides..."
echo ""

mkdir -p "$VIZ_DIR"

python batch_visualize_embeddings.py \
    --checkpoint "$CHECKPOINT" \
    --graph_dir "$OUTPUT_DIR" \
    --output_dir "$VIZ_DIR" \
    --mode subgraph \
    --patch_size 1000.0 \
    --overlap 0.5 \
    --viz_method cluster \
    --n_clusters 10 \
    --device cuda

echo ""

N_VIZ=$(find "$VIZ_DIR" -name "*_embedding_map.png" 2>/dev/null | wc -l)

if [ $N_VIZ -gt 0 ]; then
    echo "✓ Generated $N_VIZ spatial embedding maps"
else
    echo "✗ Visualization generation failed"
    exit 1
fi

echo ""

# ============================================================================
# STEP 5: Create Summary Report
# ============================================================================

echo "Step 5: Creating Summary Report"
echo "================================"
echo ""

REPORT_FILE="$OUTPUT_DIR/pipeline_summary.txt"

cat > "$REPORT_FILE" << EOF
NUCLEUS GRAPH + DINO PIPELINE SUMMARY
Generated: $(date)

STEP 1: WSI Processing
- Input directory: $WSI_DIR
- Output directory: $OUTPUT_DIR
- Graphs generated: $N_GRAPHS

STEP 2: DINO Training
- Checkpoint: $CHECKPOINT
- Architecture: 256-dim hidden, 8 heads, 4 layers
- Output dimension: 1280
- Training: 100 epochs

STEP 3: Slide Embeddings
- Embeddings file: $EMBEDDINGS_FILE
- Number of slides: $N_GRAPHS
- Embedding dimension: 1280

STEP 4: Spatial Visualizations
- Output directory: $VIZ_DIR
- Visualizations generated: $N_VIZ
- Method: Subgraph clustering (10 clusters)
- Patch size: 1000 microns

OUTPUTS:
- Nucleus graphs: $OUTPUT_DIR/*/nucleus_graph.pkl
- Features: $OUTPUT_DIR/*/nucleus_features.csv
- DINO checkpoint: $CHECKPOINT
- Slide embeddings: $EMBEDDINGS_FILE
- Spatial maps: $VIZ_DIR/*_embedding_map.png
- Embedding data: $VIZ_DIR/*_embedding_data.npz

NEXT STEPS:
1. View spatial maps in $VIZ_DIR
2. Analyze slide embeddings for classification/clustering
3. Compare patterns across different tissue samples
4. Correlate with clinical/demographic data

For more information, see:
- TRAINING_GUIDE.md
- SPATIAL_VISUALIZATION_GUIDE.md
EOF

echo "✓ Summary report saved to: $REPORT_FILE"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================"
echo ""
echo "Summary:"
echo "  - Processed $N_GRAPHS WSI slides"
echo "  - Trained DINO model for 100 epochs"
echo "  - Generated embeddings for all slides"
echo "  - Created $N_VIZ spatial embedding maps"
echo ""
echo "Outputs:"
echo "  📊 Graphs: $OUTPUT_DIR"
echo "  🧠 Model: $CHECKPOINT"
echo "  📈 Embeddings: $EMBEDDINGS_FILE"
echo "  🎨 Visualizations: $VIZ_DIR"
echo ""
echo "View Results:"
echo "  Spatial maps: ls $VIZ_DIR/*.png"
echo "  Tensorboard: tensorboard --logdir $CHECKPOINT_DIR/logs"
echo ""
echo "Documentation:"
echo "  Training: TRAINING_GUIDE.md"
echo "  Visualization: SPATIAL_VISUALIZATION_GUIDE.md"
echo ""
echo "================================================================"