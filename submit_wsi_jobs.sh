#!/bin/bash
# submit_wsi_jobs.sh
# Helper script to submit WSI processing jobs to SLURM
# run with ./submit_wsi_jobs.sh /shares/sinha/anamikay/GTEX_ovary_histology/ /shares/sinha/rding/graph-transformers/outputs array
set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR="${1:-/path/to/wsi/slides}"
OUTPUT_DIR="${2:-/path/to/output/results}"
MODE="${3:-array}"  # "sequential" or "array"

# ============================================================================
# FUNCTIONS
# ============================================================================

count_svs_files() {
    local dir=$1
    find "$dir" -maxdepth 1 -name "*.svs" | wc -l
}

calculate_array_size() {
    local n_files=$1
    local files_per_job=$2
    echo $(( (n_files + files_per_job - 1) / files_per_job - 1 ))
}

print_header() {
    echo "================================================"
    echo "$1"
    echo "================================================"
}

# ============================================================================
# MAIN
# ============================================================================

print_header "WSI Nucleus Graph Processing - Job Submission"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Count .svs files
N_FILES=$(count_svs_files "$INPUT_DIR")
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of .svs files: $N_FILES"
echo ""

if [ $N_FILES -eq 0 ]; then
    echo "Error: No .svs files found in $INPUT_DIR"
    exit 1
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Submit appropriate job type
if [ "$MODE" == "sequential" ]; then
    # ========================================================================
    # SEQUENTIAL PROCESSING (Single Job)
    # ========================================================================
    print_header "Submitting Sequential Job"
    
    echo "Processing mode: Sequential (single node)"
    echo "Estimated time: ~1-2 hours per slide"
    echo ""
    
    # Update directories in script
    sed -i "s|INPUT_DIR=\".*\"|INPUT_DIR=\"$INPUT_DIR\"|g" run_sequential.sh
    sed -i "s|OUTPUT_DIR=\".*\"|OUTPUT_DIR=\"$OUTPUT_DIR\"|g" run_sequential.sh
    
    # Submit job
    JOB_ID=$(sbatch run_sequential.sh | awk '{print $4}')
    
    echo "✓ Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/nucleus_graph_${JOB_ID}.out"
    
elif [ "$MODE" == "array" ]; then
    # ========================================================================
    # ARRAY JOB PROCESSING (Parallel)
    # ========================================================================
    print_header "Submitting Array Job"
    
    # Ask for files per job
    read -p "Files per job [default: 1]: " FILES_PER_JOB
    FILES_PER_JOB=${FILES_PER_JOB:-1}
    
    # Calculate array size
    ARRAY_MAX=$(calculate_array_size $N_FILES $FILES_PER_JOB)
    
    echo "Processing mode: Parallel (array job)"
    echo "Files per job: $FILES_PER_JOB"
    echo "Array size: 0-$ARRAY_MAX ($(($ARRAY_MAX + 1)) tasks)"
    echo "Estimated time per task: ~1-2 hours × $FILES_PER_JOB slides"
    echo ""
    
    # Update script
    sed -i "s|INPUT_DIR=\".*\"|INPUT_DIR=\"$INPUT_DIR\"|g" run_array.sh
    sed -i "s|OUTPUT_DIR=\".*\"|OUTPUT_DIR=\"$OUTPUT_DIR\"|g" run_array.sh
    sed -i "s|FILES_PER_JOB=.*|FILES_PER_JOB=$FILES_PER_JOB|g" run_array.sh
    sed -i "s|#SBATCH --array=.*|#SBATCH --array=0-$ARRAY_MAX|g" run_array.sh
    
    # Confirm submission
    echo "Ready to submit $((ARRAY_MAX + 1)) parallel jobs"
    read -p "Continue? (y/n): " CONFIRM
    
    if [ "$CONFIRM" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
    
    # Submit job
    JOB_ID=$(sbatch run_array.sh | awk '{print $4}')
    
    echo ""
    echo "✓ Array job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/nucleus_array_${JOB_ID}_*.out"
    echo ""
    echo "Check completed tasks:"
    echo "  sacct -j $JOB_ID --format=JobID,State,Elapsed,MaxRSS"
    
else
    echo "Error: Invalid mode '$MODE'. Use 'sequential' or 'array'"
    exit 1
fi

echo ""
print_header "Job Submission Complete"
echo ""
echo "Output will be saved to: $OUTPUT_DIR"
echo "Logs are in: logs/"
echo ""
echo "To cancel job(s): scancel $JOB_ID"
echo ""