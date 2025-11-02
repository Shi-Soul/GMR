#!/bin/bash

# Batch motion interpolation script
# Usage: ./batch_interpolate_motion.sh <input_dir> [additional args for motion_interpolation_pkl.py]
# Example: ./batch_interpolate_motion.sh /home/bai/GMR/data/motion-cjx/motion_g1_pbhc --start_inter_frame 30 --end_inter_frame 30 --default_pose /path/to/pose.npy

set -e  # Exit on error

# Check if input directory is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_dir> [additional args...]"
    echo "Example: $0 /home/bai/GMR/data/motion-cjx/motion_g1_pbhc --start_inter_frame 30 --end_inter_frame 30"
    exit 1
fi

INPUT_DIR="$1"
shift  # Remove first argument, keep the rest for motion_interpolation_pkl.py

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Set output directory
OUTPUT_DIR="${INPUT_DIR}_interp"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the script directory (where motion_interpolation_pkl.py is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERP_SCRIPT="$SCRIPT_DIR/motion_interpolation_pkl.py"

# Validate interpolation script exists
if [ ! -f "$INTERP_SCRIPT" ]; then
    echo "Error: Interpolation script not found: $INTERP_SCRIPT"
    exit 1
fi

# Counter for tracking progress
total_files=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pkl" | wc -l)
current_file=0

echo "=========================================="
echo "Batch Motion Interpolation"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Total files to process: $total_files"
echo "Additional arguments: $*"
echo "=========================================="
echo ""

# Process each .pkl file in the input directory
# Use process substitution to avoid subshell issue with counter
while IFS= read -r input_file; do
    current_file=$((current_file + 1))
    
    # Get filename without path and extension
    filename=$(basename "$input_file")
    filename_no_ext="${filename%.pkl}"
    
    # Set output path
    output_file="$OUTPUT_DIR/$filename_no_ext"
    
    echo "[$current_file/$total_files] Processing: $filename"
    
    # Build command with all additional arguments
    python "$INTERP_SCRIPT" \
        --origin_file_name "$input_file" \
        --output "$output_file" \
        "$@"
    
    echo "  -> Output: $output_file.pkl"
    echo ""
done < <(find "$INPUT_DIR" -maxdepth 1 -name "*.pkl" -type f | sort)

echo "=========================================="
echo "Batch processing completed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

