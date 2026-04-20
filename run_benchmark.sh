#!/bin/bash
# Benchmark script wrapper for Qwen2.5-7B-Instruct
# Usage: ./run_benchmark.sh [dataset] [model_path] [num_samples]

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET=${1:-math500}
MODEL_PATH=${2:-}
NUM_SAMPLES=${3:-100}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Forest-of-Thought Benchmark Runner${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path is required${NC}"
    echo "Usage: $0 [dataset] [model_path] [num_samples]"
    echo ""
    echo "Examples:"
    echo "  $0 math500 /path/to/Qwen2.5-7B-Instruct 100"
    echo "  $0 aime /path/to/Qwen2.5-7B-Instruct 30"
    echo "  $0 gsm8k /path/to/Qwen2.5-7B-Instruct 500"
    exit 1
fi

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_PATH${NC}"
    exit 1
fi

# Set dataset-specific defaults
case $DATASET in
    math500)
        DATASET_PATH="datasets/math500/test.parquet"
        MAX_END_ID=500
        echo -e "${GREEN}Dataset: MATH500${NC}"
        ;;
    aime)
        DATASET_PATH="datasets/aime2024/aime_2024_problems.parquet"
        MAX_END_ID=30
        echo -e "${GREEN}Dataset: AIME 2024${NC}"
        ;;
    gsm8k)
        DATASET_PATH="datasets/gsm8k/test.parquet"
        MAX_END_ID=1319
        echo -e "${GREEN}Dataset: GSM8K${NC}"
        ;;
    *)
        echo -e "${RED}Error: Unknown dataset: $DATASET${NC}"
        echo "Supported datasets: math500, aime, gsm8k"
        exit 1
        ;;
esac

# Validate dataset path
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset file not found: $DATASET_PATH${NC}"
    exit 1
fi

# Validate number of samples
if [ "$NUM_SAMPLES" -gt "$MAX_END_ID" ]; then
    echo -e "${YELLOW}Warning: Requested samples ($NUM_SAMPLES) exceeds max available ($MAX_END_ID)${NC}"
    NUM_SAMPLES=$MAX_END_ID
fi

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Dataset: $DATASET"
echo "  Dataset Path: $DATASET_PATH"
echo "  Model: $(basename "$MODEL_PATH")"
echo "  Model Path: $MODEL_PATH"
echo "  Samples: 0-$NUM_SAMPLES"
echo ""

# Run benchmark
cd "$SCRIPT_DIR"

echo -e "${BLUE}Starting benchmark...${NC}"
python benchmark_qwen.py \
    --dataset "$DATASET" \
    --dataset_filepath "$DATASET_PATH" \
    --model_path "$MODEL_PATH" \
    --start_id 0 \
    --end_id "$NUM_SAMPLES" \
    --max_tokens 2048 \
    --output_dir "./benchmark_results"

echo ""
echo -e "${GREEN}Benchmark completed!${NC}"
echo -e "${BLUE}Results saved to: ./benchmark_results${NC}"
