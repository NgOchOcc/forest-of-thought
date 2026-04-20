# Benchmark Guide - Qwen2.5-7B-Instruct

This guide explains how to use the benchmark scripts for evaluating Qwen2.5-7B-Instruct on math reasoning datasets (MATH500, AIME, GSM8K).

## Quick Start

### 1. Basic Usage (Python)

```bash
python benchmark_qwen.py \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --dataset math500 \
    --end_id 100
```

### 2. Using Shell Script Wrapper

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh math500 /path/to/Qwen2.5-7B-Instruct 100
```

### 3. Using Advanced Runner (Presets)

```bash
python benchmark_runner.py --model /path/to/Qwen2.5-7B-Instruct --preset math500-quick
```

---

## Scripts Overview

### `benchmark_qwen.py` (Main Script)

The core benchmark script that runs inference and evaluation.

**Key Features:**
- Loads Qwen2.5-7B-Instruct model
- Supports math500, aime, and gsm8k datasets
- Extracts and validates answers
- Computes accuracy metrics
- Saves detailed results

**Usage:**
```bash
python benchmark_qwen.py [OPTIONS]
```

**All Options:**
```bash
python benchmark_qwen.py \
    --dataset {math500, aime, gsm8k}          # Dataset to benchmark (default: math500)
    --dataset_filepath PATH                    # Path to dataset file (auto-detected if not provided)
    --model_path PATH                          # Path to Qwen2.5-7B-Instruct model (REQUIRED)
    --start_id INT                             # Start index (default: 0)
    --end_id INT                               # End index (default: 500)
    --max_tokens INT                           # Max generation tokens (default: 2048)
    --temperature FLOAT                        # Temperature (default: 0.3)
    --top_p FLOAT                              # Top-p (default: 0.9)
    --output_dir PATH                          # Output directory (default: ./benchmark_results)
    --dynamic_self_correction                  # Enable self-correction
    --correct_threshold FLOAT                  # Confidence threshold (default: -0.5)
```

**Example:**
```bash
python benchmark_qwen.py \
    --dataset math500 \
    --model_path /home/models/Qwen2.5-7B-Instruct \
    --start_id 0 \
    --end_id 200 \
    --output_dir ./results
```

---

### `run_benchmark.sh` (Shell Wrapper)

Convenient shell script wrapper for quick benchmarking.

**Usage:**
```bash
./run_benchmark.sh [DATASET] [MODEL_PATH] [NUM_SAMPLES]
```

**Arguments:**
- `DATASET`: math500, aime, or gsm8k
- `MODEL_PATH`: Path to model directory
- `NUM_SAMPLES`: Number of samples to run (optional, default: 100)

**Examples:**
```bash
# MATH500 - 100 samples
./run_benchmark.sh math500 /path/to/Qwen2.5-7B-Instruct 100

# AIME - all 30 samples
./run_benchmark.sh aime /path/to/Qwen2.5-7B-Instruct 30

# GSM8K - 500 samples
./run_benchmark.sh gsm8k /path/to/Qwen2.5-7B-Instruct 500
```

---

### `benchmark_runner.py` (Advanced Runner)

Advanced runner with preset configurations for quick benchmarking.

**Usage:**
```bash
python benchmark_runner.py --model MODEL_PATH [OPTIONS]
```

**Options:**
```bash
--preset {math500-quick, math500-standard, math500-full, aime-full, gsm8k-quick, gsm8k-standard}
--all                          # Run all presets
--list                         # List available presets
--dataset {math500, aime, gsm8k}  # Custom dataset
--samples INT                  # Number of samples for custom benchmark
--output-dir PATH              # Output directory
```

**Examples:**

List all presets:
```bash
python benchmark_runner.py --model /path/to/model --list
```

Run MATH500 quick preset (50 samples):
```bash
python benchmark_runner.py --model /path/to/model --preset math500-quick
```

Run MATH500 full benchmark (all 500 samples):
```bash
python benchmark_runner.py --model /path/to/model --preset math500-full
```

Run all benchmarks sequentially:
```bash
python benchmark_runner.py --model /path/to/model --all
```

Run custom benchmark (AIME with 30 samples):
```bash
python benchmark_runner.py --model /path/to/model --dataset aime --samples 30
```

---

## Available Presets

| Preset | Dataset | Samples | Use Case |
|--------|---------|---------|----------|
| `math500-quick` | MATH500 | 50 | Quick testing |
| `math500-standard` | MATH500 | 200 | Standard evaluation |
| `math500-full` | MATH500 | 500 | Full benchmark |
| `aime-full` | AIME 2024 | 30 | Full AIME evaluation |
| `gsm8k-quick` | GSM8K | 100 | Quick GSM8K test |
| `gsm8k-standard` | GSM8K | 500 | Standard GSM8K evaluation |

---

## Dataset Information

### MATH500
- **Path:** `datasets/math500/test.parquet`
- **Size:** 500 problems
- **Subjects:** Algebra, Geometry, Precalculus, etc.
- **Levels:** 1-5 (difficulty)

### AIME 2024
- **Path:** `datasets/aime2024/aime_2024_problems.parquet`
- **Size:** ~30 problems
- **Format:** Competition problems with numerical answers

### GSM8K
- **Path:** `datasets/gsm8k/test.parquet`
- **Size:** 1,319 samples
- **Type:** Grade school math word problems

---

## Output Structure

Results are saved to `./benchmark_results` (or custom `--output_dir`):

```
benchmark_results/
├── MATH500_qwen2.5-7b-instruct_20240520_143022/
│   ├── benchmark_results.json      # Detailed results for all samples
│   └── summary.json                # Summary statistics
├── AIME_qwen2.5-7b-instruct_20240520_145011/
│   ├── benchmark_results.json
│   └── summary.json
└── ...
```

### Result Format

**benchmark_results.json:**
```json
{
  "config": {
    "dataset": "math500",
    "model_path": "/path/to/model",
    "max_tokens": 2048,
    "num_samples": 100,
    "timestamp": "2024-05-20T14:30:22.123456"
  },
  "results": [
    {
      "sample_id": 0,
      "query": "Convert the point (0,3) in rectangular coordinates to polar coordinates.",
      "ground_truth": "\\left( 3, \\frac{\\pi}{2} \\right)",
      "inference_result": {
        "success": true,
        "is_correct": true,
        "extracted_answer": "( 3, π/2 )",
        "confidence": 0.95,
        "inference_time": 2.34,
        "error": null,
        "response": "..."  // Full model response
      }
    },
    ...
  ],
  "summary": {
    "total_samples": 100,
    "correct_count": 87,
    "accuracy": "87.00%",
    "total_inference_time": "234.56s",
    "average_inference_time": "2.35s",
    "samples_per_second": "0.43"
  }
}
```

**summary.json:**
```json
{
  "config": { ... },
  "summary": {
    "total_samples": 100,
    "correct_count": 87,
    "accuracy": "87.00%",
    "total_inference_time": "234.56s",
    "average_inference_time": "2.35s",
    "samples_per_second": "0.43",
    "benchmark_datetime": "2024-05-20T14:30:22.123456"
  }
}
```

---

## Common Commands

### Run Quick Test (10 samples)
```bash
python benchmark_qwen.py \
    --model_path /path/to/model \
    --dataset math500 \
    --end_id 10 \
    --output_dir ./test_results
```

### Run on Specific Subset
```bash
python benchmark_qwen.py \
    --model_path /path/to/model \
    --dataset math500 \
    --start_id 100 \
    --end_id 200
```

### Run with Self-Correction
```bash
python benchmark_qwen.py \
    --model_path /path/to/model \
    --dataset aime \
    --dynamic_self_correction \
    --correct_threshold 0.5 \
    --end_id 30
```

### Benchmark Multiple Datasets
```bash
# MATH500
python benchmark_qwen.py --model_path /path/to/model --dataset math500 --end_id 500

# AIME
python benchmark_qwen.py --model_path /path/to/model --dataset aime --end_id 30

# GSM8K
python benchmark_qwen.py --model_path /path/to/model --dataset gsm8k --end_id 1000
```

---

## Performance Recommendations

### For GPU Memory Optimization
- **Reduce `--max_tokens`:** Lower to 1024 if memory-constrained
- **Reduce `--end_id`:** Run in batches (e.g., 100 samples at a time)

### For Faster Benchmarking
- Use smaller `--end_id` values (50-100)
- Run on GPU with sufficient VRAM
- Use `--temperature 0` for deterministic inference

### For Better Accuracy
- Increase `--max_tokens` (up to 4096)
- Enable `--dynamic_self_correction`
- Use `--temperature 0.3` or lower

---

## Troubleshooting

### Model Loading Error
```
ValueError: Model path does not exist
```
**Solution:** Ensure the model path is correct and model files exist.

### Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Reduce `--max_tokens` (default: 2048 → try 1024)
2. Reduce `--end_id` (run in smaller batches)
3. Use CPU inference: Modify code to use `device='cpu'`

### Dataset Not Found
```
FileNotFoundError: datasets/math500/test.parquet
```
**Solutions:**
1. Check dataset path with `--dataset_filepath`
2. Convert JSONL to Parquet if needed:
   ```python
   import pandas as pd
   df = pd.read_json('datasets/math500/test.jsonl', lines=True)
   df.to_parquet('datasets/math500/test.parquet')
   ```

### Slow Inference
- Check GPU utilization: `nvidia-smi`
- Reduce `--max_tokens` for faster inference
- Ensure model is on GPU (check logs)

---

## Integration with Forest-of-Thought

These benchmarks complement the main Forest-of-Thought MCTS framework:

**Simple Benchmark (this guide):**
- Direct inference
- Answer extraction and validation
- Accuracy reporting

**Advanced Framework (run_with_mcf.py):**
- Multiple reasoning trees (MCTS)
- Tree search and refinement
- Consensus decision-making

You can use these benchmarks for:
1. **Baseline evaluation** - Direct model performance
2. **Quick testing** - Validate model setup
3. **Comparison** - Benchmark vs FoT improvements

---

## Example Complete Workflow

```bash
#!/bin/bash

# Set paths
MODEL_PATH="/home/models/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./benchmark_results_$(date +%Y%m%d_%H%M%S)"

# Run all standard benchmarks
python benchmark_runner.py \
    --model "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --preset math500-standard &

python benchmark_runner.py \
    --model "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --preset aime-full &

python benchmark_runner.py \
    --model "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --preset gsm8k-standard &

# Wait for all to complete
wait

echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
```

---

## Performance Expectations

Typical performance on Qwen2.5-7B-Instruct:

| Dataset | Accuracy | Avg Time | Samples |
|---------|----------|----------|---------|
| MATH500 | ~50-60% | ~2-3s | 500 |
| AIME | ~20-30% | ~3-5s | 30 |
| GSM8K | ~70-80% | ~1-2s | 1000 |

*Results vary based on prompting strategy and model variant*

---

## Support

For issues or questions:
1. Check logs in `./benchmark_results/*/`
2. Verify dataset files exist
3. Ensure model is properly loaded
4. Check `--max_tokens` and GPU memory

For more information about Forest-of-Thought:
- See `README.md`
- Check `run_with_mcf.py` for advanced MCTS usage
- Review `utils/examples.py` for few-shot prompting
