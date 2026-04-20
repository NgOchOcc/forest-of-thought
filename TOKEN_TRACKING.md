# Token Tracking Feature

This document explains the token tracking feature added to the `Pipeline` class in `models/load_local_model.py`.

## Overview

The token tracking feature automatically counts and logs:
- **Total tokens** used (input + output)
- **Input tokens** (prompt tokens)
- **Output tokens** (generated tokens)
- **Average tokens** per inference
- **Throughput** (tokens/second)

## Implementation Details

### 1. New Tracking Variables

Added to the `Pipeline.__init__()` method (lines 50-52):

```python
self.total_tokens = 0           # Total tokens used
self.total_input_tokens = 0     # Sum of input token counts
self.total_output_tokens = 0    # Sum of output token counts
```

### 2. Token Counting in Inference Methods

Token counting is now implemented in:
- `get_respond_llama()` - Lines 116, 129-135
- `get_respond_qwen()` - Lines 187, 200-206
- `get_respond_glm()` - Lines 152, 170-176
- `get_respond_deepseek()` - Lines 225, 233-239

**Example (from get_respond_llama):**
```python
input_token_count = inputs['input_ids'].shape[1]
# ... inference happens ...
num_tokens = generated_ids.shape[-1]
output_token_count = num_tokens - input_token_count

# Track tokens
self.total_input_tokens += input_token_count
self.total_output_tokens += output_token_count
self.total_tokens += num_tokens
```

### 3. New Methods for Reporting

#### `print_token_statistics()` (Lines 289-308)

Prints formatted token statistics to console:

```python
model.print_token_statistics()
```

Output example:
```
================================================================================
TOKEN USAGE STATISTICS
================================================================================
Total Inferences:        100
Total Tokens Used:       245,123
Total Input Tokens:      123,456
Total Output Tokens:     121,667
Avg Tokens per Inference: 2451.23
Avg Input Tokens:       1234.56
Avg Output Tokens:      1216.67
Avg Tokens/Second:      1234.56
================================================================================
```

#### `log_token_stats_json()` (Lines 310-327)

Returns token statistics as a dictionary for JSON logging:

```python
stats = model.log_token_stats_json()
print(json.dumps(stats, indent=2))
```

Output example:
```json
{
  "total_inferences": 100,
  "total_tokens": 245123,
  "total_input_tokens": 123456,
  "total_output_tokens": 121667,
  "avg_tokens_per_inference": 2451.23,
  "avg_input_tokens": 1234.56,
  "avg_output_tokens": 1216.67,
  "avg_tokens_per_second": 1234.56
}
```

## Usage Examples

### Example 1: Simple Token Counting

```python
from models.load_local_model import Pipeline

# Initialize model
model = Pipeline(
    model_id="/path/to/Qwen2.5-7B-Instruct",
    model_type='qwen'
)

# Run inference
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Solve: 2 + 2"}
]

response, confidence = model.get_respond(messages)

# Print token statistics
model.print_token_statistics()
```

### Example 2: Integration with Benchmark Script

The `benchmark_qwen.py` script automatically calls:

```python
# Print to console
self.model.print_token_statistics()

# Include in results JSON
token_stats = self.model.log_token_stats_json()
self.results['summary']['token_statistics'] = token_stats
```

### Example 3: Manual Token Tracking

```python
from models.load_local_model import Pipeline

model = Pipeline(model_id=model_path, model_type='qwen')

# Run multiple inferences
for problem in problems:
    response, _ = model.get_respond(messages)
    # Tokens are automatically tracked

# Get final statistics
stats = model.log_token_stats_json()
print(f"Total tokens used: {stats['total_tokens']}")
print(f"Average tokens per inference: {stats['avg_tokens_per_inference']:.2f}")
```

## Modified Files

### 1. `models/load_local_model.py`

**Changes:**
- Added token tracking variables to `__init__()` (lines 50-52)
- Updated `get_respond_llama()` to track tokens (lines 116, 129-135)
- Updated `get_respond_glm()` to track tokens (lines 152, 170-176)
- Updated `get_respond_qwen()` to track tokens (lines 187, 200-206)
- Updated `get_respond_deepseek()` to track tokens (lines 225, 233-239)
- Added `print_token_statistics()` method (lines 289-308)
- Added `log_token_stats_json()` method (lines 310-327)

### 2. `benchmark_qwen.py`

**Changes:**
- Updated `_compute_summary()` to include token statistics (lines 287-313)
- Added call to `model.print_token_statistics()` (line 273)
- Token stats included in JSON output with benchmark results

## Output Formats

### Console Output

Token statistics are printed with clear formatting:
```
================================================================================
TOKEN USAGE STATISTICS
================================================================================
Total Inferences:        100
Total Tokens Used:       245,123
Total Input Tokens:      123,456
Total Output Tokens:     121,667
Avg Tokens per Inference: 2451.23
Avg Input Tokens:       1234.56
Avg Output Tokens:      1216.67
Avg Tokens/Second:      1234.56
================================================================================
```

### JSON Output (in benchmark_results.json)

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
    "benchmark_datetime": "2024-05-20T14:30:22.123456",
    "token_statistics": {
      "total_inferences": 100,
      "total_tokens": 245123,
      "total_input_tokens": 123456,
      "total_output_tokens": 121667,
      "avg_tokens_per_inference": 2451.23,
      "avg_input_tokens": 1234.56,
      "avg_output_tokens": 1216.67,
      "avg_tokens_per_second": 1234.56
    }
  },
  "results": [ ... ]
}
```

## Model Support

Token tracking is implemented for all model types:
- ✅ LLaMA (`get_respond_llama`)
- ✅ Qwen (`get_respond_qwen`)
- ✅ GLM (`get_respond_glm`)
- ✅ Deepseek (`get_respond_deepseek`)
- ⚠️ Mistral/HuggingFace Pipeline (no token counting, returns 0)

## API Reference

### Pipeline.print_token_statistics()

Prints formatted token usage statistics to console.

**Signature:**
```python
def print_token_statistics(self) -> None
```

**Parameters:** None

**Returns:** None (prints to stdout)

**Example:**
```python
model.print_token_statistics()
```

### Pipeline.log_token_stats_json()

Returns token statistics as a dictionary suitable for JSON serialization.

**Signature:**
```python
def log_token_stats_json(self) -> Dict[str, Union[int, float]]
```

**Returns:**
```python
{
    'total_inferences': int,
    'total_tokens': int,
    'total_input_tokens': int,
    'total_output_tokens': int,
    'avg_tokens_per_inference': float,      # if infer_times > 0
    'avg_input_tokens': float,               # if infer_times > 0
    'avg_output_tokens': float,              # if infer_times > 0
    'avg_tokens_per_second': float,          # if tokens_per_second_sum > 0
}
```

**Example:**
```python
stats = model.log_token_stats_json()
print(json.dumps(stats, indent=2))
```

## Benchmark Integration

When running `benchmark_qwen.py`, token statistics are:

1. **Automatically tracked** during inference
2. **Printed to console** after benchmark completion
3. **Included in JSON results** for analysis

**Example command:**
```bash
python benchmark_qwen.py \
    --model_path /path/to/model \
    --dataset math500 \
    --end_id 100
```

**Output will include:**
```
================================================================================
TOKEN USAGE STATISTICS
================================================================================
Total Inferences:        100
Total Tokens Used:       245,123
...
```

And results will be saved to:
- `benchmark_results/MATH500_qwen2.5-7b-instruct_YYYYMMDD_HHMMSS/benchmark_results.json`
- `benchmark_results/MATH500_qwen2.5-7b-instruct_YYYYMMDD_HHMMSS/summary.json`

## Testing

Run the example script to test token tracking:

```bash
python example_token_tracking.py
```

This will:
1. Initialize the model
2. Run 3 test problems
3. Print token statistics
4. Display statistics as JSON

## Notes

- Token counting begins from first inference and accumulates
- Input tokens = prompt tokens (messages before generation)
- Output tokens = newly generated tokens
- Total tokens = input + output tokens
- All statistics are computed from the `total_*` counters
- Zero division is prevented with `if` checks

## Future Enhancements

Possible improvements:
1. Per-sample token tracking
2. Token cost calculation (based on OpenAI pricing)
3. Token distribution analysis
4. Export token statistics to CSV
5. Real-time token tracking visualization
