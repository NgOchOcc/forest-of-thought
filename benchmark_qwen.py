#!/usr/bin/env python3
"""
Benchmark script for Forest-of-Thought with Qwen2.5-7B-Instruct
Supports math500 and aime datasets with configurable parameters
"""

import os
import json
import time
import argparse
import hashlib
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
import pandas as pd
from models.load_local_model import Pipeline
from utils.utils import mcts_load_data, get_query_gt_list, check, extract_label
from utils.examples import get_examples


class BenchmarkRunner:
    """Main benchmark runner for Qwen2.5-7B-Instruct"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Model: {args.model_path}")

        # Initialize model
        self.model = self._init_model()

        # Load dataset
        self.dataset = self._load_dataset()

        # Initialize results storage
        self.results = {
            'config': self._get_config(),
            'results': [],
            'summary': {}
        }

        # Create output directory
        self.output_dir = self._init_output_dir()

    def _init_model(self) -> Pipeline:
        """Initialize Qwen2.5-7B-Instruct model"""
        logger.info("Initializing model...")

        # Verify model path exists
        if not os.path.exists(self.args.model_path):
            raise ValueError(f"Model path does not exist: {self.args.model_path}")

        # Initialize pipeline
        pipeline = Pipeline(
            model_id=self.args.model_path,
            model_type='qwen',
            correction=self.args.dynamic_self_correction,
            correct_threshold=self.args.correct_threshold,
            dataname=self.args.dataset.split('-')[0],
            task='benchmark'
        )

        logger.info(f"Model loaded successfully from {self.args.model_path}")
        return pipeline

    def _load_dataset(self):
        """Load dataset from parquet/jsonl"""
        logger.info(f"Loading dataset from {self.args.dataset_filepath}...")

        # Convert JSONL to Parquet if needed
        filepath = self.args.dataset_filepath
        if not filepath.endswith('.parquet'):
            logger.info("Converting JSONL to Parquet...")
            df = pd.read_json(filepath, lines=True)
            filepath = filepath.replace('.jsonl', '.parquet')
            df.to_parquet(filepath)

        # Load using HF datasets
        st = time.time()
        dataset = load_dataset('parquet', data_files=filepath, split='train')
        elapsed = time.time() - st
        logger.info(f"Dataset loaded in {elapsed:.2f}s with {len(dataset)} samples")

        return dataset

    def _init_output_dir(self) -> Path:
        """Initialize output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = self.args.dataset.split('-')[0].upper()
        model_name = os.path.basename(self.args.model_path).lower()

        output_dir = Path(self.args.output_dir) / f"{dataset_name}_{model_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        return output_dir

    def _get_config(self) -> Dict[str, Any]:
        """Get benchmark configuration"""
        return {
            'dataset': self.args.dataset,
            'dataset_path': self.args.dataset_filepath,
            'model_path': self.args.model_path,
            'model_name': os.path.basename(self.args.model_path),
            'max_tokens': self.args.max_tokens,
            'num_samples': self.args.end_id - self.args.start_id,
            'temperature': self.args.temperature,
            'top_p': self.args.top_p,
            'dynamic_self_correction': self.args.dynamic_self_correction,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'torch_version': torch.__version__
        }

    def _get_system_prompt(self) -> str:
        """Get system prompt for Qwen"""
        return (
            "You are a helpful mathematical assistant. "
            "Please reason step by step, and put your final answer within \\boxed{}. "
            "Be precise and show your work clearly."
        )

    def _format_query(self, question: str) -> str:
        """Format query for the model"""
        if 'gsm8k' in self.args.dataset.lower():
            return f"Question: {question}\n\nSolve this step by step."
        else:  # math500, aime
            return f"Problem: {question}\n\nSolve this step by step and put your final answer in \\boxed{{}}."

    def _run_single_inference(self, query: str, ground_truth: str) -> Dict[str, Any]:
        """Run single inference on a problem"""
        start_time = time.time()

        # Format messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self._format_query(query)}
        ]

        try:
            # Get model response
            response, confidence = self.model.get_respond(
                messages,
                max_length=self.args.max_tokens
            )

            elapsed = time.time() - start_time

            # Extract answer
            extracted_answer = extract_label(self.args.dataset.split('-')[0], response)

            # Check correctness
            is_correct = check(ground_truth, extracted_answer, self.args.dataset.split('-')[0])

            return {
                'success': True,
                'response': response,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'confidence': confidence if isinstance(confidence, (int, float)) else 0.0,
                'inference_time': elapsed,
                'error': None
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error during inference: {str(e)}")

            return {
                'success': False,
                'response': None,
                'extracted_answer': None,
                'is_correct': False,
                'confidence': 0.0,
                'inference_time': elapsed,
                'error': str(e)
            }

    def run_benchmark(self):
        """Run complete benchmark"""
        logger.info("=" * 80)
        logger.info("Starting Benchmark")
        logger.info("=" * 80)

        # Select subset of dataset
        start_id = self.args.start_id
        end_id = min(self.args.end_id, len(self.dataset))
        num_samples = end_id - start_id

        logger.info(f"Running inference on samples {start_id} to {end_id} ({num_samples} total)")

        # Track metrics
        correct_count = 0
        total_inference_time = 0

        # Run inference
        for idx in range(start_id, end_id):
            sample = self.dataset[idx]

            # Extract query and ground truth
            try:
                if 'math' in self.args.dataset.lower():
                    query = sample.get('problem')
                    ground_truth = sample.get('answer')
                elif 'aime' in self.args.dataset.lower():
                    query = sample.get('Problem')
                    ground_truth = sample.get('Answer')
                elif 'gsm8k' in self.args.dataset.lower():
                    query = sample.get('question')
                    ground_truth = sample.get('answer')
                else:
                    query = sample.get('problem') or sample.get('Problem') or sample.get('question')
                    ground_truth = sample.get('answer') or sample.get('Answer')

                if not query or not ground_truth:
                    logger.warning(f"Sample {idx} missing query or ground truth")
                    continue

                # Run inference
                logger.info(f"[{idx+1}/{num_samples}] Processing sample {idx}...")
                result = self._run_single_inference(query, ground_truth)

                # Store result
                result_data = {
                    'sample_id': idx,
                    'query': query,
                    'ground_truth': ground_truth,
                    'inference_result': result
                }
                self.results['results'].append(result_data)

                # Update metrics
                if result['success']:
                    if result['is_correct']:
                        correct_count += 1
                        logger.info(f"✓ Correct (Time: {result['inference_time']:.2f}s)")
                    else:
                        logger.info(
                            f"✗ Incorrect - Got: {result['extracted_answer']}, "
                            f"Expected: {ground_truth} (Time: {result['inference_time']:.2f}s)"
                        )
                    total_inference_time += result['inference_time']
                else:
                    logger.warning(f"✗ Error: {result['error']}")

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue

        # Compute summary statistics
        self._compute_summary(correct_count, total_inference_time, num_samples)

        # Print token usage statistics
        self.model.print_token_statistics()

        # Save results
        self._save_results()

        logger.info("=" * 80)
        logger.info("Benchmark Complete")
        logger.info("=" * 80)

    def _compute_summary(self, correct_count: int, total_time: float, num_samples: int):
        """Compute summary statistics"""
        accuracy = (correct_count / num_samples * 100) if num_samples > 0 else 0
        avg_time = (total_time / num_samples) if num_samples > 0 else 0

        # Get token statistics from model
        token_stats = self.model.log_token_stats_json()

        self.results['summary'] = {
            'total_samples': num_samples,
            'correct_count': correct_count,
            'accuracy': f"{accuracy:.2f}%",
            'total_inference_time': f"{total_time:.2f}s",
            'average_inference_time': f"{avg_time:.2f}s",
            'samples_per_second': f"{num_samples/total_time:.2f}" if total_time > 0 else "N/A",
            'benchmark_datetime': datetime.now().isoformat(),
            'token_statistics': token_stats
        }

        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        for key, value in self.results['summary'].items():
            if key != 'token_statistics':
                logger.info(f"{key}: {value}")
        logger.info("\nTOKEN STATISTICS:")
        for key, value in token_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 80 + "\n")

    def _save_results(self):
        """Save results to JSON file"""
        output_file = self.output_dir / "benchmark_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

        # Also save summary to separate file
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'config': self.results['config'],
                'summary': self.results['summary']
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen2.5-7B-Instruct on math reasoning datasets"
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['math500', 'aime', 'gsm8k'],
        default='math500',
        help='Dataset to benchmark on'
    )
    parser.add_argument(
        '--dataset_filepath',
        type=str,
        default=None,
        help='Path to dataset file (default: auto-detect based on dataset name)'
    )

    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to Qwen2.5-7B-Instruct model'
    )

    # Inference arguments
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='Maximum tokens for generation'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for sampling'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p for nucleus sampling'
    )

    # Benchmark arguments
    parser.add_argument(
        '--start_id',
        type=int,
        default=0,
        help='Start index for dataset'
    )
    parser.add_argument(
        '--end_id',
        type=int,
        default=500,
        help='End index for dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results'
    )

    # Self-correction arguments
    parser.add_argument(
        '--dynamic_self_correction',
        action='store_true',
        help='Enable dynamic self-correction'
    )
    parser.add_argument(
        '--correct_threshold',
        type=float,
        default=-0.5,
        help='Confidence threshold for self-correction'
    )

    args = parser.parse_args()

    # Set default dataset path if not provided
    if args.dataset_filepath is None:
        dataset_paths = {
            'math500': 'datasets/math500/test.parquet',
            'aime': 'datasets/aime2024/aime_2024_problems.parquet',
            'gsm8k': 'datasets/gsm8k/test.parquet'
        }
        args.dataset_filepath = dataset_paths.get(args.dataset)

    # Verify model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return

    # Run benchmark
    runner = BenchmarkRunner(args)
    runner.run_benchmark()


if __name__ == '__main__':
    main()
