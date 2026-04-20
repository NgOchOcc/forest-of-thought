#!/usr/bin/env python3
"""
Advanced Benchmark Runner with preset configurations
Supports multiple datasets and model configurations
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List


class BenchmarkConfig:
    """Predefined benchmark configurations"""

    # Math500 configurations
    MATH500_QUICK = {
        'name': 'MATH500 Quick',
        'dataset': 'math500',
        'num_samples': 50,
        'description': 'Quick benchmark on 50 MATH500 samples'
    }

    MATH500_STANDARD = {
        'name': 'MATH500 Standard',
        'dataset': 'math500',
        'num_samples': 200,
        'description': 'Standard benchmark on 200 MATH500 samples'
    }

    MATH500_FULL = {
        'name': 'MATH500 Full',
        'dataset': 'math500',
        'num_samples': 500,
        'description': 'Full benchmark on all 500 MATH500 samples'
    }

    # AIME configurations
    AIME_FULL = {
        'name': 'AIME Full',
        'dataset': 'aime',
        'num_samples': 30,
        'description': 'Full AIME 2024 benchmark'
    }

    # GSM8K configurations
    GSM8K_QUICK = {
        'name': 'GSM8K Quick',
        'dataset': 'gsm8k',
        'num_samples': 100,
        'description': 'Quick GSM8K benchmark'
    }

    GSM8K_STANDARD = {
        'name': 'GSM8K Standard',
        'dataset': 'gsm8k',
        'num_samples': 500,
        'description': 'Standard GSM8K benchmark'
    }

    @classmethod
    def get_all(cls) -> Dict[str, Dict]:
        """Get all configurations"""
        return {
            'math500-quick': cls.MATH500_QUICK,
            'math500-standard': cls.MATH500_STANDARD,
            'math500-full': cls.MATH500_FULL,
            'aime-full': cls.AIME_FULL,
            'gsm8k-quick': cls.GSM8K_QUICK,
            'gsm8k-standard': cls.GSM8K_STANDARD,
        }


class AdvancedBenchmarkRunner:
    """Advanced benchmark runner with multiple configurations"""

    def __init__(self, model_path: str, output_base_dir: str = './benchmark_results'):
        self.model_path = model_path
        self.output_base_dir = output_base_dir
        self._validate_model()

    def _validate_model(self):
        """Validate model path exists"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        print(f"✓ Model found: {self.model_path}")

    def run_preset(self, preset: str):
        """Run a preset benchmark configuration"""
        configs = BenchmarkConfig.get_all()

        if preset not in configs:
            print(f"Unknown preset: {preset}")
            print(f"Available presets: {', '.join(configs.keys())}")
            return

        config = configs[preset]
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}\n")

        self.run_benchmark(
            dataset=config['dataset'],
            num_samples=config['num_samples']
        )

    def run_benchmark(self, dataset: str, num_samples: int):
        """Run benchmark with specified parameters"""
        cmd = [
            'python', 'benchmark_qwen.py',
            '--dataset', dataset,
            '--model_path', self.model_path,
            '--start_id', '0',
            '--end_id', str(num_samples),
            '--output_dir', self.output_base_dir
        ]

        print(f"Running command: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed with return code: {e.returncode}")
            return False

    def run_all_presets(self):
        """Run all available presets sequentially"""
        configs = BenchmarkConfig.get_all()
        results = {}

        print(f"{'='*80}")
        print(f"Running All Benchmarks")
        print(f"{'='*80}\n")

        for preset_name, config in configs.items():
            print(f"\n[{preset_name}] {config['name']}")
            print(f"Description: {config['description']}")

            success = self.run_benchmark(
                dataset=config['dataset'],
                num_samples=config['num_samples']
            )

            results[preset_name] = {
                'name': config['name'],
                'success': success
            }

        # Print summary
        print(f"\n{'='*80}")
        print("Benchmark Summary")
        print(f"{'='*80}")
        for preset_name, result in results.items():
            status = "✓ PASSED" if result['success'] else "✗ FAILED"
            print(f"{status} - {result['name']}")

    def list_presets(self):
        """List all available preset configurations"""
        configs = BenchmarkConfig.get_all()

        print("\nAvailable Benchmark Presets:")
        print("=" * 80)

        for key, config in configs.items():
            print(f"\n{key}:")
            print(f"  Name: {config['name']}")
            print(f"  Dataset: {config['dataset']}")
            print(f"  Samples: {config['num_samples']}")
            print(f"  Description: {config['description']}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Benchmark Runner for Qwen2.5-7B-Instruct',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run preset benchmark
  python benchmark_runner.py --model /path/to/model --preset math500-quick

  # Run all presets
  python benchmark_runner.py --model /path/to/model --all

  # List available presets
  python benchmark_runner.py --list

  # Custom benchmark
  python benchmark_runner.py --model /path/to/model --dataset aime --samples 30
        '''
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to Qwen2.5-7B-Instruct model'
    )
    parser.add_argument(
        '--preset',
        type=str,
        help='Run preset configuration (use --list to see all)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['math500', 'aime', 'gsm8k'],
        help='Custom dataset (use with --samples)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        help='Number of samples for custom benchmark'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preset benchmarks'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available presets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create runner
    runner = AdvancedBenchmarkRunner(
        model_path=args.model,
        output_base_dir=args.output_dir
    )

    # Handle different modes
    if args.list:
        runner.list_presets()
    elif args.all:
        runner.run_all_presets()
    elif args.preset:
        runner.run_preset(args.preset)
    elif args.dataset and args.samples:
        runner.run_benchmark(
            dataset=args.dataset,
            num_samples=args.samples
        )
    else:
        parser.print_help()
        print("\n" + "="*80)
        print("ERROR: Please specify either --preset, --all, --list, or --dataset + --samples")
        print("="*80)


if __name__ == '__main__':
    main()
