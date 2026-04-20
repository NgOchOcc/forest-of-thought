#!/usr/bin/env python3
"""
Example script showing how to use the token tracking feature
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.load_local_model import Pipeline

def main():
    # Initialize model (replace with your actual model path)
    MODEL_PATH = "/path/to/Qwen2.5-7B-Instruct"  # Update this!

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path does not exist: {MODEL_PATH}")
        print("Please update MODEL_PATH in this script")
        return

    print("Initializing Qwen2.5-7B-Instruct model...")
    model = Pipeline(
        model_id=MODEL_PATH,
        model_type='qwen',
        dataname='math500',
        task='benchmark'
    )

    # Example problems for testing
    test_problems = [
        {
            'query': 'Convert the point (0,3) in rectangular coordinates to polar coordinates.',
            'expected': '(3, π/2)'
        },
        {
            'query': 'What is 2 + 2?',
            'expected': '4'
        },
        {
            'query': 'Solve for x: 2x + 5 = 15',
            'expected': 'x = 5'
        }
    ]

    print("\nRunning inference on test problems...\n")

    # Run inference on test problems
    for i, problem in enumerate(test_problems, 1):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful mathematical assistant. Please reason step by step."
            },
            {
                "role": "user",
                "content": f"Problem: {problem['query']}\n\nSolve this step by step and put your final answer in \\boxed{{}}."
            }
        ]

        print(f"[Problem {i}] {problem['query']}")
        print(f"Expected answer: {problem['expected']}")

        # Get response
        response, confidence = model.get_respond(messages, max_length=2048)
        print(f"Model response: {response[:200]}..." if len(response) > 200 else f"Model response: {response}")
        print()

    # Print token usage statistics
    model.print_token_statistics()

    # You can also get the stats as a dictionary
    stats = model.log_token_stats_json()
    print("\nToken statistics as dictionary:")
    import json
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
