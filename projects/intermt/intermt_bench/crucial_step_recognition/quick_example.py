#!/usr/bin/env python3
"""
Quick Example: Crucial Step Recognition Evaluation

This script demonstrates how to run a simple evaluation of crucial step recognition
for a single model on a small subset of data.

Usage:
    python quick_example.py
"""

import os
import json
from recognition_inference import parse_args, run_inference, run_evaluation, main
from config import TEST_FILE, MODEL_NAME, TEMPERATURE

def quick_evaluation_demo():
    """
    Run a quick demonstration of the crucial step recognition evaluation.
    
    This function sets up and runs an evaluation with minimal configuration
    to help users understand how the system works.
    """
    
    print("=" * 60)
    print("Crucial Step Recognition Evaluation - Quick Demo")
    print("=" * 60)
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        print("Please ensure you have the test dataset in the correct location.")
        return
    
    print(f"✅ Test file found: {TEST_FILE}")
    
    # Create a simple configuration for demo
    class DemoArgs:
        def __init__(self):
            self.inference_model = MODEL_NAME
            self.judge_model = 'gpt-4o'  # Use a reliable judge model
            self.temperature = TEMPERATURE
            self.test_file = TEST_FILE
            self.output_dir = './demo_output'
            self.cache_dir = './demo_cache'
            self.limit = 5  # Only process 5 samples for demo
            self.verbose = True
    
    args = DemoArgs()
    
    print(f"\n📋 Demo Configuration:")
    print(f"  • Inference Model: {args.inference_model}")
    print(f"  • Judge Model: {args.judge_model}")
    print(f"  • Temperature: {args.temperature}")
    print(f"  • Sample Limit: {args.limit}")
    print(f"  • Output Directory: {args.output_dir}")
    
    # Check API configuration
    from config import API_KEY, API_BASE_URL
    if not API_KEY or not API_BASE_URL:
        print(f"\n⚠️  Warning: API credentials not configured in config.py")
        print(f"   Please set API_KEY and API_BASE_URL before running evaluation.")
        return
    
    print(f"\n✅ API configuration found")
    
    try:
        # Run the main evaluation function
        print(f"\n🚀 Starting quick evaluation demo...")
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        
        # Load a small subset of data
        from data_loader import load_data
        print(f"\n📂 Loading test data...")
        data = load_data(args.test_file)
        
        # Limit to specified number of samples
        if args.limit > 0:
            data = data[:args.limit]
            print(f"   Limited to {len(data)} samples for demo")
        
        # Run inference
        print(f"\n🔍 Running inference...")
        inference_results = run_inference(args, data)
        
        # Run evaluation
        print(f"\n⚖️  Running evaluation...")
        evaluation_results = run_evaluation(args, inference_results)
        
        # Calculate and display results
        scores = [item['evaluation']['score'] for item in evaluation_results 
                  if 'evaluation' in item and item['evaluation']['score'] is not None]
        
        if scores:
            avg_score = sum(scores) / len(scores)
            score_distribution = {i: scores.count(i) for i in range(1, 6)}
            
            print(f"\n📊 Demo Results:")
            print(f"  • Average Score: {avg_score:.2f}/5.0")
            print(f"  • Samples Evaluated: {len(scores)}/{len(evaluation_results)}")
            print(f"  • Score Distribution:")
            for score, count in sorted(score_distribution.items()):
                if count > 0:
                    print(f"    Score {score}: {count} samples")
        else:
            print(f"\n❌ No valid evaluation results obtained")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"   Full results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print(f"   Please check your configuration and try again.")
        return

def show_sample_data():
    """Show a sample of the test data structure"""
    try:
        from data_loader import load_data
        data = load_data(TEST_FILE)
        
        if data:
            sample = data[0]
            print(f"\n📋 Sample Data Structure:")
            print(f"  • ID: {sample.get('id', 'N/A')}")
            print(f"  • Conversations: {len(sample.get('conversations', []))} rounds")
            if 'annotation' in sample:
                print(f"  • Has Annotation: ✅")
            else:
                print(f"  • Has Annotation: ❌")
        
    except Exception as e:
        print(f"❌ Could not load sample data: {str(e)}")

if __name__ == "__main__":
    print("Crucial Step Recognition Evaluation System")
    print("Quick Example and Demo")
    print()
    
    # Show sample data structure
    show_sample_data()
    
    # Ask user if they want to run the demo
    print(f"\nThis demo will:")
    print(f"  1. Process 5 sample conversations")
    print(f"  2. Run crucial step recognition inference")
    print(f"  3. Evaluate the results using a judge model")
    print(f"  4. Display summary statistics")
    
    response = input(f"\nDo you want to run the demo? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        quick_evaluation_demo()
    else:
        print(f"\nDemo cancelled. To run the full evaluation, use:")
        print(f"  python recognition_inference.py --help")
        print(f"  bash run_recognition_evaluation.sh") 