import json
import os
import ray
import argparse
from tqdm import tqdm
from datetime import datetime
from data_loader import load_data, get_evaluation_data
from api_utils import process_message_content, batch_api_call
from system_prompt import JUDGE_PROMPT, INFERENCE_PROMPT
from config import TEST_FILE

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crucial Step Recognition Evaluation Tool')
    
    # Model parameters
    parser.add_argument('--inference-model', type=str, default='gpt-3.5-turbo',
                      help='Model name for inference')
    parser.add_argument('--judge-model', type=str, default='gpt-4',
                      help='Model name for judging')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Generation temperature')
    
    # File paths
    parser.add_argument('--test-file', type=str, default=TEST_FILE,
                      help='Test dataset path')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                      help='Cache directory')
    
    # Other parameters
    parser.add_argument('--limit', type=int, default=0,
                      help='Limit number of samples to process, 0 means process all samples')
    parser.add_argument('--verbose', action='store_true',
                      help='Show detailed debugging information')
    
    return parser.parse_args()

def run_inference(args, test_data):
    """Execute batch inference"""
    print("\n==== Executing Crucial Step Recognition Inference ====")
    
    # Prepare batch processing data
    system_prompts = []
    message_contents = []
    debug_data = []
    
    if not ray.is_initialized():
        ray.init(cpus=20)
    print(f"Processing {len(test_data)} data samples...")
    for idx, item in enumerate(tqdm(test_data)):
        # Get evaluation data
        basic_prompt, image_inputs, inference_prompt = get_evaluation_data(item)
        
        # Process message content
        processed_content = process_message_content(basic_prompt['content'])
        
        # Add to batch processing list
        system_prompts.append(inference_prompt)
        message_contents.append(processed_content)
        
        # Save original input for debugging
        debug_item = {
            'input': {
                'system_prompt': inference_prompt,
                'basic_prompt': basic_prompt
            },
            'item_id': idx,
            'raw_item': item,
            'ground_truth': {
                'reason_crucial_step': item['annotation']['reason_crucial_step_recognition'] if 'annotation' in item and 'reason_crucial_step_recognition' in item['annotation'] else ""
            }
        }
        debug_data.append(debug_item)
    
    # Execute batch inference
    print("Executing batch inference...")
    results = batch_api_call(
        system_contents=system_prompts,
        message_contents=message_contents,
        model_name=args.inference_model,
        temperature=args.temperature,
        cache_dir=args.cache_dir
    )
    
    # Integrate results
    for i, result in enumerate(results):
        debug_data[i]['output'] = result
    
    return debug_data

def extract_score_and_reason(eval_result):
    """Extract score and reason, supporting multiple formats"""
    # Standardize text processing
    text = eval_result.strip()
    
    # Try multiple regex patterns to support different result extraction formats
    # 1. Try to match "score: [[number]]" and "reason: [[text]]" format
    score_match_1 = re.search(r"score:\s*\[\[(\d+)\]\]", text, re.IGNORECASE)
    reason_match_1 = re.search(r"reason:\s*\[\[(.*?)\]\]", text, re.IGNORECASE | re.DOTALL)
    
    # 2. Try to match "score: number" and "reason: text" format
    score_match_2 = re.search(r"score:\s*(\d+)", text, re.IGNORECASE)
    
    # If there's a compact format like "score: number, reason: text"
    score_reason_match = re.search(r"score:\s*(\d+),\s*reason:\s*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    
    # 3. Try to find a number directly from the beginning (might only give a score)
    score_match_3 = re.search(r"^(\d+)$", text.strip())
    
    # Extract score
    score = None
    if score_match_1:
        score = int(score_match_1.group(1))
    elif score_reason_match:
        score = int(score_reason_match.group(1))
    elif score_match_2:
        score = int(score_match_2.group(1))
    elif score_match_3:
        score = int(score_match_3.group(1))
    
    # Extract reason
    reason = None
    if reason_match_1:
        reason = reason_match_1.group(1).strip()
    elif score_reason_match:
        reason = score_reason_match.group(2).strip()
    else:
        # If there's no clear reason identifier, try to extract all content after the score as reason
        if score_match_2:
            reason_text = text[score_match_2.end():].strip()
            # Ignore possible leading colons or commas
            reason = re.sub(r"^[,:\s]+", "", reason_text).strip()
        else:
            # If even the score can't be found, use the whole text as reason
            reason = text
    
    # Validate if the extracted score is within valid range
    if score is not None and (score < 1 or score > 5):
        score = None
        reason = f"Extracted score {score} is invalid (needs to be between 1-5), original text: {text[:100]}..."
    
    return score, reason if reason else "Evaluation reason not found"

def run_evaluation(args, inference_results):
    """Evaluate inference results using Judge Model"""
    print("\n==== Executing Inference Result Evaluation ====")
    
    # Prepare batch processing data
    system_prompts = []
    message_contents = []
    skipped_count = 0
    
    # Use JUDGE_PROMPT as system prompt
    for idx, item in enumerate(tqdm(inference_results)):
        # Get inference result and ground truth
        model_inference = item['output']
        reference_answer = item['ground_truth']['reason_crucial_step']
        
        if not reference_answer:
            # If no ground truth, skip evaluation
            item['evaluation'] = {
                'score': None,
                'reason': "No reference answer provided, cannot perform evaluation"
            }
            skipped_count += 1
            continue
        
        # Create evaluation prompt
        user_prompt = f"""
Now evalute the following response and give your score and reason. Your score should be in the range of 1 to 5 and in the format of "score: [[score]], reason: [[reason]]".

Reference Answer:
{reference_answer}

Model Inference:
{model_inference}
"""
        
        # Process message content
        processed_content = [{
            'type': 'text',
            'text': user_prompt
        }]
        
        # Add to batch processing list
        system_prompts.append(JUDGE_PROMPT)
        message_contents.append(processed_content)
    
    if skipped_count > 0:
        print(f"Note: Skipped {skipped_count} samples without reference answers")
    
    # Execute batch evaluation
    print("Executing batch evaluation...")
    evaluation_results = batch_api_call(
        system_contents=system_prompts,
        message_contents=message_contents,
        model_name=args.judge_model,
        temperature=args.temperature,
        cache_dir=args.cache_dir
    )
    
    # Integrate evaluation results
    eval_idx = 0
    success_count = 0
    failed_count = 0
    
    for idx, item in enumerate(inference_results):
        if 'evaluation' in item and item['evaluation']['score'] is None:
            # Already marked as skipped samples
            continue
        
        # Parse evaluation result
        eval_result = evaluation_results[eval_idx]
        eval_idx += 1
        
        # Extract score and reason
        try:
            score, reason = extract_score_and_reason(eval_result)
            
            item['evaluation'] = {
                'score': score,
                'reason': reason,
                'raw_result': eval_result
            }
            
            if score is not None:
                success_count += 1
                
                if args.verbose:
                    print(f"Sample {idx}: Extracted score {score}, reason: {reason[:50]}...")
            else:
                failed_count += 1
                
                if args.verbose:
                    print(f"Sample {idx}: Score extraction failed, reason: {reason[:50]}...")
                    print(f"Original evaluation text: {eval_result[:100]}...")
                
        except Exception as e:
            item['evaluation'] = {
                'score': None,
                'reason': f"Evaluation result parsing failed: {str(e)}",
                'raw_result': eval_result
            }
            failed_count += 1
            
            if args.verbose:
                print(f"Sample {idx}: Parsing exception: {str(e)}")
    
    print(f"Evaluation parsing statistics: Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    
    return inference_results

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"crucial_step_{args.inference_model}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Output will be saved to: {output_path}")
    
    # Read test data
    data = load_data(args.test_file)
    
    # Limit sample count
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to processing {args.limit} samples")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Execute inference
    inference_results = run_inference(args, data)
    
    # Save original inference results
    inference_file = os.path.join(output_path, "inference_results.json")
    with open(inference_file, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
    
    # Evaluate inference results
    evaluation_results = run_evaluation(args, inference_results)
    
    # Save evaluation results
    evaluation_file = os.path.join(output_path, "evaluation_results.json")
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # Calculate statistics
    scores = [item['evaluation']['score'] for item in evaluation_results 
              if 'evaluation' in item and item['evaluation']['score'] is not None]
    
    if scores:
        avg_score = sum(scores) / len(scores)
        score_distribution = {i: scores.count(i) for i in range(1, 6)}
        
        # Save statistics
        stats = {
            'average_score': avg_score,
            'score_distribution': score_distribution,
            'total_evaluated': len(scores),
            'total_samples': len(evaluation_results),
            'config': {
                'inference_model': args.inference_model,
                'judge_model': args.judge_model,
                'temperature': args.temperature,
                'test_file': args.test_file
            }
        }
        
        stats_file = os.path.join(output_path, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        print("\n==== Evaluation Statistics ====")
        print(f"Average score: {avg_score:.2f}")
        print("Score distribution:")
        for score, count in sorted(score_distribution.items()):
            print(f"  Score {score}: {count} samples ({count/len(scores)*100:.2f}%)")
        print(f"Evaluated samples: {len(scores)} / {len(evaluation_results)}")
    else:
        print("No valid evaluation results found")
    
    print(f"\nComplete results saved to: {output_path}")

if __name__ == "__main__":
    # Import regex module (for parsing evaluation results)
    import re
    main()
