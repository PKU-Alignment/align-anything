import json
import os
import ray
import re
import numpy as np
import argparse
from scipy.stats import pearsonr
from datetime import datetime
from tqdm import tqdm
from data_loader import get_global_evaluation_data, get_single_turn_evaluation_data
from api_utils import process_message_content, api_call, batch_api_call
from utils.config import (
    TEST_FILE, MODEL_NAME, TEMPERATURE, 
    GLOBAL_EVALUATION_CATEGORIES, LOCAL_EVALUATION_CATEGORIES,
    INCLUDE_REASON, OUTPUT_DIR, CACHE_DIR
)
import random

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-turn dialogue evaluation tool')
    
    # Model parameters
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                        help='Model name to use')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help='Generation temperature')
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, choices=['global', 'local', 'both'], default='global',
                        help='Evaluation mode: global(overall evaluation), local(per-turn evaluation), both(both modes)')
    
    # Default categories based on mode
    default_categories = GLOBAL_EVALUATION_CATEGORIES
    
    # Evaluation parameters
    parser.add_argument('--categories', type=str, default=','.join(default_categories),
                        help='Evaluation categories, separated by comma, or use "all" for all categories')
    parser.add_argument('--reason', action='store_true', default=INCLUDE_REASON,
                        help='Include reasoning')
    parser.add_argument('--no-reason', action='store_false', dest='reason',
                        help='Do not include reasoning')
    
    # File paths
    parser.add_argument('--test-file', type=str, default=TEST_FILE,
                        help='Test dataset path')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR,
                        help='Cache directory')
    
    # Other parameters
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit the number of samples to process, 0 means process all samples')
    
    args = parser.parse_args()
    
    # Process evaluation categories
    if args.categories == 'all':
        args.categories = ['all']
    else:
        args.categories = args.categories.split(',')
    
    return args

# Regular expressions to extract model output scores and reasons
def extract_scores_and_reasons(output, categories, include_reason=True):
    scores = []
    reasons = []
    
    # Clean output text, remove possible interfering characters
    output = output.strip()
    
    # Handle possible "Evaluation list:" prefix
    if "Evaluation list:" in output:
        output = output.split("Evaluation list:", 1)[1].strip()
    
    # General boxed pattern, adapting to various boxed formats
    def make_boxed_pattern(base_pattern):
        # Create a pattern that adapts to various boxed formats
        boxed_variations = [
            r"\\\\boxed\{(\d+)\}",      # \\boxed{N}
            r"\\boxed\{(\d+)\}",         # \boxed{N}
            r"\\\\boxed\{\{(\d+)\}\}",   # \\boxed{{N}}
            r"\\boxed\{\{(\d+)\}\}"      # \boxed{{N}}
        ]
        patterns = []
        for boxed in boxed_variations:
            patterns.append(base_pattern.replace("BOXED_PLACEHOLDER", boxed))
        return patterns
    
    # Try overall format detection first to determine the most likely format type
    format_types = {
        "comma_separated_brackets": r"\[\s*[\w_]+\s*,[^[]*?BOXED_PLACEHOLDER\s*\],",  # [category, text, \boxed{N}],
        "newline_separated_brackets": r"\[\s*[\w_]+\s*,[^[]*?BOXED_PLACEHOLDER\s*\]\s*\n",  # [category, text, \boxed{N}]\n
        "double_brackets": r"\[\[\s*[\w_]+\s*,[^[]*?BOXED_PLACEHOLDER\s*\]\]",  # [[category, text, \boxed{N}]]
        "no_comma_brackets": r"\[\[\s*[\w_]+\s+[^,\[]*?BOXED_PLACEHOLDER\s*\]\]",  # [[category text \boxed{N}]]
    }
    
    # Detect which format matches the most
    format_counts = {}
    for fmt_name, base_pattern in format_types.items():
        patterns = make_boxed_pattern(base_pattern)
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
            count += len(matches)
        format_counts[fmt_name] = count
    
    # Determine the best format (format with most matches)
    best_format = max(format_counts.items(), key=lambda x: x[1])[0] if format_counts else None
    
    # Define category extraction logic for different formats
    def extract_by_format(output_text, category, format_type):
        score_found = None
        reason_found = ""
        
        # Basic patterns for different formats
        if format_type == "comma_separated_brackets":
            base_pattern = r"\[\s*" + re.escape(category) + r"\s*,\s*(.*?),\s*BOXED_PLACEHOLDER\s*\],"
        elif format_type == "newline_separated_brackets":
            base_pattern = r"\[\s*" + re.escape(category) + r"\s*,\s*(.*?),\s*BOXED_PLACEHOLDER\s*\]\s*[\n,]"
        elif format_type == "double_brackets":
            base_pattern = r"\[\[\s*" + re.escape(category) + r"\s*,\s*(.*?),\s*BOXED_PLACEHOLDER\s*\]\]"
        elif format_type == "no_comma_brackets":
            base_pattern = r"\[\[\s*" + re.escape(category) + r"\s+(.*?)BOXED_PLACEHOLDER\s*\]\]"
        else:
            return score_found, reason_found
        
        # Try all boxed variants
        patterns = make_boxed_pattern(base_pattern)
        for pattern in patterns:
            match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
            if match:
                if format_type == "no_comma_brackets":
                    reason_found = match.group(1).strip()
                    score_found = int(match.group(2))
                else:
                    reason_found = match.group(1).strip()
                    score_found = int(match.group(2))
                return score_found, reason_found
        
        return score_found, reason_found
    
    # First try to extract using the best format
    if best_format:
        all_extracted = True
        temp_scores = []
        temp_reasons = []
        
        for category in categories:
            score, reason = extract_by_format(output, category, best_format)
            if score is None:
                all_extracted = False
            temp_scores.append(score)
            temp_reasons.append(reason)
        
        # If best format successfully extracted all categories, use this result
        if all_extracted:
            return temp_scores, temp_reasons
    
    # If best format cannot extract all categories, try the most suitable format for each category
    mixed_scores = [None] * len(categories)
    mixed_reasons = [""] * len(categories)
    
    for i, category in enumerate(categories):
        # Try all formats until finding one that can extract
        for fmt in format_types.keys():
            score, reason = extract_by_format(output, category, fmt)
            if score is not None:
                mixed_scores[i] = score
                mixed_reasons[i] = reason
                break
    
    # If mixed format extraction result is better than single format, use mixed result
    if sum(1 for s in mixed_scores if s is not None) > sum(1 for s in temp_scores if s is not None):
        temp_scores, temp_reasons = mixed_scores, mixed_reasons
    
    # If above methods still cannot extract all scores, use more general regex patterns
    if None in temp_scores:
        for i, category in enumerate(categories):
            if temp_scores[i] is None:
                # Try more flexible patterns for this category
                base_patterns = [
                    # Various bracket and boxed combinations
                    r"\[[\[\s]*" + re.escape(category) + r"[^]]*?BOXED_PLACEHOLDER[^]]*?\][\]\s]*",
                    # Directly find any content containing category and boxed
                    r"" + re.escape(category) + r"[^\\]*?BOXED_PLACEHOLDER",
                ]
                
                for base_pattern in base_patterns:
                    patterns = make_boxed_pattern(base_pattern)
                    for pattern in patterns:
                        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                        if match:
                            temp_scores[i] = int(match.group(1))
                            
                            # Extract reason
                            if include_reason:
                                # Try various boxed patterns to extract reason
                                for boxed_var in [r"\\\\boxed", r"\\boxed"]:
                                    # Get text between category name and boxed as reason
                                    reason_pattern = re.escape(category) + r"(.*?)" + boxed_var + r"\{" + str(temp_scores[i]) + r"\}"
                                    reason_match = re.search(reason_pattern, output, re.IGNORECASE | re.DOTALL)
                                    
                                    if reason_match:
                                        raw_reason = reason_match.group(1).strip()
                                        
                                        # Clean reason
                                        if raw_reason.startswith(','):
                                            raw_reason = raw_reason[1:].strip()
                                        temp_reasons[i] = raw_reason.strip()
                                        break
                            break
                    
                    if temp_scores[i] is not None:
                        break
    
    # Last two fallback strategies:
    
    # 1. Directly extract all boxed tags and try to match to correct categories
    if None in temp_scores:
        # First collect all boxed tags
        all_boxed_values = []
        boxed_patterns = [
            r"\\\\boxed\{(\d+)\}",
            r"\\boxed\{(\d+)\}",
            r"\\\\boxed\{\{(\d+)\}\}",
            r"\\boxed\{\{(\d+)\}\}"
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
            all_boxed_values.extend([int(m) for m in matches])
        
        # If found boxed values match category count and all in reasonable range, assign in order
        if len(all_boxed_values) == len(categories) and all(1 <= v <= 5 for v in all_boxed_values):
            for i, value in enumerate(all_boxed_values):
                if temp_scores[i] is None:
                    temp_scores[i] = value
        
        # If found boxed values are all the same value (e.g., all 3), assign this value to all unextracted categories
        elif len(set(all_boxed_values)) == 1 and all_boxed_values:
            common_value = all_boxed_values[0]
            for i in range(len(temp_scores)):
                if temp_scores[i] is None:
                    temp_scores[i] = common_value
    
    # 2. Try to search for category and score combinations in the text
    if None in temp_scores:
        for i, category in enumerate(categories):
            if temp_scores[i] is None:
                # Try to find category and number combinations in the text
                for score_value in range(1, 6):  # Assume score range is 1 to 5
                    # Search for patterns where category is followed by a number
                    category_score_pattern = r"" + re.escape(category) + r".*?(\D|^)" + str(score_value) + r"(\D|$)"
                    if re.search(category_score_pattern, output, re.IGNORECASE | re.DOTALL):
                        temp_scores[i] = score_value
                        break
    
    return temp_scores, temp_reasons

# Print extracted scores and matching results
def print_score_comparison(item_id, categories, model_scores, gt_scores, matches, reasons=None, round_num=None):
    round_info = f"Round {round_num} " if round_num is not None else ""
    print(f"\nSample {item_id} {round_info}Score Comparison:")
    print(f"{'Category':<25} {'Model Score':<10} {'Ground Truth':<10} {'Match':<8} {'Reason':<30}")
    print("-" * 80)
    for i, category in enumerate(categories):
        model_score = model_scores[i] if model_scores[i] is not None else "Not Extracted"
        match_str = "✓" if matches[i] else "✗" if matches[i] is not None else "No Match"
        reason = (reasons[i][:27] + "..." if len(reasons[i]) > 30 else reasons[i]) if reasons and reasons[i] else ""
        print(f"{category:<25} {model_score:<10} {gt_scores[i]:<10} {match_str:<8} {reason:<30}")
    print("-" * 80)

# Calculate Pearson correlation coefficient
def calculate_pearson(model_scores, gt_scores):
    # Filter out None values
    valid_pairs = [(m, g) for m, g in zip(model_scores, gt_scores) if m is not None]
    if len(valid_pairs) < 2:  # Pearson correlation requires at least 2 points
        return None
    
    m_scores, g_scores = zip(*valid_pairs)
    
    # Check for constant arrays
    if len(set(m_scores)) <= 1 or len(set(g_scores)) <= 1:
        return {'coefficient': "Constant Array", 'p_value': None, 'note': "Input arrays contain constant values, cannot calculate correlation coefficient"}
    
    try:
        coef, p_value = pearsonr(m_scores, g_scores)
        return {'coefficient': coef, 'p_value': p_value}
    except Exception as e:
        return {'coefficient': None, 'p_value': None, 'error': str(e)}

def run_global_evaluation(args, test_data, output_path):
    """Run global evaluation logic"""
    print("\n==== Executing Global Evaluation ====")
    
    # Parse evaluation categories
    evaluation_category = args.categories
    if 'all' in evaluation_category:
        evaluation_category = GLOBAL_EVALUATION_CATEGORIES
    else:
        # Filter categories, ensure only global evaluation categories are used
        evaluation_category = [cat for cat in evaluation_category if cat in GLOBAL_EVALUATION_CATEGORIES]
        if not evaluation_category:  # If empty after filtering, use all global evaluation categories
            evaluation_category = GLOBAL_EVALUATION_CATEGORIES
            print("Warning: Specified categories are not applicable to global evaluation, using default global evaluation categories")
    
    # Prepare batch processing data
    system_prompts = []
    message_contents = []
    debug_data = []
    
    print(f"Processing {len(test_data)} data items...")
    for idx, item in enumerate(tqdm(test_data)):
        # Get evaluation data
        basic_prompt, image_inputs, global_judge_prompt = get_global_evaluation_data(
            item, if_reason=args.reason, evaluation_category=evaluation_category
        )
        ground_truth_list = item['conversations'][0]['annotations']
        evaluation_gt = [ground_truth_list[category] for category in evaluation_category]
        
        # Process message content
        processed_content = process_message_content(basic_prompt['content'])
        
        # Add to batch processing list
        system_prompts.append(global_judge_prompt)
        message_contents.append(processed_content)
        
        # Save original input for debugging
        debug_item = {
            'input': {
                'system_prompt': global_judge_prompt,
                'basic_prompt': basic_prompt
            },
            'item_id': idx,
            'raw_item': item,
            'ground_truth': {
                'categories': evaluation_category,
                'scores': evaluation_gt
            }
        }
        debug_data.append(debug_item)
    
    # Execute batch inference
    print("Executing batch inference...")
    results = batch_api_call(
        system_contents=system_prompts,
        message_contents=message_contents,
        model_name=args.model,
        temperature=args.temperature,
        cache_dir=args.cache_dir
    )
    
    # Compare model output with ground truth
    category_matches = {category: {'total': 0, 'matches': 0} for category in evaluation_category}
    category_pearson_data = {category: {'model': [], 'ground_truth': []} for category in evaluation_category}
    total_items = 0
    
    # Integrate results
    for i, result in enumerate(results):
        debug_data[i]['output'] = result
        
        # Extract model scores and reasons
        model_scores, reasons = extract_scores_and_reasons(result, evaluation_category, args.reason)
        debug_data[i]['extracted_scores'] = model_scores
        debug_data[i]['extracted_reasons'] = reasons
        
        # Compare scores
        gt_scores = debug_data[i]['ground_truth']['scores']
        matches = []
        
        for j, (model_score, gt_score) in enumerate(zip(model_scores, gt_scores)):
            if model_score is not None:
                category = evaluation_category[j]
                category_matches[category]['total'] += 1
                # Add to Pearson correlation calculation data
                category_pearson_data[category]['model'].append(model_score)
                category_pearson_data[category]['ground_truth'].append(gt_score)
                
                if int(model_score) == int(gt_score):
                    category_matches[category]['matches'] += 1
                    matches.append(True)
                else:
                    matches.append(False)
            else:
                matches.append(None)
        
        debug_data[i]['score_matches'] = matches
        
        # Print detailed scoring for first 5 samples
        if i < 5:
            print_score_comparison(
                debug_data[i]['item_id'], 
                evaluation_category, 
                model_scores, 
                gt_scores, 
                matches,
                reasons
            )
        
        total_items += 1
    
    # Calculate consistency ratio and Pearson correlation coefficient for each dimension
    accuracy_results = {}
    for category, stats in category_matches.items():
        if stats['total'] > 0:
            accuracy = stats['matches'] / stats['total']
            pearson_data = category_pearson_data[category]
            
            # Calculate Pearson correlation coefficient
            pearson_result = None
            if len(pearson_data['model']) >= 2:
                pearson_result = calculate_pearson(
                    pearson_data['model'], 
                    pearson_data['ground_truth']
                )
            
            accuracy_results[category] = {
                'accuracy': accuracy,
                'matches': stats['matches'],
                'total': stats['total'],
                'pearson': pearson_result
            }
        else:
            accuracy_results[category] = {
                'accuracy': 0,
                'matches': 0,
                'total': 0,
                'pearson': None
            }
    
    # Save complete debug data
    global_dir = os.path.join(output_path, "global")
    os.makedirs(global_dir, exist_ok=True)
    
    debug_file = os.path.join(global_dir, "debug.json")
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    
    # Save clean output results
    clean_results = []
    for i, result in enumerate(results):
        clean_item = {
            'item_id': debug_data[i]['item_id'],
            'output': result,
            'extracted_scores': debug_data[i]['extracted_scores'],
            'extracted_reasons': debug_data[i]['extracted_reasons'],
            'ground_truth': debug_data[i]['ground_truth']['scores'],
            'score_matches': debug_data[i]['score_matches']
        }
        clean_results.append(clean_item)
    
    clean_file = os.path.join(global_dir, "clean.json")
    with open(clean_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Save consistency ratio and Pearson coefficient results
    accuracy_file = os.path.join(global_dir, "accuracy.json")
    with open(accuracy_file, 'w', encoding='utf-8') as f:
        json.dump(accuracy_results, f, ensure_ascii=False, indent=2)
    
    # Save evaluation configuration information
    config_info = {
        'model_name': args.model,
        'temperature': args.temperature,
        'evaluation_categories': evaluation_category,
        'include_reason': args.reason,
        'test_file': args.test_file,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'limit': args.limit
    }
    config_file = os.path.join(global_dir, "config_info.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)
    
    # Calculate average Pearson coefficient for all categories
    valid_pearson_values = []
    for category, stats in accuracy_results.items():
        if (stats['pearson'] and 
            not isinstance(stats['pearson']['coefficient'], str) and 
            stats['pearson']['coefficient'] is not None and 
            not np.isnan(stats['pearson']['coefficient'])):
            valid_pearson_values.append(stats['pearson']['coefficient'])
    
    average_pearson = np.mean(valid_pearson_values) if valid_pearson_values else None
    
    # Save overall model performance summary
    overall_accuracy = sum([stats['matches'] for category, stats in category_matches.items()]) / sum([stats['total'] for category, stats in category_matches.items()]) if sum([stats['total'] for category, stats in category_matches.items()]) > 0 else 0
    
    summary = {
        'overall_accuracy': overall_accuracy,
        'average_pearson': average_pearson,
        'category_results': accuracy_results
    }
    
    summary_file = os.path.join(global_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nGlobal evaluation completed! Results saved to directory: {global_dir}")
    print("\nConsistency ratio and Pearson correlation coefficient for each dimension:")
    for category, stats in accuracy_results.items():
        pearson_info = ""
        if stats['pearson']:
            if isinstance(stats['pearson']['coefficient'], str):
                # Handle constant array case
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']}"
                if 'note' in stats['pearson']:
                    pearson_info += f" ({stats['pearson']['note']})"
            elif stats['pearson']['coefficient'] is not None:
                # Normal case
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']:.4f}"
                if stats['pearson']['p_value'] is not None:
                    pearson_info += f" (p={stats['pearson']['p_value']:.4f})"
            else:
                # Error case
                pearson_info = ", Pearson coefficient: Calculation failed"
                if 'error' in stats['pearson']:
                    pearson_info += f" ({stats['pearson']['error']})"
        
        print(f"{category}: {stats['accuracy']*100:.2f}% ({stats['matches']}/{stats['total']}){pearson_info}")
    
    print(f"\nOverall accuracy: {overall_accuracy*100:.2f}%")
    if average_pearson is not None:
        print(f"Average Pearson coefficient: {average_pearson:.4f}")
    
    return overall_accuracy, average_pearson, accuracy_results

def run_local_evaluation(args, test_data, output_path):
    """Run local evaluation logic (evaluate each dialogue turn)"""
    print("\n==== Executing Local Evaluation (Per-Turn Evaluation) ====")
    
    # Parse evaluation categories
    local_eval_category = args.categories
    if 'all' in local_eval_category:
        local_eval_category = LOCAL_EVALUATION_CATEGORIES
        print('Use all local evaluation categories')
    else:
        # Filter categories, ensure only local evaluation categories are used
        local_eval_category = [cat for cat in local_eval_category if cat in LOCAL_EVALUATION_CATEGORIES]
        if not local_eval_category:  # If empty after filtering, use all local evaluation categories
            local_eval_category = LOCAL_EVALUATION_CATEGORIES
            print("Warning: The specified categories are not applicable to local evaluation, using default local evaluation categories")
    
    # Prepare batch processing data
    system_prompts = []
    message_contents = []
    debug_data = []
    
    # Track mapping between global ID and turn information
    id_round_mapping = []
    
    print(f"Processing each turn of {len(test_data)} data items...")
    global_idx = 0
    
    for item_idx, item in enumerate(tqdm(test_data)):
        # Get evaluation data
        basic_prompt, image_inputs, local_judge_prompt, total_round_number = get_single_turn_evaluation_data(
            item, if_reason=args.reason, evaluation_category=local_eval_category
        )
        
        for round_num in range(1, total_round_number + 1):
            # Add specific evaluation instruction for each turn
            round_specific_prompt = f"{local_judge_prompt}\n\nNow Please evaluate round {round_num}"
            
            # Process message content
            processed_content = process_message_content(basic_prompt['content'])
            
            # Add to batch processing list
            system_prompts.append(round_specific_prompt)
            message_contents.append(processed_content)
            
            # Get ground truth for this turn
            ground_truth_list = item['conversations'][round_num]['annotations']
         
            evaluation_gt = []
            for category in local_eval_category:
                if category in ground_truth_list:
                    evaluation_gt.append(ground_truth_list[category])
                else:
                    # If no annotation for this category, mark as -1 for missing
                    evaluation_gt.append(-1)
            # Save original input for debugging
            debug_item = {
                'input': {
                    'system_prompt': round_specific_prompt,
                    'basic_prompt': basic_prompt
                },
                'item_id': item_idx,
                'round_num': round_num,
                'raw_item': item,
                'ground_truth': {
                    'categories': local_eval_category,
                    'scores': evaluation_gt
                }
            }
            debug_data.append(debug_item)
            
            # Record mapping relationship
            id_round_mapping.append({
                'global_idx': global_idx,
                'item_idx': item_idx,
                'round_num': round_num
            })
            global_idx += 1
    
    if not system_prompts:
        print("No evaluable turn data found")
        return 0, None, {}
    
    # Execute batch inference
    print(f"Executing batch inference, total {len(system_prompts)} evaluation requests...")
    results = batch_api_call(
        system_contents=system_prompts,
        message_contents=message_contents,
        model_name=args.model,
        temperature=args.temperature,
        cache_dir=args.cache_dir
    )
    
    # Compare model output with ground truth
    category_matches = {category: {'total': 0, 'matches': 0} for category in local_eval_category}
    category_pearson_data = {category: {'model': [], 'ground_truth': []} for category in local_eval_category}
    
    # Integrate results
    for i, result in enumerate(results):
        debug_data[i]['output'] = result
        item_idx = id_round_mapping[i]['item_idx']
        round_num = id_round_mapping[i]['round_num']
        
        # Extract model scores and reasons
        model_scores, reasons = extract_scores_and_reasons(result, local_eval_category, args.reason)
        debug_data[i]['extracted_scores'] = model_scores
        debug_data[i]['extracted_reasons'] = reasons
        
        # Compare scores
        gt_scores = debug_data[i]['ground_truth']['scores']
        matches = []
        
        for j, (model_score, gt_score) in enumerate(zip(model_scores, gt_scores)):
            # Skip evaluations without ground truth
            if gt_score == -1:
                matches.append(None)
                continue
                
            if model_score is not None:
                category = local_eval_category[j]
                category_matches[category]['total'] += 1
                # Add to Pearson correlation calculation data
                category_pearson_data[category]['model'].append(model_score)
                category_pearson_data[category]['ground_truth'].append(gt_score)
                
                if int(model_score) == int(gt_score):
                    category_matches[category]['matches'] += 1
                    matches.append(True)
                else:
                    matches.append(False)
            else:
                matches.append(None)
        
        debug_data[i]['score_matches'] = matches
        
        # Print detailed scoring for first 5 samples
        if i < 5:
            print_score_comparison(
                item_idx, 
                local_eval_category, 
                model_scores, 
                gt_scores, 
                matches,
                reasons,
                round_num
            )
    
    # Calculate consistency ratio and Pearson correlation coefficient for each dimension
    accuracy_results = {}
    for category, stats in category_matches.items():
        if stats['total'] > 0:
            accuracy = stats['matches'] / stats['total']
            pearson_data = category_pearson_data[category]
            
            # Calculate Pearson correlation coefficient
            pearson_result = None
            if len(pearson_data['model']) >= 2:
                pearson_result = calculate_pearson(
                    pearson_data['model'], 
                    pearson_data['ground_truth']
                )
            
            accuracy_results[category] = {
                'accuracy': accuracy,
                'matches': stats['matches'],
                'total': stats['total'],
                'pearson': pearson_result
            }
        else:
            accuracy_results[category] = {
                'accuracy': 0,
                'matches': 0,
                'total': 0,
                'pearson': None
            }
    
    # Save complete debug data
    local_dir = os.path.join(output_path, "local")
    os.makedirs(local_dir, exist_ok=True)
    
    debug_file = os.path.join(local_dir, "debug.json")
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    
    # Save clean output results
    clean_results = []
    for i, result in enumerate(results):
        item_idx = id_round_mapping[i]['item_idx']
        round_num = id_round_mapping[i]['round_num']
        
        clean_item = {
            'item_id': item_idx,
            'round_num': round_num,
            'output': result,
            'extracted_scores': debug_data[i]['extracted_scores'],
            'extracted_reasons': debug_data[i]['extracted_reasons'],
            'ground_truth': debug_data[i]['ground_truth']['scores'],
            'score_matches': debug_data[i]['score_matches']
        }
        clean_results.append(clean_item)
    
    clean_file = os.path.join(local_dir, "clean.json")
    with open(clean_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Save consistency ratio and Pearson coefficient results
    accuracy_file = os.path.join(local_dir, "accuracy.json")
    with open(accuracy_file, 'w', encoding='utf-8') as f:
        json.dump(accuracy_results, f, ensure_ascii=False, indent=2)
    
    # Save evaluation configuration information
    config_info = {
        'model_name': args.model,
        'temperature': args.temperature,
        'evaluation_categories': local_eval_category,
        'include_reason': args.reason,
        'test_file': args.test_file,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'limit': args.limit
    }
    config_file = os.path.join(local_dir, "config_info.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)
    
    # Calculate average Pearson coefficient for all categories
    valid_pearson_values = []
    for category, stats in accuracy_results.items():
        if (stats['pearson'] and 
            not isinstance(stats['pearson']['coefficient'], str) and 
            stats['pearson']['coefficient'] is not None and 
            not np.isnan(stats['pearson']['coefficient'])):
            valid_pearson_values.append(stats['pearson']['coefficient'])
    
    average_pearson = np.mean(valid_pearson_values) if valid_pearson_values else None
    
    # Save overall model performance summary
    overall_accuracy = sum([stats['matches'] for category, stats in category_matches.items()]) / sum([stats['total'] for category, stats in category_matches.items()]) if sum([stats['total'] for category, stats in category_matches.items()]) > 0 else 0
    
    summary = {
        'overall_accuracy': overall_accuracy,
        'average_pearson': average_pearson,
        'category_results': accuracy_results
    }
    
    summary_file = os.path.join(local_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nLocal evaluation completed! Results saved to directory: {local_dir}")
    print("\nConsistency ratio and Pearson correlation coefficient for each dimension:")
    for category, stats in accuracy_results.items():
        pearson_info = ""
        if stats['pearson']:
            if isinstance(stats['pearson']['coefficient'], str):
                # Handle constant array case
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']}"
                if 'note' in stats['pearson']:
                    pearson_info += f" ({stats['pearson']['note']})"
            elif stats['pearson']['coefficient'] is not None:
                # Normal case
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']:.4f}"
                if stats['pearson']['p_value'] is not None:
                    pearson_info += f" (p={stats['pearson']['p_value']:.4f})"
            else:
                # Error case
                pearson_info = ", Pearson coefficient: Calculation failed"
                if 'error' in stats['pearson']:
                    pearson_info += f" ({stats['pearson']['error']})"
        
        print(f"{category}: {stats['accuracy']*100:.2f}% ({stats['matches']}/{stats['total']}){pearson_info}")
    
    print(f"\nOverall accuracy: {overall_accuracy*100:.2f}%")
    if average_pearson is not None:
        print(f"Average Pearson coefficient: {average_pearson:.4f}")
    
    return overall_accuracy, average_pearson, accuracy_results

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize log and cache directories
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create output directory with model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reason_tag = "_with_reason" if args.reason else "_no_reason"
    output_path = os.path.join(args.output_dir, f"{args.model}{reason_tag}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Output will be saved to: {output_path}")
    
    # Read test data
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Limit sample count
    if args.limit > 0:
        test_data = test_data[:args.limit]
        print(f"Limited to processing {args.limit} samples")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Backup original category parameters to ensure no interference during mode switching
    original_categories = args.categories.copy()
    
    # Execute evaluation
    global_results = local_results = None
    
    if args.mode in ['global', 'both']:
        # For global evaluation, filter categories to ensure global evaluation categories are used
        if 'all' in original_categories:
            args.categories = ['all']  # Use 'all' flag, let run_global_evaluation select all global categories
        else:
            # Filter valid global categories
            valid_global_categories = [cat for cat in original_categories if cat in GLOBAL_EVALUATION_CATEGORIES]
            args.categories = valid_global_categories if valid_global_categories else ['all']
        
        # Execute global evaluation
        global_accuracy, global_pearson, global_category_results = run_global_evaluation(args, test_data, output_path)
        global_results = {
            'accuracy': global_accuracy,
            'pearson': global_pearson,
            'category_results': global_category_results
        }
    
    if args.mode in ['local', 'both']:
        # For local evaluation, filter categories to ensure local evaluation categories are used
        if 'all' in original_categories:
            args.categories = ['all']  # Use 'all' flag, let run_local_evaluation select all local categories
        else:
            # Filter valid local categories
            valid_local_categories = [cat for cat in original_categories if cat in LOCAL_EVALUATION_CATEGORIES]
            args.categories = valid_local_categories if valid_local_categories else ['all']
        
        # Execute local evaluation
        local_accuracy, local_pearson, local_category_results = run_local_evaluation(args, test_data, output_path)
        local_results = {
            'accuracy': local_accuracy,
            'pearson': local_pearson,
            'category_results': local_category_results
        }
    
    # Save comprehensive summary (if both modes are executed)
    if args.mode == 'both':
        combined_summary = {
            'global': global_results,
            'local': local_results
        }
        combined_file = os.path.join(output_path, "combined_summary.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_summary, f, ensure_ascii=False, indent=2)
        
        print("\n==== Comprehensive Evaluation Results ====")
        print(f"Global evaluation overall accuracy: {global_accuracy*100:.2f}%")
        if global_pearson is not None:
            print(f"Global evaluation average Pearson coefficient: {global_pearson:.4f}")
        
        print(f"Local evaluation overall accuracy: {local_accuracy*100:.2f}%")
        if local_pearson is not None:
            print(f"Local evaluation average Pearson coefficient: {local_pearson:.4f}")
    
    print(f"\nComplete summary saved to: {output_path}")

if __name__ == "__main__":
    main()