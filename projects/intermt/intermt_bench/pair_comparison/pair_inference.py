import json
import os
import re
import numpy as np
import argparse
from datetime import datetime
from config import GLOBAL_TEST_FILE, LOCAL_TEST_FILE, MODEL_NAME, TEMPERATURE, GLOBAL_EVALUATION_CATEGORIES, LOCAL_EVALUATION_CATEGORIES
from tqdm import tqdm
from pair_data_loader import get_global_evaluation_data, get_local_evaluation_data
from api_utils import process_message_content, batch_api_call
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
                        help='Evaluation mode: global(overall evaluation), local(local evaluation), both(both modes)')
    
    # Evaluation parameters
    parser.add_argument('--categories', type=str, default=None,
                        help='Evaluation categories, comma-separated, or use "all" for all categories')
    parser.add_argument('--reason', action='store_true', default=True,
                        help='Include reasoning')
    parser.add_argument('--no-reason', action='store_false', dest='reason',
                        help='Do not include reasoning')
    
    # File paths
    parser.add_argument('--output-dir', type=str, default="./output",
                        help='Output directory')
    parser.add_argument('--cache-dir', type=str, default="./cache",
                        help='Cache directory')
    
    # Other parameters
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit the number of samples to process, 0 means process all samples')
    
    args = parser.parse_args()
    
    # Process evaluation categories
    if args.categories:
        if args.categories == 'all':
            args.categories = ['all']
        else:
            args.categories = args.categories.split(',')
    else:
        # Set default categories based on mode
        if args.mode == 'global':
            args.categories = GLOBAL_EVALUATION_CATEGORIES
        elif args.mode == 'local':
            args.categories = LOCAL_EVALUATION_CATEGORIES
        else:  # both
            args.categories = ['all']
    
    return args

def load_data(test_file):
    with open(test_file, 'r') as f:
        data = json.load(f)
    return data

def extract_preference_and_reasons(output, categories):
    """
    Extract preference (ResponseA or ResponseB) and reasons for each category from model output
    
    Args:
        output: Model output text
        categories: List of evaluation categories
        
    Returns:
        preferences: List of extracted preferences (1 for ResponseA, 2 for ResponseB)
        reasons: List of extracted reasons
    """
    preferences = []
    reasons = []
    
    # Clean output text
    output = output.strip()
    
    # Handle possible "Evaluation list:" prefix
    if "Evaluation list:" in output:
        output = output.split("Evaluation list:", 1)[1].strip()
    
    # Create patterns that adapt to various boxed formats
    def make_boxed_pattern(base_pattern):
        boxed_variations = [
            r"\\\\boxed\{\{(ResponseA|ResponseB)\}\}",  # \\boxed{{ResponseX}}
            r"\\boxed\{\{(ResponseA|ResponseB)\}\}",    # \boxed{{ResponseX}}
            r"\\\\boxed\{(ResponseA|ResponseB)\}",      # \\boxed{ResponseX}
            r"\\boxed\{(ResponseA|ResponseB)\}"         # \boxed{ResponseX}
        ]
        patterns = []
        for boxed in boxed_variations:
            patterns.append(base_pattern.replace("BOXED_PLACEHOLDER", boxed))
        return patterns
    
    # Try different format patterns
    format_types = {
        "comma_separated_brackets": r"\[\s*[\w_]+\s*,[^[]*?BOXED_PLACEHOLDER\s*\],",  # [category, text, \boxed{ResponseX}],
        "newline_separated_brackets": r"\[\s*[\w_]+\s*,[^[]*?BOXED_PLACEHOLDER\s*\]\s*\n",  # [category, text, \boxed{ResponseX}]\n
        "double_brackets": r"\[\[\s*[\w_]+\s*,[^[]*?BOXED_PLACEHOLDER\s*\]\]",  # [[category, text, \boxed{ResponseX}]]
        "no_comma_brackets": r"\[\[\s*[\w_]+\s+[^,\[]*?BOXED_PLACEHOLDER\s*\]\]",  # [[category text \boxed{ResponseX}]]
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
    
    # Determine the best format
    best_format = max(format_counts.items(), key=lambda x: x[1])[0] if format_counts else None
    
    # Define extraction logic for different formats
    def extract_by_format(output_text, category, format_type):
        preference_found = None
        reason_found = ""
        
        # Base patterns for different formats
        if format_type == "comma_separated_brackets":
            base_pattern = r"\[\s*" + re.escape(category) + r"\s*,\s*(.*?),\s*BOXED_PLACEHOLDER\s*\],"
        elif format_type == "newline_separated_brackets":
            base_pattern = r"\[\s*" + re.escape(category) + r"\s*,\s*(.*?),\s*BOXED_PLACEHOLDER\s*\]\s*[\n,]"
        elif format_type == "double_brackets":
            base_pattern = r"\[\[\s*" + re.escape(category) + r"\s*,\s*(.*?),\s*BOXED_PLACEHOLDER\s*\]\]"
        elif format_type == "no_comma_brackets":
            base_pattern = r"\[\[\s*" + re.escape(category) + r"\s+(.*?)BOXED_PLACEHOLDER\s*\]\]"
        else:
            return preference_found, reason_found
        
        # Try all boxed variants
        patterns = make_boxed_pattern(base_pattern)
        for pattern in patterns:
            match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
            if match:
                if format_type == "no_comma_brackets":
                    reason_found = match.group(1).strip()
                    preference_found = match.group(2).lower()
                else:
                    reason_found = match.group(1).strip()
                    preference_found = match.group(2).lower()
                
                # Convert to numeric representation
                if preference_found == "responsea":
                    preference_found = 1
                elif preference_found == "responseb":
                    preference_found = 2
                    
                return preference_found, reason_found
        
        return preference_found, reason_found
    
    # First try to extract using the best format
    if best_format:
        all_extracted = True
        temp_preferences = []
        temp_reasons = []
        
        for category in categories:
            preference, reason = extract_by_format(output, category, best_format)
            if preference is None:
                all_extracted = False
            temp_preferences.append(preference)
            temp_reasons.append(reason)
        
        # If the best format successfully extracted all categories, use this result
        if all_extracted:
            return temp_preferences, temp_reasons
    
    # If the best format can't extract all categories, try the most suitable format for each category
    mixed_preferences = [None] * len(categories)
    mixed_reasons = [""] * len(categories)
    
    for i, category in enumerate(categories):
        # Try all formats until finding one that can extract
        for fmt in format_types.keys():
            preference, reason = extract_by_format(output, category, fmt)
            if preference is not None:
                mixed_preferences[i] = preference
                mixed_reasons[i] = reason
                break
    
    # If mixed format extraction results are better than single format, use mixed results
    if sum(1 for p in mixed_preferences if p is not None) > sum(1 for p in temp_preferences if p is not None):
        temp_preferences, temp_reasons = mixed_preferences, mixed_reasons
    
    # If the above methods still can't extract all preferences, use more general regex patterns
    if None in temp_preferences:
        for i, category in enumerate(categories):
            if temp_preferences[i] is None:
                # Try to directly match combinations of category and ResponseA/ResponseB
                patterns = [
                    r"\[\[\s*" + re.escape(category) + r".*?(ResponseA|ResponseB)\s*\]\]",
                    r"\[\s*" + re.escape(category) + r".*?(ResponseA|ResponseB)\s*\]",
                    re.escape(category) + r".*?(ResponseA|ResponseB)"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                    if match:
                        preference = match.group(1).lower()
                        if preference == "responsea":
                            temp_preferences[i] = 1
                        elif preference == "responseb":
                            temp_preferences[i] = 2
                        break
    
    return temp_preferences, temp_reasons

def calculate_pearson(model_scores, gt_scores):
    """Calculate Pearson correlation coefficient"""
    # Filter out None values
    valid_pairs = [(m, g) for m, g in zip(model_scores, gt_scores) if m is not None]
    if len(valid_pairs) < 2:  # Pearson correlation requires at least 2 points
        return None
    
    m_scores, g_scores = zip(*valid_pairs)
    
    # Check for constant arrays
    if len(set(m_scores)) <= 1 or len(set(g_scores)) <= 1:
        return {'coefficient': "Constant array", 'p_value': None, 'note': "Input arrays contain constant values, cannot calculate correlation coefficient"}
    
    try:
        from scipy.stats import pearsonr
        coef, p_value = pearsonr(m_scores, g_scores)
        return {'coefficient': coef, 'p_value': p_value}
    except Exception as e:
        return {'coefficient': None, 'p_value': None, 'error': str(e)}

def calculate_additional_metrics(model_scores, gt_scores):
    """Calculate additional evaluation metrics"""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    import numpy as np
    
    # Filter out None values
    valid_pairs = [(m, g) for m, g in zip(model_scores, gt_scores) if m is not None]
    if len(valid_pairs) < 2:
        return {'accuracy': None, 'kappa': None, 'f1': None, 'note': "Insufficient samples"}
    
    m_scores, g_scores = zip(*valid_pairs)
    
    try:
        # Calculate accuracy
        accuracy = accuracy_score(g_scores, m_scores)
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(g_scores, m_scores)
        
        # For F1 score, need to binarize labels
        # Assuming labels are 1 and 2, we convert 2 to 0 for binary F1 calculation
        binary_gt = [1 if g == 1 else 0 for g in g_scores]
        binary_model = [1 if m == 1 else 0 for m in m_scores]
        f1 = f1_score(binary_gt, binary_model, average='binary')
        
        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'f1': f1,
            'sample_size': len(m_scores)
        }
    except Exception as e:
        return {'accuracy': None, 'kappa': None, 'f1': None, 'error': str(e)}

def calculate_pearson_with_ci(model_scores, gt_scores, confidence=0.95):
    """Calculate Pearson correlation coefficient with confidence interval"""
    from scipy.stats import pearsonr
    from numpy import arctanh, tanh
    import numpy as np
    
    # Filter out None values
    valid_pairs = [(m, g) for m, g in zip(model_scores, gt_scores) if m is not None]
    if len(valid_pairs) < 5:  # Need at least 5 samples
        return {'coefficient': None, 'p_value': None, 'ci_low': None, 'ci_high': None, 'note': "Insufficient samples"}
    
    m_scores, g_scores = zip(*valid_pairs)
    
    # Check for constant arrays
    if len(set(m_scores)) <= 1 or len(set(g_scores)) <= 1:
        return {'coefficient': "Constant array", 'p_value': None, 'ci_low': None, 'ci_high': None, 'note': "Input arrays contain constant values, cannot calculate correlation coefficient"}
    
    try:
        coef, p_value = pearsonr(m_scores, g_scores)
        n = len(m_scores)
        
        # Use Fisher transformation to calculate confidence interval
        z = 0.5 * np.log((1 + coef) / (1 - coef))  # Fisher transformation
        se = 1 / np.sqrt(n - 3)
        z_crit = 1.96  # z-value for 95% confidence interval
        
        z_low = z - z_crit * se
        z_high = z + z_crit * se
        
        # Inverse transform back to correlation coefficient
        r_low = tanh(z_low)
        r_high = tanh(z_high)
        
        return {
            'coefficient': coef, 
            'p_value': p_value,
            'ci_low': r_low,
            'ci_high': r_high,
            'sample_size': n
        }
    except Exception as e:
        return {'coefficient': None, 'p_value': None, 'ci_low': None, 'ci_high': None, 'error': str(e)}

def analyze_preference_distribution(evaluation_category, test_data):
    """Analyze preference distribution"""
    distribution = {cat: {'1': 0, '2': 0, '0': 0} for cat in evaluation_category}
    
    for item in test_data:
        for cat in evaluation_category:
            if 'overall_preference' in item:
                pref = item['overall_preference'].get(cat, None)
            elif 'local_overall_preference' in item:
                pref = item['local_overall_preference'].get(cat, None)
            else:
                pref = None
            if pref is not None:
                distribution[cat][str(pref)] += 1
    
    print("\nPreference distribution by dimensions:")
    for cat, counts in distribution.items():
        total = sum(counts.values())
        if total > 0:
            print(f"{cat}: Total samples {total}, Preference 1: {counts['1']}({counts['1']/total*100:.1f}%), Preference 2: {counts['2']}({counts['2']/total*100:.1f}%), Equal: {counts['0']}({counts['0']/total*100:.1f}%)")
    
    return distribution

def global_inference(if_reason:bool = False, evaluation_category:list = None, output_dir:str = "./output", model_name:str = MODEL_NAME, temperature:float = TEMPERATURE, limit:int = 0, cache_dir:str = "./cache"):
    test_data = load_data(GLOBAL_TEST_FILE)
    if limit > 0:
        test_data = random.sample(test_data, min(limit, len(test_data)))
    system_prompts = []
    message_contents = []
    debug_data = []
    
    distribution = analyze_preference_distribution(evaluation_category, test_data)
    
    for idx, item in tqdm(enumerate(test_data), desc='Global Inference, preparing data', total=len(test_data)):
        basic_prompt, image_inputs, global_judge_prompt, random_choice = get_global_evaluation_data(item, if_reason=if_reason, evaluation_category=evaluation_category)
        ground_truth = item['overall_preference']
        if random_choice == 1:
            for k,v in ground_truth.items():
                ground_truth[k] = 3 - v
        
        processed_content = process_message_content(basic_prompt['content'])
        system_prompts.append(global_judge_prompt)
        message_contents.append(processed_content)
        
        debug_item = {
            'input': {
                'system_prompt': global_judge_prompt,
                'basic_prompt': basic_prompt
            },
            'item_id': idx,
            'raw_item': item,
            'ground_truth': {
                'preference': ground_truth
            }
        }
        debug_data.append(debug_item)
    print(f'Global Inference, preparing data done, {len(debug_data)} items')
    print(f'Global Inference, start inference')
    results = batch_api_call(
        system_contents=system_prompts,
        message_contents=message_contents,
        model_name=model_name,
        temperature=temperature,
        cache_dir=cache_dir
    )
    
    print(f'Global Inference, inference done, {len(results)} items')

    # Create global output directory
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    
    # Compare model output with ground truth
    category_matches = {category: {'total': 0, 'matches': 0} for category in evaluation_category}
    category_pearson_data = {category: {'model': [], 'ground_truth': []} for category in evaluation_category}
    
    # Integrate results
    for i, result in enumerate(results):
        debug_data[i]['output'] = result
        
        # Extract model preferences and reasons
        model_preferences, reasons = extract_preference_and_reasons(result, evaluation_category)
        debug_data[i]['extracted_preferences'] = model_preferences
        debug_data[i]['extracted_reasons'] = reasons
        
        # Compare preferences
        gt_preferences = [debug_data[i]['ground_truth']['preference'].get(category, None) for category in evaluation_category]
        matches = []
        
        for j, (model_pref, gt_pref) in enumerate(zip(model_preferences, gt_preferences)):
            if model_pref is not None and gt_pref is not None and gt_pref != 0:  # Exclude cases where gt_pref is 0
                category = evaluation_category[j]
                category_matches[category]['total'] += 1
                # Add to Pearson correlation calculation data
                category_pearson_data[category]['model'].append(model_pref)
                category_pearson_data[category]['ground_truth'].append(gt_pref)
                
                if int(model_pref) == int(gt_pref):
                    category_matches[category]['matches'] += 1
                    matches.append(True)
                else:
                    matches.append(False)
            else:
                matches.append(None)
        
        debug_data[i]['preference_matches'] = matches
        
        # Print detailed evaluation results for the first 5 samples
        if i < 5:
            print(f"\nSample {debug_data[i]['item_id']} evaluation comparison:")
            print(f"{'Category':<25} {'Model Pref':<10} {'Ground Truth':<10} {'Match':<8} {'Reason':<30}")
            print("-" * 80)
            for j, category in enumerate(evaluation_category):
                model_pref = model_preferences[j] if model_preferences[j] is not None else "Not extracted"
                gt_pref = gt_preferences[j] if gt_preferences[j] is not None else "Not labeled"
                match_str = "✓" if matches[j] == True else "✗" if matches[j] == False else "No match"
                reason = (reasons[j][:27] + "..." if len(reasons[j]) > 30 else reasons[j]) if reasons and reasons[j] else ""
                print(f"{category:<25} {model_pref:<10} {gt_pref:<10} {match_str:<8} {reason:<30}")
            print("-" * 80)
    
    # Calculate each dimension's consistency ratio and Pearson correlation coefficient
    accuracy_results = {}
    for category, stats in category_matches.items():
        if stats['total'] > 0:
            accuracy = stats['matches'] / stats['total']
            pearson_data = category_pearson_data[category]
            
            # Calculate Pearson correlation coefficient
            pearson_result = None
            if len(pearson_data['model']) >= 2:
                pearson_result = calculate_pearson_with_ci(
                    pearson_data['model'], 
                    pearson_data['ground_truth']
                )
            
            # Calculate additional metrics
            additional_metrics = None
            if len(pearson_data['model']) >= 2:
                additional_metrics = calculate_additional_metrics(
                    pearson_data['model'],
                    pearson_data['ground_truth']
                )
            
            accuracy_results[category] = {
                'accuracy': accuracy,
                'matches': stats['matches'],
                'total': stats['total'],
                'pearson': pearson_result,
                'additional_metrics': additional_metrics
            }
        else:
            accuracy_results[category] = {
                'accuracy': 0,
                'matches': 0,
                'total': 0,
                'pearson': None,
                'additional_metrics': None
            }
    
    # Save complete debug data
    debug_file = os.path.join(global_dir, "debug.json")
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    
    # Save clean output results
    clean_results = []
    for i, result in enumerate(results):
        clean_item = {
            'item_id': debug_data[i]['item_id'],
            'output': result,
            'extracted_preferences': debug_data[i]['extracted_preferences'],
            'extracted_reasons': debug_data[i]['extracted_reasons'],
            'ground_truth': [debug_data[i]['ground_truth']['preference'].get(category, None) for category in evaluation_category],
            'preference_matches': debug_data[i]['preference_matches']
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
        'model_name': model_name,
        'temperature': temperature,
        'evaluation_categories': evaluation_category,
        'include_reason': if_reason,
        'test_file': GLOBAL_TEST_FILE,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'limit': limit if limit > 0 else len(test_data)
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
    
    # Save model overall performance summary
    overall_accuracy = sum([stats['matches'] for stats in category_matches.values()]) / sum([stats['total'] for stats in category_matches.values()]) if sum([stats['total'] for stats in category_matches.values()]) > 0 else 0
    
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
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']}"
            elif stats['pearson']['coefficient'] is not None:
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']:.4f}"
                if stats['pearson']['p_value'] is not None:
                    pearson_info += f" (p={stats['pearson']['p_value']:.4f})"
                # Add confidence interval information
                if 'ci_low' in stats['pearson'] and stats['pearson']['ci_low'] is not None:
                    pearson_info += f", 95%CI:[{stats['pearson']['ci_low']:.4f}, {stats['pearson']['ci_high']:.4f}]"
                # Add sample size information
                if 'sample_size' in stats['pearson']:
                    pearson_info += f", n={stats['pearson']['sample_size']}"
            
        # Add additional metrics information
        if stats.get('additional_metrics'):
            metrics = stats['additional_metrics']
            if metrics.get('accuracy') is not None:
                pearson_info += f", Accuracy: {metrics['accuracy']:.4f}"
            if metrics.get('kappa') is not None:
                pearson_info += f", Kappa: {metrics['kappa']:.4f}"
            if metrics.get('f1') is not None:
                pearson_info += f", F1: {metrics['f1']:.4f}"
        
        print(f"{category}: {stats['accuracy']*100:.2f}% ({stats['matches']}/{stats['total']}){pearson_info}")
    
    print(f"\nOverall accuracy: {overall_accuracy*100:.2f}%")
    if average_pearson is not None:
        print(f"Average Pearson coefficient: {average_pearson:.4f}")
    
    return overall_accuracy, average_pearson, accuracy_results


def local_inference(if_reason:bool = False, evaluation_category:list = None, output_dir:str = "./output", model_name:str = MODEL_NAME, temperature:float = TEMPERATURE, limit:int = 0, cache_dir:str = "./cache"):
    test_data = load_data(LOCAL_TEST_FILE)
    if limit > 0:
        test_data = random.sample(test_data, min(limit, len(test_data)))
    system_prompts = []
    message_contents = []
    debug_data = []
    
    distribution = analyze_preference_distribution(evaluation_category, test_data)
    
    for idx, item in tqdm(enumerate(test_data), desc='Local Inference, preparing data', total=len(test_data)):
        basic_prompt, image_inputs, local_judge_prompt, random_choice = get_local_evaluation_data(item, if_reason=if_reason, evaluation_category=evaluation_category)
        ground_truth = item['local_overall_preference']
        if random_choice == 1:
            for k,v in ground_truth.items():
                ground_truth[k] = 3 - v
        
        processed_content = process_message_content(basic_prompt['content'])
        system_prompts.append(local_judge_prompt)
        message_contents.append(processed_content)
        
        debug_item = {
            'input': {
                'system_prompt': local_judge_prompt,
                'basic_prompt': basic_prompt
            },
            'item_id': idx,
            'raw_item': item,
            'ground_truth': {
                'preference': ground_truth
            }
        }
        debug_data.append(debug_item)
    print(f'Local Inference, preparing data done, {len(debug_data)} items')
    print(f'Local Inference, start inference')
    results = batch_api_call(
        system_contents=system_prompts,
        message_contents=message_contents,
        model_name=model_name,
        temperature=temperature,
        cache_dir=cache_dir
    )
    
    print(f'Local Inference, inference done, {len(results)} items')
    
    # Create local evaluation output directory
    local_dir = os.path.join(output_dir, "local")
    os.makedirs(local_dir, exist_ok=True)
    
    # Compare model output with ground truth
    category_matches = {category: {'total': 0, 'matches': 0} for category in evaluation_category}
    category_pearson_data = {category: {'model': [], 'ground_truth': []} for category in evaluation_category}
    
    # Integrate results
    for i, result in enumerate(results):
        debug_data[i]['output'] = result
        
        # Extract model preferences and reasons
        model_preferences, reasons = extract_preference_and_reasons(result, evaluation_category)
        debug_data[i]['extracted_preferences'] = model_preferences
        debug_data[i]['extracted_reasons'] = reasons
        
        # Compare preferences
        gt_preferences = [debug_data[i]['ground_truth']['preference'].get(category, None) for category in evaluation_category]
        matches = []
        
        for j, (model_pref, gt_pref) in enumerate(zip(model_preferences, gt_preferences)):
            if model_pref is not None and gt_pref is not None and gt_pref != 0:  # Exclude cases where gt_pref is 0
                category = evaluation_category[j]
                category_matches[category]['total'] += 1
                # Add to Pearson correlation calculation data
                category_pearson_data[category]['model'].append(model_pref)
                category_pearson_data[category]['ground_truth'].append(gt_pref)
                
                if int(model_pref) == int(gt_pref):
                    category_matches[category]['matches'] += 1
                    matches.append(True)
                else:
                    matches.append(False)
            else:
                matches.append(None)
        
        debug_data[i]['preference_matches'] = matches
        
        # Print detailed evaluation results for the first 5 samples
        if i < 5:
            print(f"\nSample {debug_data[i]['item_id']} evaluation comparison:")
            print(f"{'Category':<25} {'Model Pref':<10} {'Ground Truth':<10} {'Match':<8} {'Reason':<30}")
            print("-" * 80)
            for j, category in enumerate(evaluation_category):
                model_pref = model_preferences[j] if model_preferences[j] is not None else "Not extracted"
                gt_pref = gt_preferences[j] if gt_preferences[j] is not None else "Not labeled"
                match_str = "✓" if matches[j] == True else "✗" if matches[j] == False else "No match"
                reason = (reasons[j][:27] + "..." if len(reasons[j]) > 30 else reasons[j]) if reasons and reasons[j] else ""
                print(f"{category:<25} {model_pref:<10} {gt_pref:<10} {match_str:<8} {reason:<30}")
            print("-" * 80)
    
    # Calculate each dimension's consistency ratio and Pearson correlation coefficient
    accuracy_results = {}
    for category, stats in category_matches.items():
        if stats['total'] > 0:
            accuracy = stats['matches'] / stats['total']
            pearson_data = category_pearson_data[category]
            
            # Calculate Pearson correlation coefficient
            pearson_result = None
            if len(pearson_data['model']) >= 2:
                pearson_result = calculate_pearson_with_ci(
                    pearson_data['model'], 
                    pearson_data['ground_truth']
                )
            
            # Calculate additional metrics
            additional_metrics = None
            if len(pearson_data['model']) >= 2:
                additional_metrics = calculate_additional_metrics(
                    pearson_data['model'],
                    pearson_data['ground_truth']
                )
            
            accuracy_results[category] = {
                'accuracy': accuracy,
                'matches': stats['matches'],
                'total': stats['total'],
                'pearson': pearson_result,
                'additional_metrics': additional_metrics
            }
        else:
            accuracy_results[category] = {
                'accuracy': 0,
                'matches': 0,
                'total': 0,
                'pearson': None,
                'additional_metrics': None
            }
    
    # Save complete debug data
    debug_file = os.path.join(local_dir, "debug.json")
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    
    # Save clean output results
    clean_results = []
    for i, result in enumerate(results):
        clean_item = {
            'item_id': debug_data[i]['item_id'],
            'output': result,
            'extracted_preferences': debug_data[i]['extracted_preferences'],
            'extracted_reasons': debug_data[i]['extracted_reasons'],
            'ground_truth': [debug_data[i]['ground_truth']['preference'].get(category, None) for category in evaluation_category],
            'preference_matches': debug_data[i]['preference_matches']
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
        'model_name': model_name,
        'temperature': temperature,
        'evaluation_categories': evaluation_category,
        'include_reason': if_reason,
        'test_file': LOCAL_TEST_FILE,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'limit': limit if limit > 0 else len(test_data)
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
    
    # Save model overall performance summary
    overall_accuracy = sum([stats['matches'] for stats in category_matches.values()]) / sum([stats['total'] for stats in category_matches.values()]) if sum([stats['total'] for stats in category_matches.values()]) > 0 else 0
    
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
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']}"
            elif stats['pearson']['coefficient'] is not None:
                pearson_info = f", Pearson coefficient: {stats['pearson']['coefficient']:.4f}"
                if stats['pearson']['p_value'] is not None:
                    pearson_info += f" (p={stats['pearson']['p_value']:.4f})"
                # Add confidence interval information
                if 'ci_low' in stats['pearson'] and stats['pearson']['ci_low'] is not None:
                    pearson_info += f", 95%CI:[{stats['pearson']['ci_low']:.4f}, {stats['pearson']['ci_high']:.4f}]"
                # Add sample size information
                if 'sample_size' in stats['pearson']:
                    pearson_info += f", n={stats['pearson']['sample_size']}"
            
        # Add additional metrics information
        if stats.get('additional_metrics'):
            metrics = stats['additional_metrics']
            if metrics.get('accuracy') is not None:
                pearson_info += f", Accuracy: {metrics['accuracy']:.4f}"
            if metrics.get('kappa') is not None:
                pearson_info += f", Kappa: {metrics['kappa']:.4f}"
            if metrics.get('f1') is not None:
                pearson_info += f", F1: {metrics['f1']:.4f}"
        
        print(f"{category}: {stats['accuracy']*100:.2f}% ({stats['matches']}/{stats['total']}){pearson_info}")
    
    print(f"\nOverall accuracy: {overall_accuracy*100:.2f}%")
    if average_pearson is not None:
        print(f"Average Pearson coefficient: {average_pearson:.4f}")
    
    return overall_accuracy, average_pearson, accuracy_results

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create root output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reason_tag = "_with_reason" if args.reason else "_no_reason"
    root_output_dir = os.path.join(args.output_dir, f"{args.model}{reason_tag}_{timestamp}")
    os.makedirs(root_output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {root_output_dir}")
    
    # Execute different evaluations based on mode
    if args.mode in ['global', 'both']:
        # For global evaluation, ensure using global evaluation categories
        global_categories = args.categories
        if 'all' in global_categories:
            global_categories = GLOBAL_EVALUATION_CATEGORIES
        else:
            # Filter valid global categories
            global_categories = [cat for cat in global_categories if cat in GLOBAL_EVALUATION_CATEGORIES]
            if not global_categories:
                global_categories = GLOBAL_EVALUATION_CATEGORIES
                print("Warning: Specified categories are not suitable for global evaluation, using default global evaluation categories")
        
        print(f"\n==== Executing global evaluation ====")
        print(f"Model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print(f"Evaluation categories: {', '.join(global_categories)}")
        print(f"Include reasoning: {args.reason}")
        if args.limit > 0:
            print(f"Sample limit: {args.limit}")
        
        # Execute global evaluation
        global_accuracy, global_pearson, global_category_results = global_inference(
            if_reason=args.reason, 
            evaluation_category=global_categories,
            output_dir=root_output_dir,
            model_name=args.model,
            temperature=args.temperature,
            limit=args.limit,
            cache_dir=args.cache_dir
        )
    
    if args.mode in ['local', 'both']:
        # For local evaluation, ensure using local evaluation categories
        local_categories = args.categories
        if 'all' in local_categories:
            local_categories = LOCAL_EVALUATION_CATEGORIES
        else:
            # Filter valid local categories
            local_categories = [cat for cat in local_categories if cat in LOCAL_EVALUATION_CATEGORIES]
            if not local_categories:
                local_categories = LOCAL_EVALUATION_CATEGORIES
                print("Warning: Specified categories are not suitable for local evaluation, using default local evaluation categories")
        
        print(f"\n==== Executing local evaluation ====")
        print(f"Model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print(f"Evaluation categories: {', '.join(local_categories)}")
        print(f"Include reasoning: {args.reason}")
        if args.limit > 0:
            print(f"Sample limit: {args.limit}")
        
        # Execute local evaluation
        local_accuracy, local_pearson, local_category_results = local_inference(
            if_reason=args.reason, 
            evaluation_category=local_categories,
            output_dir=root_output_dir,
            model_name=args.model,
            temperature=args.temperature,
            limit=args.limit,
            cache_dir=args.cache_dir
        )
    
    # If both modes are executed, output combined results
    if args.mode == 'both':
        print("\n==== Combined evaluation results ====")
        print(f"Global evaluation overall accuracy: {global_accuracy*100:.2f}%")
        if global_pearson is not None:
            print(f"Global evaluation average Pearson coefficient: {global_pearson:.4f}")
        
        print(f"Local evaluation overall accuracy: {local_accuracy*100:.2f}%")
        if local_pearson is not None:
            print(f"Local evaluation average Pearson coefficient: {local_pearson:.4f}")
        
        # Save combined results
        combined_results = {
            'global': {
                'accuracy': global_accuracy,
                'pearson': global_pearson,
                'category_results': global_category_results
            },
            'local': {
                'accuracy': local_accuracy,
                'pearson': local_pearson,
                'category_results': local_category_results
            },
            'config': {
                'model': args.model,
                'temperature': args.temperature,
                'include_reason': args.reason,
                'limit': args.limit,
                'timestamp': timestamp
            }
        }
        
        combined_file = os.path.join(root_output_dir, "combined_summary.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nCombined evaluation results saved to: {combined_file}")
    
    print(f"\nAll evaluation results saved to: {root_output_dir}")








