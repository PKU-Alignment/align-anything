import json

from qwen_vl_utils import process_vision_info
from system_prompt.score_eval_global_judge import get_global_judge_prompt
from system_prompt.score_eval_local_judge import get_local_judge_prompt
from utils.config import GLOBAL_EVALUATION_CATEGORIES, LOCAL_EVALUATION_CATEGORIES

def load_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def whole_conversation_data(raw_sample):
    prefix = 'Here is the whole conversation'

    # Get prompt and images
    basic_prompt = {
        'role': 'user',
        'content': [],
    }
    basic_prompt['content'].append({
        'type': 'text',
        'text': prefix,
    })
    for turn in raw_sample['conversations']:
        round_number = turn['round']
        if round_number == 0:
            continue
        turn_prompt = turn["round_conversations"]["prompt"]
        turn_response = turn["round_conversations"]["response"]
        turn_prompt_image = turn['round_conversations']['prompt_images'] if 'prompt_images' in turn['round_conversations'] else []
        turn_response_image = turn['round_conversations']['response_images'] if 'response_images' in turn['round_conversations'] else []
        
        # format prompt
        if len(turn_prompt_image) > 0 :
            for image in turn_prompt_image:
                basic_prompt['content'].append({
                    'type': 'image',
                    'image': image,
                })
            basic_prompt['content'].append({
                'type': 'text',
                'text': f'Round {round_number} Question: {turn_prompt}',
            })
        else:
            basic_prompt['content'].append({
                'type': 'text',
                'text': f'Round {round_number} Question: {turn_prompt}',
            })
        # format response 
        if len(turn_response_image) > 0:
            for image in turn_response_image:
                basic_prompt['content'].append({
                    'type': 'image',
                    'image': image,
                })
            basic_prompt['content'].append({
                'type': 'text',
                'text': f'Round {round_number} Response: {turn_response}',
            })
        else:
            basic_prompt['content'].append({
                'type': 'text',
                'text': f'Round {round_number} Response: {turn_response}',
            })
        
    # print(basic_prompt)
    image_inputs, video_inputs = process_vision_info([basic_prompt])
    return basic_prompt, image_inputs


def get_global_evaluation_data(test_item, if_reason:bool = False, evaluation_category:list = None):
    """Get global evaluation data
    
    Args:
        test_item: Test item
        if_reason: Whether to include reasoning
        evaluation_category: List of evaluation categories, if None or contains 'all', use all global evaluation categories
        
    Returns:
        basic_prompt, image_inputs, global_judge_prompt
    """
    if evaluation_category is None:
        evaluation_category = GLOBAL_EVALUATION_CATEGORIES
    elif 'all' in evaluation_category:
        evaluation_category = GLOBAL_EVALUATION_CATEGORIES
    
    basic_prompt, image_inputs = whole_conversation_data(test_item)
    global_judge_prompt = get_global_judge_prompt(if_reason, evaluation_category)
    
    return basic_prompt, image_inputs, global_judge_prompt
    
  
def get_single_turn_evaluation_data(test_item, if_reason:bool = False, evaluation_category:list = None):
    """Get single turn evaluation data
    
    Args:
        test_item: Test item
        if_reason: Whether to include reasoning
        evaluation_category: List of evaluation categories, if None or contains 'all', use all local evaluation categories
        
    Returns:
        basic_prompt, image_inputs, local_judge_prompt, total_round_number
    """
    if evaluation_category is None:
        evaluation_category = LOCAL_EVALUATION_CATEGORIES
    elif 'all' in evaluation_category:
        evaluation_category = LOCAL_EVALUATION_CATEGORIES
    
    basic_prompt, image_inputs = whole_conversation_data(test_item)
    total_round_number = len(test_item['conversations']) - 1  # Subtract 1 because basic_prompt includes round 0
    local_judge_prompt = get_local_judge_prompt(if_reason, evaluation_category)
    
    return basic_prompt, image_inputs, local_judge_prompt, total_round_number

