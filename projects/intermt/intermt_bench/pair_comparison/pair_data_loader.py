import json
from qwen_vl_utils import process_vision_info
from system_prompt.pair_eval_local_judge import get_local_judge_prompt
from system_prompt.pair_eval_global_judge import get_global_judge_prompt
from config import LOCAL_EVALUATION_CATEGORIES, GLOBAL_EVALUATION_CATEGORIES


import random
def get_local_evaluation_conversation(raw_sample):
    basic_prompt = {
        'role': 'user',
        'content': [],
    }
    if len(raw_sample['front_convs']) > 0:
        for turn in raw_sample['front_convs']:
            
            turn_prompt = turn['prompt']
            turn_response = turn['response']
            turn_prompt_images = turn['prompt_images'] if 'prompt_images' in turn else []
            turn_response_images = turn['response_images'] if 'response_images' in turn else []
            
            # format prompt
            if len(turn_prompt_images) > 0:
                for image in turn_prompt_images:
                    if image == '':
                        continue
                    basic_prompt['content'].append({
                        'type': 'image',
                        'image': image,
                    })
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Question: {turn_prompt}",
                })
            else:
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Question: {turn_prompt}",
                })
            if len(turn_response_images) > 0:
                for image in turn_response_images:
                    if image == '':
                        continue
                    basic_prompt['content'].append({
                        'type': 'image',
                        'image': image,
                    })
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Response: {turn_response}",
                })
            else:
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Response: {turn_response}",
                })
    pair_conv = raw_sample['paired_convs']
    last_prompt = pair_conv['prompt']
    last_prompt_images = pair_conv['prompt_images'] if 'prompt_images' in pair_conv else []
    
    # format prompt
    if len(last_prompt_images) > 0:
        for image in last_prompt_images:
            if image == '':
                continue
            basic_prompt['content'].append({
                'type': 'image',
                'image': image,
            })
    basic_prompt['content'].append({
        'type': 'text',
        'text': f"Round {pair_conv['round']} Question: {last_prompt}",
    })
    
    basic_prompt['content'].append({
        'type': 'text',
        'text': 'Now judge the quality of the following two responses (ResponseA and ResponseB).',
    })
    
    random_choice = random.choice([0, 1])
    response_1 = pair_conv['response_1']
    response_2 = pair_conv['response_2']
    if random_choice == 1:
        response_1, response_2 = response_2, response_1
    response_1_images = pair_conv['response_1_images'] if 'response_1_images' in pair_conv else []
    response_2_images = pair_conv['response_2_images'] if 'response_2_images' in pair_conv else []
    if random_choice == 1:
        response_1_images, response_2_images = response_2_images, response_1_images
    
    # format response
    if len(response_1_images) > 0:
        for image in response_1_images:
            if image == '':
                continue
            basic_prompt['content'].append({
                'type': 'image',
                'image': image,
            })
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ResponseA: {response_1}',
        })
    else:
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ResponseA: {response_1}',
        })
    if len(response_2_images) > 0:
        for image in response_2_images:
            if image == '':
                continue
            basic_prompt['content'].append({
                'type': 'image',
                'image': image,
            })
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ResponseB: {response_2}',
        })
    else:
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ResponseB: {response_2}',
        })

    image_inputs, video_inputs = process_vision_info([basic_prompt])
    return basic_prompt, image_inputs, random_choice



def get_global_evaluation_conversation(raw_sample):
    basic_prompt = {
        'role': 'user',
        'content': [],
    }
    if len(raw_sample['front_convs']) > 0:
        for turn in raw_sample['front_convs']:
            
            turn_prompt = turn['prompt']
            turn_response = turn['response']
            turn_prompt_images = turn['prompt_images'] if 'prompt_images' in turn else []
            turn_response_images = turn['response_images'] if 'response_images' in turn else []
            
            # format prompt
            if len(turn_prompt_images) > 0:
                for image in turn_prompt_images:
                    if image == '':
                        continue
                    basic_prompt['content'].append({
                        'type': 'image',
                        'image': image,
                    })
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Question: {turn_prompt}",
                })
            else:
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Question: {turn_prompt}",
                })
            if len(turn_response_images) > 0:
                for image in turn_response_images:
                    if image == '':
                        continue
                    basic_prompt['content'].append({
                        'type': 'image',
                        'image': image,
                    })
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Response: {turn_response}",
                })
            else:
                basic_prompt['content'].append({
                    'type': 'text',
                    'text': f"Round {turn['round']} Response: {turn_response}",
                })
    pair_conv = raw_sample['paired_convs']
    last_prompt = pair_conv['prompt']
    last_prompt_images = pair_conv['prompt_images'] if 'prompt_images' in pair_conv else []
    
    # format prompt
    if len(last_prompt_images) > 0:
        for image in last_prompt_images:
            if image == '':
                continue
            basic_prompt['content'].append({
                'type': 'image',
                'image': image,
            })
    basic_prompt['content'].append({
        'type': 'text',
        'text': f"Round {pair_conv['round']} Question: {last_prompt}",
    })
    
    basic_prompt['content'].append({
        'type': 'text',
        'text': 'Now judge the quality of the overall two conversations. (ResponseA and ResponseB).',
    })
    
    random_choice = random.choice([0, 1])
    response_1 = pair_conv['response_1']
    response_2 = pair_conv['response_2']
    if random_choice == 1:
        response_1, response_2 = response_2, response_1
    response_1_images = pair_conv['response_1_images'] if 'response_1_images' in pair_conv else []
    response_2_images = pair_conv['response_2_images'] if 'response_2_images' in pair_conv else []
    if random_choice == 1:
        response_1_images, response_2_images = response_2_images, response_1_images
    
    # format response
    if len(response_1_images) > 0:
        for image in response_1_images:
            if image == '':
                continue
            basic_prompt['content'].append({
                'type': 'image',
                'image': image,
            })
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ConversationA last response: {response_1}',
        })
    else:
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ConversationA last response: {response_1}',
        })
    if len(response_2_images) > 0:
        for image in response_2_images:
            if image == '':
                continue
            basic_prompt['content'].append({
                'type': 'image',
                'image': image,
            })
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ConversationB last response: {response_2}',
        })
    else:
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'ConversationB last response: {response_2}',
        })

    image_inputs, video_inputs = process_vision_info([basic_prompt])
    return basic_prompt, image_inputs, random_choice


def get_local_evaluation_data(test_item, if_reason:bool = False, evaluation_category:list = None):
    if evaluation_category is None:
        evaluation_category = LOCAL_EVALUATION_CATEGORIES
    local_judge_prompt = get_local_judge_prompt(if_reason=if_reason, evaluation_list=evaluation_category)
    basic_prompt, image_inputs, random_choice = get_local_evaluation_conversation(test_item)
    return basic_prompt, image_inputs, local_judge_prompt, random_choice






def get_global_evaluation_data(test_item, if_reason:bool = False, evaluation_category:list = None):
    if evaluation_category is None:
        evaluation_category = GLOBAL_EVALUATION_CATEGORIES
    global_judge_prompt = get_global_judge_prompt(if_reason=if_reason, evaluation_list=evaluation_category)
    basic_prompt, image_inputs, random_choice = get_global_evaluation_conversation(test_item)
    return basic_prompt, image_inputs, global_judge_prompt, random_choice
    
    
    
    
    
    
    