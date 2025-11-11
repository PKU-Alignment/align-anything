import json
from qwen_vl_utils import process_vision_info   
from system_prompt import JUDGE_PROMPT, INFERENCE_PROMPT

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
        turn_prompt = turn["prompt"]
        turn_response = turn["response"]
        turn_prompt_image = turn['prompt_images'] if 'prompt_images' in turn else []
        turn_response_image = turn['response_images'] if 'response_images' in turn else []

        # format prompt
        if len(turn_prompt_image) > 0 :
            for image in turn_prompt_image:
                if image == '':
                    continue
                basic_prompt['content'].append({
                    'type': 'image',
                    'image': image,
                })
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'Round {round_number} Question: {turn_prompt}',
        })
        if len(turn_response_image) > 0 :
            for image in turn_response_image:
                if image == '':
                    continue
                basic_prompt['content'].append({
                    'type': 'image',
                    'image': image,
                })
        basic_prompt['content'].append({
            'type': 'text',
            'text': f'Round {round_number} Response: {turn_response}',
        })
    image_inputs = process_vision_info([basic_prompt])
    return basic_prompt, image_inputs
    

def get_evaluation_data(raw_sample):
    basic_prompt, image_inputs = whole_conversation_data(raw_sample)
    inference_prompt = INFERENCE_PROMPT

    return basic_prompt, image_inputs, inference_prompt



