# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pickle
import os
from datasets import load_dataset
import argparse
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.evaluation.eval_logger import EvalLogger
import requests
import time
from tqdm import tqdm

gpt_prompt = """
You are a helpful and precise assistant for checking the quality of the answer.
"[Detailed Audio Description]
XAudioX
[Question]
XQuestionX

[The Start of Assistant 1s Answer]
XAssistant1X
[The End of Assistant 1s Answer]

[The Start of Assistant 2s Answer]
XAssistant2X
[The End of Assistant 2s Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.

Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. 
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. 
Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. 
The two scores are separated by a space.
"""

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def evaluator(test_dataset, output_data, api_key, base_url, file_path):
    num_sum = 0
    question_id = set()
    tot_score = 0.0
    for output_item in tqdm(output_data, desc="Evaluating"):
        for test_item in test_dataset:
            if test_item['uniq_id'] == output_item['question_id'] and output_item['question_id'] not in question_id:
                question_id.add(output_item['question_id'])
                score = judger(test_item['meta_info'], test_item['question'], test_item['answer_gt'], output_item['response'], api_key, base_url)
                if score < 0:
                    continue
                tot_score += score
                num_sum += 1
                save_detail(test_item['question'], output_item["prompt_text"], test_item['answer_gt'], output_item["response"], score, file_path)
                break

    return tot_score, num_sum

def judger(meta_info, question, answer_gt, response, api_key, base_url):
    for _ in range(5):
        def get_response(prompt):
            data = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

        prompt = gpt_prompt.replace("XAudioX", meta_info).replace("XQuestionX", question).replace("XAssistant1X", answer_gt).replace("XAssistant2X", response)
        try:
            gpt_score = get_response(prompt).strip().replace('\n','')
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            continue
            
        scores = gpt_score.split(' ')
        try:
            score = int(scores[1])
        except (IndexError, ValueError):
            continue
        if score <= 10 and score >= 1:
            return score

    return -1

def main():
    cache_path = ".cache"
    assert os.path.exists(cache_path), ".cache folder not found. ds_infer failed?"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, _ = read_eval_cfgs('air-bench', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs = dict_to_namedtuple(dict_configs)

    raw_outputs = {}
    uuid_path = os.path.join(cache_path, dict_configs.default.eval_cfgs.uuid)
    assert os.path.exists(uuid_path), "uuid_path not found. ds_infer failed?"
    task_dirs = [(task, os.path.join(uuid_path, task)) for task in os.listdir(uuid_path) if os.path.isdir(os.path.join(uuid_path, task))]
    for task, task_dir in task_dirs:
        task_files = os.listdir(task_dir)
        InferenceOutputs = []
        for file in task_files:
            if file.endswith(".pkl"):
                file_path = os.path.join(task_dir, file)
                with open(file_path, 'rb') as f:
                    InferenceOutputs.extend(pickle.load(f))
        raw_outputs[task] = InferenceOutputs

    data_cfgs = dict_configs.default.data_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    
    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    logger = EvalLogger('Align-Anything-Evaluation', dict_configs.default.eval_cfgs.output_dir)
    logger.log_dir = eval_configs.output_dir

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)
    
    task_ids = ['speech', 'sound', 'music', 'mixed-audio']
    total_num_dict = {'speech':0, 'sound':0, 'music':0, 'mixed-audio':0}
    score_sum_dict = {'speech':0, 'sound':0, 'music':0, 'mixed-audio':0}

    for task_name, _ in raw_outputs.items():
        test_data = load_dataset(data_cfgs.task_dir, split='train', data_files='Chat/Chat_meta.json')
        file_path = f"{uuid_path}/{task_name}.json"
        num_score, num_sum = evaluator(test_data, raw_outputs[task_name], api_key, base_url, file_path)
        
        if task_name == 'speech_QA' or task_name == 'speech_dialogue_QA':
            task_id = 'speech'
        elif task_name == 'music_QA' or task_name == 'music_generation_analysis_QA':
            task_id = 'music'
        elif task_name == 'sound_QA' or task_name == 'sound_generation_QA':
            task_id = 'sound'
        elif task_name == 'speech_and_sound_QA' or task_name == 'speech_and_music_QA':
            task_id = 'mixed-audio'
            
        total_num_dict[task_id] += num_sum
        score_sum_dict[task_id] += num_score
        
    for task_id in task_ids:
        
        output_dict = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'average_score': [score_sum_dict[task_id] / total_num_dict[task_id] if total_num_dict[task_id] > 0  else 0],
            'valid_num': [total_num_dict[task_id]]
        }
        logger.print_table(title=f'AIRBench/{task_id} Benchmark', data=output_dict)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task_id}")
        logger.log('info', f"model_id: {output_dict['model_id'][0]}")
        logger.log('info', f"average_score: {output_dict['average_score'][0]}/10")
        logger.log('info', f"valid_num: {output_dict['valid_num'][0]}")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__=="__main__":
    main()