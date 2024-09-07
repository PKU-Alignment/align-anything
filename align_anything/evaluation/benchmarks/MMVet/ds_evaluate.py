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

gpt_system_prompt = """
Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <image> in the question indicates where an image is. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

Below are examples and a new case for which you need to provide the correctness score:

**Examples:**

| Question | Ground truth | Prediction | Correctness |
| --- | --- | --- | --- |
| What is x in the equation?<image> | -1 <AND> -5 | x = 3 | 0.0 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -1 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -5 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -5 or 5 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -1 or x = -5 | 1.0 |
| Can you explain this meme?<image> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4 |
| Can you explain this meme?<image> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0 |

**New Case to Evaluate:**

| Question | Ground truth | Prediction | Correctness |
| --- | --- | --- | --- |
| {INSERT_PROMPT_HERE} | {INSERT_GROUND_TRUTH_HERE} | {INSERT_PREDICTION_HERE} |   |
"""

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def evaluator(test_dataset, output_data, api_key, base_url, file_path):
    num_sum = 0
    question_id = set()
    tot_score = 0.0
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for output_item in output_data:
            if test_item['question_id'] == output_item['question_id'] and output_item['question_id'] not in question_id:
                question_id.add(output_item['question_id'])
                time.sleep(0.01)
                num_sum += 1
                score = judger(test_item['question'], test_item['answer'].lower(), output_item['response'][0].lower(), api_key, base_url)
                tot_score += score
                save_detail(test_item['question'], output_item["prompt_text"], test_item['answer'].lower(), output_item["response"][0].lower(), score, file_path)

    return tot_score / num_sum, num_sum

def judger(question, correct_answer, response, api_key, base_url):
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
                json=data
            )
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

        prompt = gpt_system_prompt.format(
            INSERT_PROMPT_HERE=question,
            INSERT_GROUND_TRUTH_HERE=correct_answer,
            INSERT_PREDICTION_HERE=response
        )
        Correctness = get_response(prompt)
        score = re.findall(r'\d\.\d', Correctness)
        score = float(score[-1]) if score else 0.0
        if score <= 1.0 and score >= 0.0:
            return score

    return 0.0
    
def main():
    cache_path = ".cache"
    assert os.path.exists(cache_path), ".cache folder not found. ds_infer failed?"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, _ = read_eval_cfgs('mmvet', 'deepspeed')
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

    for task, _ in raw_outputs.items():
        test_data = load_dataset(data_cfgs.task_dir, task)[data_cfgs.split]

        file_path = f"{uuid_path}/{task}.json"
        score, num_sum = evaluator(test_data, raw_outputs[task], api_key, base_url, file_path)
        
        output_dict = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_sum': [num_sum],
            'score': [score],
        }
        logger.print_table(title=f'MMVet/{task} Benchmark', data=output_dict)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {output_dict['model_id'][0]},")
        logger.log('info', f"num_sum: {output_dict['num_sum'][0]},")
        logger.log('info', f"score: {output_dict['score'][0]}/1,")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__=="__main__":
    main()