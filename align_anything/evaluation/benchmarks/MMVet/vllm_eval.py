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

import argparse
import json
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.eval_logger import EvalLogger
from tqdm import tqdm
import requests
import re
import os

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
categories_info = {
    "rec": "REC (Recognition: Focus on detecting and analyzing patterns, objects, or faces in the input data. Ensure accurate identification and classification by leveraging visual features and contextual information.)",
    "ocr": "OCR (Optical Character Recognition: Precisely detect and convert text from images into readable characters. Ensure high accuracy in character recognition and that the entire text is extracted without errors.)",
    "know": "KNOW (Knowledge: Provide accurate and relevant answers based on established knowledge. Ensure that responses are factually correct, directly address the question, and are well-supported by relevant information.)",
    "gen": "GEN (Language Generation: Create text that is coherent and contextually relevant to the given prompt. Ensure that generated content is creative, fits the context, and aligns with the prompt's requirements.)",
    "spat": "SPAT (Spatial Awareness: Interpret and reason about spatial relationships and geometric properties. Ensure accurate understanding of spatial information and that answers involving geometry are correctly reasoned and solved.)",
    "math": "MATH (Math: Solve mathematical problems by applying clear, logical steps. Decompose complex problems into simpler components and ensure all calculations are precise and thoroughly explained.)"
}

class MMVetDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['answer']
        
    def build_example_prompt(self, data, with_answer=True):
        cate_info = ''
        categories = data['capability'].split(',')
        for i in range(len(categories)):
            categorie = categories[i]
            info = categories_info[categorie.lower()]
            cate_info += f'Related_fields_{i + 1}: {info}\n'
        return f"{cate_info}\nQuestion: {data['question']}"

    def build_prompt(self, data):
        assert self.num_shot == 0, "MMVet does not support few-shot learning."
        prompt = "The following are problems that may involve one or more of the following fields: recognition, OCR, knowledge, language generation, spatial perception, and mathematical computation.\n\n"
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]

        return question
    
    def preprocess(self, data):
        return self.build_prompt(data)
    
    def load_dataset(self) -> DatasetDict:
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            prompts = self.preprocess(dataset)
            processed_inputs[task] = []
            for prompt, image, question_id in zip(prompts, dataset['image'], dataset['question_id']):
                processed_input = InferenceInput(text=prompt, image_file=image)
                processed_input.question_id = question_id
                processed_inputs[task].append(processed_input)
        return processed_inputs
    
class MMVetGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            raw_output = self.generation(input)
            for item in raw_output:
                item.prompt = re.sub(r"<image>", "", item.prompt)
                item.raw_output.prompt = re.sub(r"<image>", "", item.raw_output.prompt)
            task2details[task] = raw_output
        
        return task2details
    
    def _generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        assert isinstance(inputs, list)
        InferenceOutputs = []
        outputs = self.model.generate([{"prompt": input.text, "multi_modal_data": {"image": input.image_file},} for input in inputs], sampling_params=self.samplingparams)
        InferenceOutputs = [InferenceOutput.from_vllm_output(question_id=input.question_id, vllm_output=output, store_raw=True) for output, input in zip(outputs, inputs)]
        return InferenceOutputs

def evaluator(test_dataset, output_data, gpt_data, gpt_data_file, api_key, base_url, file_path):
    num_sum = 0
    question_id = set()
    tot_score = 0.0
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for output_item in output_data:
            if test_item['question_id'] == output_item.question_id and output_item.question_id not in question_id:
                question_id.add(output_item.question_id)
                num_sum += 1
                gpt_id = test_item['question_id'] + output_item.response[0].lower()
                if gpt_id in gpt_data:
                    score = gpt_data[gpt_id]
                else:
                    score = judger(test_item['question'], test_item['answer'].lower(), output_item.response[0].lower(), api_key, base_url)
                    gpt_data[gpt_id] = score
                tot_score += score
                save_detail(test_item['question'], output_item.prompt, test_item['answer'].lower(), output_item.response[0].lower(), score, file_path)

    with open(gpt_data_file, 'w', encoding='utf-8') as f:
        json.dump(gpt_data, f, ensure_ascii=False, indent=4)
        
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    dict_configs, infer_configs = read_eval_cfgs('mmvet', 'vLLM')

    try:
        assert dict_configs or infer_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        print("Config file is not exist or incomplete.")
        exit()
    
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    data_cfgs = dict_configs.default.data_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    logger = EvalLogger('Evaluation', log_dir=eval_configs.output_dir)
    dataloader = MMVetDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 or dataloader.cot), "Few-shot or chain-of-thought cannot be used for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = MMVetGeneratorVLLM(model_config, infer_configs)
    raw_outputs_dir = os.path.join(eval_configs.output_dir, f"raw_outputs_{re.sub(r'/', '_', model_config.model_name_or_path)}.pkl")
    if os.path.exists(raw_outputs_dir):
        raw_outputs = load_raw_outputs(raw_outputs_dir)
    else:
        raw_outputs = eval_module.eval(test_data, eval_configs)
        save_raw_outputs(raw_outputs, raw_outputs_dir)
   
    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    gpt_data_file = os.path.join(eval_configs.output_dir, f"gpt_data.json")
    gpt_data = {}
    if os.path.exists(gpt_data_file):
        with open(gpt_data_file, 'r', encoding='utf-8') as file:
            gpt_data = json.load(file)

    for task, _ in raw_outputs.items():
        test_data = load_dataset(data_cfgs.task_dir, task)[data_cfgs.split]
        file_path = f"{uuid_path}/{task}.json"
        score, num_sum = evaluator(test_data, raw_outputs[task], gpt_data, gpt_data_file, api_key, base_url, file_path)
        
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

if __name__ == '__main__':
    main()
