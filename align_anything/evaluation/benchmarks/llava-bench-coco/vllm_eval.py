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

import os
import argparse
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.eval.base_eval import API_Single_Eval
from align_anything.evaluation.eval_logger import EvalLogger
import json
from datasets import load_dataset

class llavacocoDataLoader(BaseDataLoader):

    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
                self.data_cfgs.task
            ]
            return task_names

    def build_example_prompt(self, data, with_answer=True):
        return f"{data['question']}"

    def build_prompt(self, data):
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input="\n" + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
        
        return question
    
    def load_dataset(self):
        processed_inputs = {}
        gpt_data = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            prompts = self.build_prompt(dataset)
            images = dataset['image']

            processed_inputs[task] = [InferenceInput(text=prompt,token_ids=None, image_file=image) for prompt, image in zip(prompts, images)]
            gpt_data[task] = dataset.to_dict()
        return processed_inputs, gpt_data
    
class llavacocoGeneratorVLLM(BaseInferencer_vllm):

    def _generation(self, inputs: List[InferenceInput])-> List[InferenceOutput]:
        assert isinstance(inputs, list)
        outputs = self.model.generate([{"prompt": input.text, "multi_modal_data": {"image": input.image_file},} for input in inputs], sampling_params=self.samplingparams)

        InferenceOutputs = [
            InferenceOutput.from_vllm_output(vllm_output=output, store_raw=True)
                for output in outputs
        ]
        return InferenceOutputs

    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)
        return task2details

def fill_prompt_template(prompt_template, **kwargs):
    return prompt_template.format(**kwargs)

def get_score(response: str):
    try:
        score_pair = response.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])], response[response.find('\n'):]
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None
        

def evaluator(data: dict, task: str, api_key, base_url, file_path, eval_configs= None):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    rule_path = current_dir + "/rule.jsonl"
    rule_dict = json.load(open(os.path.expanduser(rule_path), 'r'))

    system_prompts = []
    user_prompts = []
    for i in range(len(data['question_id'])):
        rule = rule_dict[data['category'][i]]
        prompt = rule['prompt']
        role = rule['role']
        content = (f"[Context]\n{data['caption'][i]}\n\n"
                   f"[Question]\n{data['question'][i]}\n\n"
                   f"[{role} 1]\n{data['responses'][i]}\n\n[End of {role} 1]\n\n"
                   f"[{role} 2]\n{data['answer'][i]}\n\n[End of {role} 2]\n\n"
                   f"[System]\n{prompt}\n\n")
        system_prompts.append('You are a helpful and precise assistant for checking the quality of the answer.')
        user_prompts.append(content)

    judger = API_Single_Eval(model = eval_configs.judge_model, num_workers = 20, temperature = 0, template_function= None,
                      api_key=api_key, base_url=base_url)
    
    results = judger.evaluate(system_prompts, user_prompts)

    eval_case = []
    for id, system_prompt, user_prompt, result in zip(range(len(data['question_id'])), system_prompts, user_prompts, results):
        output = result.raw_output.choices[0].message.content
        score, comment = get_score(output)
        time = 0
        while score is None:
            if time >=10:
                score = [0,0]
                break
            multi_results=[]
            multi_results = judger.evaluate(system_prompts=[system_prompt],user_prompts=[user_prompt])
            output =multi_results[0].raw_output.choices[0].message.content
            score, comment = get_score(output)
            time+=1
        
        eval_case.append({
            'question_id' : data['question_id'][id],
            'category': data['category'][id],
            'score': score,
            'system_prompt': system_prompt,
            'user_prompt' : user_prompt,
            'response': output,
            'comment': comment
        })       
        save_detail(data['question'][id], '', data['answer'][id], data['responses'][id], score[0] >= score[1], file_path, output)
    
    return eval_case

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')
    dict_configs, infer_configs = read_eval_cfgs('llava-bench-coco', 'vllm')

    try:
        assert dict_configs or infer_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        logger.log('error', "Config file is not exist or incomplete.")
        exit()

    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))

    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    logger.log_dir = eval_configs.output_dir

    dataloader = llavacocoDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    test_data, gpt_data = dataloader.load_dataset()
    eval_module = llavacocoGeneratorVLLM(model_config, infer_configs)
    outputs = eval_module.eval(test_data, eval_configs)
    for task, _ in gpt_data.items():
        gpt_data[task]['responses'] = [output.response[0] for output in outputs[task]]

    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    merged_list = []
    for task, _ in gpt_data.items():
        file_path = f"{uuid_path}/{task}.json"
        output = evaluator(gpt_data[task], task, api_key, base_url, file_path, eval_configs)
        merged_list = merged_list + output

    total_score1 = 0
    total_score2 = 0
    total_count = 0
    for item in merged_list:
        total_score1+=item['score'][0]
        total_score2+=item['score'][1]
        total_count+=1
    total_score = total_score1 / total_score2

    total_average = round(float(total_score) * 100,1)

    eval_results = {
            'total_average': [float(total_average)],
            'total_question': [total_count]
            }
    logger.print_table(title=f'llava-bench-coco/{task} Benchmark', data=eval_results)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"total_average: {eval_results['total_average'][0]},")
    logger.log('info', f"total_question: {eval_results['total_question'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
