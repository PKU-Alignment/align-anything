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
import json
import argparse
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, download_video
from align_anything.evaluation.data_type import InferenceInput
from align_anything.evaluation.eval_logger import EvalLogger
from align_anything.models.VideoLLaMA2 import model_init, mm_infer
    
class VideoMMEDataLoader(BaseDataLoader):
    def init_tokenizer(self):
        pass

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

    def set_fewshot_dataset(self, dataset, task): 
        if self.cot:
            with open('../cot_fewshot/VideoMME/' + task + '.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return None
    
    def build_example_prompt(self, data, with_answer=True, cot=False):
        choices = '\n'.join([f'({label}) {data["options"][ord(label) - 65]}' for label in self.candidate_labels])
        return f"{data['sub_category']}. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. {data['question']}\n{choices}\nThe best answer is:"

    def build_prompt(self, data):
        prompt = ""
        cot_prompt = f" Let's think step by step. "
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        if len(few_shot_examples) == 0:
            question = [prompt + self.build_example_prompt(item, False) for item in data]
        else:
            few_shots = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            question = []
            for item in data:
                request = {}
                for key, value in item.items():
                    request[key] = value
                examples = few_shots + [self.build_example_prompt(request, False)]
                if self.cot:
                    question.append(prompt + '\n\n'.join(examples) + cot_prompt)
                else:
                    question.append(prompt + '\n\n'.join(examples))
        
        return question
    
    def load_dataset(self) -> DatasetDict:
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            self.few_shot_data = self.set_fewshot_dataset(dataset, task)
            prompts = self.preprocess(dataset)
            processed_inputs[task] = []
            for prompt, video_name, video_url, question_id in zip(prompts, dataset['videoID'], dataset['url'], dataset['question_id']):
                processed_input = InferenceInput(text=prompt, video_name=video_name, video_url=video_url)
                processed_input.question_id = question_id
                processed_inputs[task].append(processed_input)
        return processed_inputs

    def preprocess(self, data):
        return self.build_prompt(data)

class VideoMMEGeneratorVLLM(BaseInferencer_vllm):
    def init_model(self) -> None:
        self.model, self.processor, self.tokenizer = model_init(self.model_name_or_path)

    def eval(self, data:Dict[str, List[InferenceInput]], video_dir):
        task2details = {}

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        for task, inputs in data.items():
            task_dir = os.path.join(video_dir, task)
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            question_ids = []
            prompts = []
            video_paths = []
            responses = []
            for input in tqdm(inputs, desc='Generating'):
                video_path = os.path.join(task_dir, f'{input.video_name}.mp4')
                if not os.path.exists(video_path):
                    downloaded = download_video(input.video_url, video_path)
                    if not downloaded:
                        continue
                prompt = input.text
                response = mm_infer(self.processor(video_path), prompt, model=self.model, tokenizer=self.tokenizer)

                question_ids.append(input.question_id)
                prompts.append(prompt)
                video_paths.append(video_path)
                responses.append(response)

            task2details[task] = {
                'question_ids': question_ids,
                'prompts': prompts,
                'video_paths': video_paths,
                'responses': responses
            }

        return task2details

def evaluator(test_dataset, output_data, file_path):
    num_match = 0
    num_sum = 0
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for question_id, video_path, response in zip(output_data['question_ids'], output_data['video_paths'], output_data['responses']):
            if test_item['question_id'] == question_id:
                num_sum += 1
                true_of_false = judger(test_item['answer'], response)
                if true_of_false:
                    num_match += 1
                choices = '\n' + '\n'.join([f'({label}) {test_item["options"][ord(label) - 65]}' for label in ["A", "B", "C", "D"]])
                save_detail(test_item['question'] + '\n' + video_path, choices, test_item['answer'], response, true_of_false, file_path)

    return num_match, num_sum

def judger(correct_answer, response):
    match = re.search(r'(?<![a-zA-Z])[A-Z](?![a-zA-Z])', response)
    if match:
        return correct_answer == match.group()
    return False
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')
    
    dict_configs, infer_configs = read_eval_cfgs('videomme', 'vLLM')
    
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
    data_cfgs = dict_configs.default.data_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    logger.log_dir = eval_configs.output_dir
    dataloader = VideoMMEDataLoader(dict_configs)
    
    test_data = dataloader.load_dataset()
    eval_module = VideoMMEGeneratorVLLM(model_config, infer_configs)
    video_dir = f"./VideoMME"
    raw_outputs = eval_module.eval(test_data, video_dir)

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    for task, _ in raw_outputs.items():
        test_data = load_dataset(data_cfgs.task_dir, task)[data_cfgs.split]
        file_path = f"{uuid_path}/{task}.json"
        num_match, num_sum = evaluator(test_data, raw_outputs[task], file_path)

        eval_results = {
                'model_id': [dict_configs.default.model_cfgs.model_id],
                'num_fewshot': [eval_configs.n_shot],
                'chain_of_thought': [eval_configs.cot],
                'num_match': [num_match],
                'num_sum': [num_sum],
                'accuracy': [num_match / num_sum]
                }
        logger.print_table(title=f'VideoMME/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_match: {eval_results['num_match'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"accuracy: {eval_results['accuracy'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
