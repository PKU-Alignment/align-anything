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
import av
import argparse
import numpy as np
from typing import List, Dict
from transformers import AutoProcessor
from datasets import load_dataset
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs, read_video_pyav, process_vision
from align_anything.evaluation.data_type import InferenceInput
from align_anything.evaluation.eval_logger import EvalLogger
from transformers import LlavaNextVideoProcessor
from collections import defaultdict

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

    def get_category_datasets(self):
        dataset = load_dataset(self.task_dir, 'videomme')[self.split]
        category_datasets = defaultdict(list)
        for i in tqdm(range(len(dataset)), desc='Dataset classification'):
            category = dataset[i]['duration']
            if category in self.task_names:
                category_datasets[category].append(dataset[i])
        return category_datasets
    
    def load_dataset(self, category_datasets):
        processed_inputs = {}
        for task, dataset in category_datasets.items():
            prompts = self.preprocess(dataset)
            processed_inputs[task] = []
            for prompt, i in zip(prompts, range(len(dataset))):
                video_url, question_id = dataset[i]['url'], dataset[i]['question_id']
                video_name = video_url.split('watch?v=')[-1]
                processed_input = InferenceInput(text=prompt, video_name=video_name, video_url=video_url)
                processed_input.question_id = question_id
                processed_inputs[task].append(processed_input)
        return processed_inputs

    def preprocess(self, data):
        return self.build_prompt(data)

class VideoMMEGeneratorVLLM(BaseInferencer_vllm):
    def init_model(self) -> None:
        self.BUILD_PROMPT = {
            "qwenVL": self.build_prompt_qwenvl,
            "llavaNextVideo": self.build_prompt_llavaNextVideo,
        }

        self.samplingparams = SamplingParams(
            top_p=self.sp_top_p,
            temperature=self.sp_temperature,
            max_tokens=self.sp_max_tokens,
        )

        model_name_lower = self.model_name_or_path.lower()
        if 'qwen' in model_name_lower and 'vl' in model_name_lower:
            self.model_flag = 'qwenVL'
            self.model = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=4,
                rope_scaling={
                    "type": "mrope",
                    "mrope_section": [16, 24, 24],
                },
                gpu_memory_utilization=0.5,
                max_num_seqs=self.llm_max_num_seqs,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        elif 'llava-next-video' in model_name_lower:
            self.model_flag = 'llavaNextVideo'
            self.model = LLM(
                model=self.model_name_or_path,
                tokenizer=self.model_name_or_path,
                tokenizer_mode=self.llm_tokenizer_mode,
                trust_remote_code=self.llm_trust_remote_code,
                tensor_parallel_size=self.llm_tensor_parallel_size,
                gpu_memory_utilization=0.5,
                max_num_seqs = self.llm_max_num_seqs
            )
            self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_name_or_path)
        else:
            raise ValueError(f"Model '{self.model_name_or_path}' is not supported or unknown. Supported models are: chameleon, stable-diffusion, flux, sdxl.")

    def build_prompt_qwenvl(self, llm_inputs, prompt, video_path, task):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 2.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        video_inputs = process_vision(messages, 8)

        mm_data = {
            "video": video_inputs
        }

        llm_inputs.append({
            "prompt": text,
            "multi_modal_data": mm_data,
        }) 

    def build_prompt_llavaNextVideo(self, llm_inputs, prompt, video_path, task):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 4).astype(int)
        video_inputs = read_video_pyav(container, indices)

        mm_data = {
            "video": video_inputs
        }

        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })

    def eval(self, data:Dict[str, List[InferenceInput]], video_dir):
        task2details = defaultdict(list)

        for task, inputs in data.items():
            llm_inputs = []
            for input in tqdm(inputs, desc='Prompts building'):
                video_path = os.path.join(video_dir, f'{input.video_name}.mp4')
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                self.BUILD_PROMPT[self.model_flag](llm_inputs, input.text, video_path, task)
            outputs = self.model.generate(llm_inputs, sampling_params=self.samplingparams)
            for output, input in zip(outputs, inputs):
                generated_text = output.outputs[0].text
                video_path = os.path.join(video_dir, f'{input.video_name}.mp4')
                task2details[task].append({
                    'question_id': input.question_id,
                    'prompt': input.text,
                    'video_path': video_path,
                    'response': generated_text
                })

        return task2details

def evaluator(test_dataset, output_data, file_path):
    num_match = 0
    num_sum = 0
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for output in output_data:
            if test_item['question_id'] == output['question_id']:
                num_sum += 1
                true_of_false = judger(test_item['answer'], output['response'])
                if true_of_false:
                    num_match += 1
                choices = '\n' + '\n'.join([f'({label}) {test_item["options"][ord(label) - 65]}' for label in ["A", "B", "C", "D"]])
                save_detail(test_item['question'] + '\n video_path: ' + output['video_path'], choices, test_item['answer'], output['response'], true_of_false, file_path)

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
    
    dict_configs, infer_configs = read_eval_cfgs('videomme', 'vLLM')
    
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
    dataloader = VideoMMEDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 or dataloader.cot), "Few-shot or chain-of-thought cannot be used for this benchmark."
    dataset = dataloader.get_category_datasets()
    test_data = dataloader.load_dataset(dataset)
    eval_module = VideoMMEGeneratorVLLM(model_config, infer_configs)
    video_dir = f"./Video-MME/data"
    raw_outputs_dir = os.path.join(eval_configs.output_dir, f"raw_outputs_{re.sub(r'/', '_', model_config.model_name_or_path)}.pkl")
    if os.path.exists(raw_outputs_dir):
        raw_outputs = load_raw_outputs(raw_outputs_dir)
    else:
        raw_outputs = eval_module.eval(test_data, video_dir)
        save_raw_outputs(raw_outputs, raw_outputs_dir)

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    tot_num_match, tot_num_sum = 0.0, 0
    for task, test_data in dataset.items():
        file_path = f"{uuid_path}/{task}.json"
        num_match, num_sum = evaluator(test_data, raw_outputs[task], file_path)
        tot_num_match += num_match
        tot_num_sum += num_sum

        eval_results = {
                'model_id': [dict_configs.default.model_cfgs.model_id],
                'num_fewshot': [eval_configs.n_shot],
                'chain_of_thought': [eval_configs.cot],
                'num_match': [num_match],
                'num_sum': [num_sum],
                'accuracy': [num_match*100 / num_sum]
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

    eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'tot_num_match': [tot_num_match],
            'tot_num_sum': [tot_num_sum],
            'tot_accuracy': [tot_num_match*100 / tot_num_sum]
            }
    logger.print_table(title=f'VideoMME Benchmark', data=eval_results)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {eval_results['model_id'][0]},")
    logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
    logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
    logger.log('info', f"tot_num_match: {eval_results['tot_num_match'][0]},")
    logger.log('info', f"tot_num_sum: {eval_results['tot_num_sum'][0]},")
    logger.log('info', f"tot_accuracy: {eval_results['tot_accuracy'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
