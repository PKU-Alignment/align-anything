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
from transformers import AutoProcessor
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs, read_video_pyav, process_vision
from align_anything.evaluation.eval_logger import EvalLogger
from transformers import LlavaNextVideoProcessor
from collections import defaultdict

data_list = {
    "action_sequence": ("./data/video/star/action_sequence/", 6),
    "action_prediction": ("./data/video/star/action_prediction/", 6),
    "action_antonym": ("./data/video/ssv2_video/", 16),
    "fine_grained_action": ("./data/video/Moments_in_Time_Raw/videos/", 16),
    "unexpected_action": ("./data/video/FunQA_test/test/", 6),
    "object_existence": ("./data/video/clevrer/video_validation/", 6),
    "object_interaction": ("./data/video/star/object_interaction/", 16),
    "object_shuffle": ("./data/video/perception/videos/", 16),
    "moving_direction": ("./data/video/clevrer/video_validation/", 16),
    "action_localization": ("./data/video/sta/sta_video/", 6), 
    "scene_transition": ("./data/video/scene_qa/video/", 16),
    "action_count": ("./data/video/perception/videos/", 16),
    "moving_count": ("./data/video/clevrer/video_validation/", 16),
    "moving_attribute": ("./data/video/clevrer/video_validation/", 6),
    "state_change": ("./data/video/perception/videos/", 6),
    "fine_grained_pose": ("./data/video/nturgbd/", 6),
    "character_order": ("./data/video/perception/videos/", 16),
    "egocentric_navigation": ("./data/video/vlnqa/", 6),
    "episodic_reasoning": ("./data/video/tvqa/frames_fps3_hq/", 16),
    "counterfactual_inference": ("./data/video/clevrer/video_validation/", 6),
}

class MVBenchDataLoader(BaseDataLoader):
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

    def get_qa(self, data):
        question = f"Question: {data['question']} Please choose the correct answer from the following options:\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer

    def load_dataset(self) -> DatasetDict:
        processed_inputs = defaultdict(list)
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            task_dir = data_list[task][0]
            for data in dataset:
                question, answer = self.get_qa(data)
                video_path = os.path.join(task_dir, data['video'])
                processed_inputs[task].append({
                    'question': question,
                    'video_path': video_path,
                    'answer': answer
                })
        return processed_inputs

class MVBenchGeneratorVLLM(BaseInferencer_vllm):
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
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        video_inputs = process_vision(messages, data_list[task][1])

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
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        video_inputs = read_video_pyav(container, indices)

        mm_data = {
            "video": video_inputs
        }

        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })

    def eval(self, processed_inputs):
        raw_outputs = defaultdict(list)
        for task, data in processed_inputs.items():
            llm_inputs = []
            for input in tqdm(data, desc='Prompts building'):
                self.BUILD_PROMPT[self.model_flag](llm_inputs, input['question'], input['video_path'], task)
            outputs = self.model.generate(llm_inputs, sampling_params=self.samplingparams)
            for output, input in zip(outputs, data):
                generated_text = output.outputs[0].text
                raw_outputs[task].append({
                    'question': input['question'],
                    'video_path': input['video_path'],
                    'answer': input['answer'],
                    'response': generated_text
                })
        return raw_outputs
    
    def evaluator(self, data, file_path):
        num_match = 0
        num_sum = 0
        for output in tqdm(data, desc='Evaluating'):
            num_sum += 1
            true_or_false = judger(output['question'], output['answer'], output['response'])
            if true_or_false:
                num_match += 1
            save_detail(output['question'], output['question'], output['answer'], output['response'], true_or_false, file_path)
        
        return num_match, num_sum

def judger(question, correct_answer, response):
    match = re.search(r'(?<![a-zA-Z])[A-Z](?![a-zA-Z])', response)
    key = f'({correct_answer})'
    choices = extract_choices(question)
    text_answer = ''
    if key in choices:
        text_answer = choices[key]

    if match and correct_answer == match.group():
        return True
    if text_answer.lower() in response.lower():
        return True
    return False

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, infer_configs = read_eval_cfgs('mvbench', 'vLLM')
    
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
    eval_configs = dict_configs.default.eval_cfgs
    logger = EvalLogger('Evaluation', log_dir=eval_configs.output_dir)
    dataloader = MVBenchDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 or dataloader.cot), "Few-shot or chain-of-thought cannot be used for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = MVBenchGeneratorVLLM(model_config, infer_configs)
    raw_outputs_dir = os.path.join(eval_configs.output_dir, f"raw_outputs_{re.sub(r'/', '_', model_config.model_name_or_path)}.pkl")
    if os.path.exists(raw_outputs_dir):
        raw_outputs = load_raw_outputs(raw_outputs_dir)
    else:
        raw_outputs = eval_module.eval(test_data)
        save_raw_outputs(raw_outputs, raw_outputs_dir)

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    tot_num_match, tot_num_sum = 0, 0
    for task, outputs in raw_outputs.items():
        file_path = f"{uuid_path}/{task}.json"

        num_match, num_sum = eval_module.evaluator(outputs, file_path)
        tot_num_match += num_match
        tot_num_sum += num_sum

        eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'num_match': [num_match],
            'num_sum': [num_sum],
            'accuracy': [num_match*100 / num_sum],
        }
        logger.print_table(title=f'MVBench/{task} Benchmark', data=eval_results)
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
    logger.print_table(title=f'MVBench Benchmark', data=eval_results)
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