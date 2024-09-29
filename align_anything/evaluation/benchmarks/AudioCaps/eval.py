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
from align_anything.evaluation.inference.base_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from datasets import DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs
from align_anything.evaluation.eval_logger import EvalLogger
from diffusers import AudioLDM2Pipeline
from collections import defaultdict
import torch.multiprocessing as mp
import csv

class AudioCapsDataLoader(BaseDataLoader):
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
        
    def load_dataset(self, background_dir) -> DatasetDict:
        processed_inputs = defaultdict(list)
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            for data in dataset:
                audiocap_id = data['audiocap_id']
                caption = data['caption']
                audio_path_real = os.path.join(background_dir, f'{audiocap_id}.mp3')
                assert os.path.exists(audio_path_real), f"Audio file {audio_path_real} does not exist."
                processed_inputs[task].append({
                    'audiocap_id': audiocap_id,
                    'caption':  caption,
                    'audio_path_real': audio_path_real
                })
        return processed_inputs

class AudioCapsGenerator(BaseInferencer):
    def init_model(self, device) -> None:
        self.model = AudioLDM2Pipeline.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16).to(device)

    def parallel_eval(self, task2details, audio_dir, data, device, position):
        self.init_model(device)

        for task, inputs in data.items():    
            for input in tqdm(inputs, desc='Generating', position=position):
                audiocap_id = input['audiocap_id']
                caption = input['caption']
                audio_path_real = input['audio_path_real']
                audio_path = os.path.join(audio_dir, task, f"{audiocap_id}.mp3")
                self.text_to_audio_generation(caption, audio_path)
                task2details[task].extend(
                    [{
                        'audiocap_id': audiocap_id,
                        'caption': caption,
                        'audio_path': audio_path,
                        'audio_path_real': audio_path_real
                    }]
                )

    def eval(self, data, audio_dir):
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            
        mp.set_start_method('spawn', force=True)
        num_processes = 8
        num_gpus = 8
        
        task2details = {}
        for task, inputs in data.items():
            task_dir = os.path.join(audio_dir, task)
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            task2details[task] = mp.Manager().list()
            
        processes = []
        for i in range(num_processes):
            device = f"cuda:{i%num_gpus}"
            chunks = {}
            for task, inputs in data.items():
                chunk = inputs[i::num_processes]
                chunks[task] = chunk
            p = mp.Process(target=self.parallel_eval, args=(task2details, audio_dir, chunks, device, i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        return task2details
    
    def evaluator(self, outputs, file_path, score_path, background_dir, eval_dir):
        num_sum = 0
        os.system(f"python fad_score.py --datasetpath '{background_dir}' --logs '{eval_dir}' --name '{score_path}'")
        for output in tqdm(outputs, desc="Result Saving"):
            caption = output['caption']
            audio_path = output['audio_path']
            audio_path_real = output['audio_path_real']
            num_sum += 1
            save_detail(caption, '', audio_path_real, audio_path, '', file_path)
        
        with open(score_path, 'r') as file:
            data = json.load(file)

        return data['fad_score'], num_sum

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, infer_configs = read_eval_cfgs('audiocaps', 'vLLM')
    
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
    dataloader = AudioCapsDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    audio_dir = os.path.join(eval_configs.output_dir, f"audio/{eval_configs.uuid}")
    background_dir = './the_real_audio_folder'
    assert os.path.exists(background_dir), f"Directory {background_dir} does not exist."
    test_data = dataloader.load_dataset(background_dir)
    eval_module = AudioCapsGenerator(model_config.model_id, model_config.model_name_or_path, model_config.model_max_length, 42)
    raw_outputs = eval_module.eval(test_data, audio_dir)

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    for task, outputs in raw_outputs.items():
        file_path = f"{uuid_path}/{task}.json"
        score_path = f"{uuid_path}/fad_score.jsonl"
        eval_dir = os.path.join(audio_dir, task)
        score, num_sum = eval_module.evaluator(outputs, file_path, score_path, background_dir, eval_dir)

        eval_results = {
                'model_id': [dict_configs.default.model_cfgs.model_id],
                'num_fewshot': [eval_configs.n_shot],
                'chain_of_thought': [eval_configs.cot],
                'num_sum': [num_sum],
                'FAD_Score': [score],
                }
        logger.print_table(title=f'AudioCaps/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"FAD_Score: {eval_results['FAD_Score'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
