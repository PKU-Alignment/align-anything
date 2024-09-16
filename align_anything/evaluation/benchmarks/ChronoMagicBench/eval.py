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
from threading import Lock
from align_anything.evaluation.inference.base_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, process_videos
from align_anything.evaluation.eval_logger import EvalLogger
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
from collections import defaultdict
from gpt_judger import build_gpt_prompt, gpt_judger

file_lock = Lock()
class ChronoMagicBenchDataLoader(BaseDataLoader):
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

    def load_dataset(self) -> DatasetDict:
        processed_inputs = defaultdict(list)
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            for data in dataset:
                processed_inputs[task].append({
                    'prompt': data['name'],
                    'video_id': data['videoid']
                })
        return processed_inputs

class ChronoMagicBenchGenerator(BaseInferencer):
    def init_model(self, device) -> None:
        self.pipe = CogVideoXPipeline.from_pretrained(self.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
    def parallel_eval(self, task2details, img_dir, datas, device, position):
        self.init_model(device)

        for task, data in datas.items():
            for input in tqdm(data, desc='Generating', position=position):
                video_id = input['video_id']
                prompt = input['prompt']
                video_path = os.path.join(img_dir, task, f"{video_id}.mp4")
                self.text_to_video_generation(prompt, video_path)
                task2details[task].extend(
                    [{
                        'video_id': video_id,
                        'prompt': prompt,
                        'video_path': video_path
                    }]
                )
        
    def eval(self, data, video_dir):
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        mp.set_start_method('spawn', force=True)
        num_processes = 8
        num_gpus = 8
        
        task2details = {}
        for task, inputs in data.items():
            task_dir = os.path.join(video_dir, task)
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
            p = mp.Process(target=self.parallel_eval, args=(task2details, video_dir, chunks, device, i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        return task2details
    
    def evaluator(self, outputs, task_dir, file_path):
        num_workers = 8
        frames_dir = os.path.join(task_dir, "video_frames_folder_temp")
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        
        video_data = defaultdict()
        for output in tqdm(outputs, desc="Processing videos", unit="video", leave=False):
            video_id = output['video_id']
            prompt = output['prompt']
            video_path = output['video_path']
            frames = process_videos(video_id, video_path, frames_dir)

            if frames is None:
                continue
            
            video_data[video_id] = {
                "prompt": prompt,
                "video_path": video_path,
                "frames": frames
            }

        datas = build_gpt_prompt(video_data, frames_dir)
        progress_bar = tqdm(total=len(video_data), desc="Evaluating")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_idx = {executor.submit(self.save_output, prompt, file_path): vid for vid, prompt in datas.items()}
            for future in as_completed(future_idx):
                progress_bar.update(1)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing video ID {future_idx[future]}: {e}")
        progress_bar.close()

        with open(file_path, 'r') as file:
            data = json.load(file)
        scores = [int(item['score']) for item in data]

        return sum(scores), len(data)
    
    def save_output(self, data, output_file):
        gpt_response = gpt_judger(data['gpt_prompt'], self.api_key, self.base_url)
        split_value = gpt_response.rsplit('\n', 1)
        reason = split_value[0].replace("Brief Reasoning Statement:", "").strip()
        score = split_value[1].replace("\"Score\":", "").replace("{", "").replace("}", "").strip()
        score = extract_number(score)
        with file_lock:
            save_detail(data['prompt'], '', '', data['video_path'], score, output_file, reason)

def extract_number(score):
    match = re.search(r'\d+', score)
    if match:
        return int(match.group())
    else:
        return 0
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, infer_configs = read_eval_cfgs('chronomagicbench', 'vLLM')
    
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
    dataloader = ChronoMagicBenchDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = ChronoMagicBenchGenerator(model_config.model_id, model_config.model_name_or_path, model_config.model_max_length, 42)
    video_dir = os.path.join(eval_configs.output_dir, f"./video/{eval_configs.uuid}")
    raw_outputs = eval_module.eval(test_data, video_dir)
    
    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    eval_module.api_key = api_key
    eval_module.base_url = base_url

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    tot_score, tot_num_sum = 0, 0
    for task, outputs in raw_outputs.items():
        file_path = f"{uuid_path}/{task}.json"
        task_dir = os.path.join(video_dir, task)

        score, num_sum = eval_module.evaluator(outputs, task_dir, file_path)
        tot_score += score
        tot_num_sum += num_sum

        eval_results = {
                'model_id': [dict_configs.default.model_cfgs.model_id],
                'num_fewshot': [eval_configs.n_shot],
                'chain_of_thought': [eval_configs.cot],
                'num_sum': [num_sum],
                'avg_score': [score / num_sum],
                }
        logger.print_table(title=f'ChronoMagicBench/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"avg_score: {eval_results['avg_score'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()