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
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader, CustomImageDataset
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.tools import image_crop, inception_score
from align_anything.evaluation.eval_logger import EvalLogger
import torch.multiprocessing as mp
from pytorch_fid import fid_score
import uuid
import os

class MSCOCODataLoader(BaseDataLoader):
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
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            processed_inputs[task] = []
            for data in dataset:
                processed_inputs[task].append({
                    'prompt': data['caption'],
                    'real_image': data['image']
                })
        return processed_inputs

class MSCOCOGenerator(BaseInferencer):
    def parallel_eval(self, img_dir, data, device, position):
        self.init_model(device)

        for task, inputs in data.items():    
            for input in tqdm(inputs, desc='Generating', position=position):
                prompt = input['prompt']
                real_image = input['real_image']
                uid = str(uuid.uuid4())
                image_path = os.path.join(img_dir, task, f"{uid}.jpg")
                real_image_path = os.path.join(img_dir, f'{task}_real', f"{uid}.jpg")
                self.text_to_image_generation(prompt, image_path)
                if os.path.isfile(image_path):
                    real_image.save(real_image_path)
        
    def eval(self, data, img_dir):
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            
        mp.set_start_method('spawn', force=True)
        num_processes = 8
        num_gpus = 8
        
        task2details = {}
        for task, inputs in data.items():
            image_dir = os.path.join(img_dir, task)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            real_image_dir = os.path.join(img_dir, f'{task}_real')
            if not os.path.exists(real_image_dir):
                os.makedirs(real_image_dir)
            task2details[task] = {
                'image_dir': image_dir,
                'real_image_dir': real_image_dir
            }
            
        processes = []
        for i in range(num_processes):
            device = f"cuda:{i%num_gpus}"
            chunks = {}
            for task, inputs in data.items():
                chunk = inputs[i::num_processes]
                chunks[task] = chunk
            p = mp.Process(target=self.parallel_eval, args=(img_dir, chunks, device, i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        return task2details
    
    def evaluator(self, outputs):
        num_sum = 0
        for filename in os.listdir(outputs['image_dir']):
            img_path = os.path.join(outputs['image_dir'], filename)
            if os.path.isfile(img_path):
                num_sum += 1
        
        batch_size = min(num_sum - 1, 32)
        fid_value = fid_score.calculate_fid_given_paths([image_crop(outputs['image_dir']), image_crop(outputs['real_image_dir'])], batch_size=batch_size, device='cuda', dims=2048)
        splits = min(num_sum, 10)
        custom_dataset = CustomImageDataset(outputs['image_dir'])
        IS_score, IS_std = inception_score(custom_dataset, cuda=True, batch_size=batch_size, resize=True, splits=splits)
        
        return fid_value, IS_score, IS_std, num_sum

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, infer_configs = read_eval_cfgs('mscoco', 'vLLM')
    
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
    dataloader = MSCOCODataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = MSCOCOGenerator(model_config.model_id, model_config.model_name_or_path, model_config.model_max_length, 42)
    img_dir = os.path.join(eval_configs.output_dir, f"./images/{eval_configs.uuid}")
    raw_outputs = eval_module.eval(test_data, img_dir)
    
    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    for task, outputs in raw_outputs.items():
        FID_score, IS_score, IS_std, num_sum = eval_module.evaluator(outputs)

        eval_results = {
                'model_id': [dict_configs.default.model_cfgs.model_id],
                'num_fewshot': [eval_configs.n_shot],
                'chain_of_thought': [eval_configs.cot],
                'num_sum': [num_sum],
                'FID_score': [FID_score],
                'IS_score': [IS_score],
                }
        logger.print_table(title=f'MSCOCO/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"FID_score: {eval_results['FID_score'][0]},")
        logger.log('info', f"IS_score: {eval_results['IS_score'][0]} (±{IS_std}),")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()