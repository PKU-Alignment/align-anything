import os

import argparse
from align_anything.evaluation.inference.base_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import Union, List, Dict, Any, Tuple
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from datasets import load_dataset, DatasetDict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.base_inference import update_results

class MTBenchDataLoader(BaseDataLoader):
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
            dataset = load_dataset(self.task_dir, task)
            prompts, token_ids = self.preprocess(dataset)
            processed_inputs[task] = [InferenceInput(text=prompt, token_ids=token_id) for prompt, token_id in zip(prompts, token_ids['input_ids'])]
        return processed_inputs
    
    def build_prompt(self, data, responses_r1=None):
        template = get_template_class(self.chat_template)
        if responses_r1:
            return [template.system_prompt + \
                    template.user_prompt.format(input=item['instruction'][0]) + \
                    template.assistant_prompt.format(output=response_r1) + \
                    template.user_prompt.format(input=item['instruction'][1]) + \
                    template.assistant_prompt.format(output="") \
                    for response_r1, item in zip(responses_r1, data)]
        else:
            return [template.system_prompt + template.user_prompt.format(input=item['instruction'][0]) + template.assistant_prompt.format(output="") for item in data]
    
    def load_dataset_round2(self, outputs_r1: Dict[str, List[InferenceOutput]]):
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)
            responses_r1 = [output_r1.response[0] for output_r1 in outputs_r1[task]]
            prompts, token_ids = self.preprocess(data=dataset, responses_r1=responses_r1)
            processed_inputs[task] = [InferenceInput(text=prompt, token_ids=token_id) for prompt, token_id in zip(prompts, token_ids['input_ids'])]
        return processed_inputs
    
    def preprocess(self, data, responses_r1=None):
        prompts = self.build_prompt(data[self.split], responses_r1)
        
        token_ids = self.tokenizer(prompts)

        return prompts, token_ids

class MTBenchGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)
        
        output_dir = eval_configs.output_dir
        brief_filename = eval_configs.brief_filename
        model_id = self.model_cfgs.model_id
        detailed_filename = f'{model_id}_detailed'
        brief_filename = f'{model_id}_brief'
        update_results(output_dir, brief_filename, detailed_filename,task2details)
        
        return task2details

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    dict_configs, infer_configs = read_eval_cfgs('test_mt_bench')
    for k, v in unparsed_args.items():
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))

    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs

    dataloader = MTBenchDataLoader(dict_configs)
    test_data_round1 = dataloader.load_dataset()
    eval_module = MTBenchGeneratorVLLM(model_config, infer_configs)
    output_data_round1 = eval_module.eval(test_data_round1, eval_configs)

    test_data_round2 = dataloader.load_dataset_round2(output_data_round1)
    output_data_round2 = eval_module.eval(test_data_round2, eval_configs)

if __name__ == '__main__':
    main()
