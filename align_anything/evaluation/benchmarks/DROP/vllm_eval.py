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
from align_anything.evaluation.inference.vllm_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.vllm_inference import update_results

class DROPDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['answers_spans']

    def build_example_prompt(self, data, with_answer=True):
        return f"{data['passage']}\n{data['question']}"

    def build_prompt(self, data):
        prompt = f"The following are questions (with passages).\n\n"
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
        
        return question

    def preprocess(self, data):
        prompts = self.build_prompt(data[self.split].select(range(1000)))
        token_ids = self.tokenizer(prompts)

        return prompts, token_ids

class DROPGeneratorVLLM(BaseInferencer_vllm):

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
    dict_configs, infer_configs = read_eval_cfgs('test_drop')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    dataloader = DROPDataLoader(dict_configs)
    test_data = dataloader.load_dataset()
    eval_module = DROPGeneratorVLLM(model_config, infer_configs)
    eval_module.eval(test_data, eval_configs)
    
if __name__ == '__main__':
    main()
