import os
import pickle
import argparse
import torch.distributed as dist
import json
from align_anything.evaluation.inference.ds_inference import BaseInferencer_deepspeed
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput

class HybridQADataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            return [self.data_cfgs.task]

    def get_answer(self, data):
        return data['answer']

    def build_example_prompt(self, data, with_answer=True, cot=False):
        # Construct the question, context (table + text), and answer (if required)
        context = f"Table: {data['table']}\nText: {data['text']}"
        question = f"Question: {data['question']}"
        answer = f"Answer: {self.get_answer(data)}" if with_answer else "Answer: "
        return f"{context}\n{question}\n{answer}"

    def build_prompt(self, data):
        prompt = f"Answer the following questions using the provided table and text information.\n\n"
        cot_prompt = f" Let's think step by step. "
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)

        if len(few_shot_examples) == 0:
            question = [
                template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") 
                for item in data
            ]
        else:
            examples = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            question = []
            for item in data:
                request = {key: value for key, value in item.items()}
                example_prompts = examples + [self.build_example_prompt(request, False)]
                if self.cot:
                    question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(example_prompts)) + template.assistant_prompt.format(output=cot_prompt))
                else:
                    question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(example_prompts)) + template.assistant_prompt.format(output=""))
        
        return question

class HybridQAGeneratorDS(BaseInferencer_deepspeed):
    def eval(self, data: Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        os.makedirs(".cache", exist_ok=True)
        
        for task, input in data.items():
            task_dir = f".cache/{task}"
            os.makedirs(task_dir, exist_ok=True)
            InferenceOutputs = self.generation(input)
            if dist.is_initialized():
                file_path = f"{task_dir}/outputs_{get_rank()}.pkl"
            else:
                file_path = f"{task_dir}/outputs.pkl"
                
            with open(file_path, 'wb') as f:
                pickle.dump(InferenceOutputs, f, protocol=4)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Evaluation Configuration')
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    dict_configs, infer_configs = read_eval_cfgs('hybrida', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs = dict_to_namedtuple(dict_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    dataloader = HybridQADataLoader(dict_configs)
    inferencer = HybridQAGeneratorDS(model_config, infer_configs)
    data = dataloader.load_dataset()
    inferencer.eval(data, eval_configs)

if __name__ == '__main__':
    main()
