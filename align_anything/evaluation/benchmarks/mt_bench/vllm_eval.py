import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'

from align_anything.evaluation.base_vllm import BaseEvaluatorVLLM
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, requestoutput_to_dict
import json
from datasets import Dataset, DatasetDict
from vllm import LLM, SamplingParams
from align_anything.utils.template_registry import get_template_class
from datasets import load_dataset

class TestBenchmark(BaseEvaluatorVLLM):
    def get_task_names(self):
        task_names = [
            # 'default',
            self.data_cfgs.task
        ]
        return task_names

    def load_dataset(self, task_name):
        # TODO: 区分online数据集和本地数据集
        '''
        filename = os.path.join(self.task_dir)
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        dataset = DatasetDict(
            {
                'test': Dataset.from_list(data),
            }
        )
        '''
        dataset = load_dataset(self.task_dir, task_name)
        return dataset

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None
    '''
    def build_example_prompt(self, data, with_answer=True):
        choices = '\n'.join([f'{label}: {data["choices"][ord(label) - 65]}' for label in self.candidate_labels])
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}\n{choices}\n{answer}"
    '''
    def build_prompt(self, data):
        template = get_template_class(self.chat_template)
        if 'response_1' not in data.keys():
            return [template.system_prompt + template.user_prompt.format(input=item['instruction'][0]) + template.assistant_prompt.format(output="") for item in data]
        else:
            return [template.system_prompt + \
                    template.user_prompt.format(input=item['instruction'][0]) + \
                    template.assistant_prompt.format(output=item['response'][0]) + \
                    template.user_prompt.format(input=item['instruction'][1]) + \
                    template.assistant_prompt.format(output="") \
                    for item in data]
        # return question
    
        '''
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=item['question']) + template.assistant_prompt.format(output="") for item in data]

        return question
        '''
    
    def preproccess(self, data):
        prompts = self.build_prompt(data)
        # inputs = self.model.encode(prompts).to(self.device)
        # answers = [self.get_answer(item) for item in data]

        return {
            "prompt": prompts,
            # "answer": answers,
        }
    
    def eval(self) -> None:
        for name in self.task_names:
            details_turn1, details_turn1_2 = self.eval_task(name, self.split)
            self.update_results(details_turn1, details_turn1_2)
    
    def eval_task(self, task_name: str, split='val') -> Dict[str, Dict[str, Any]]:
        dataset = self.load_dataset(task_name)
        # self.set_fewshot_dataset(dataset)
        inputs_turn1 = self.preproccess(dataset[split])
        details_turn1 = self.eval_instance(inputs_turn1)
        for data_turn2, response_turn1 in zip(dataset[split], details_turn1):
            data_turn2['response'] = response_turn1.outputs[0].text
        inputs_turn2 = self.preproccess(data_turn2)
        details_turn2 = self.eval_instance(inputs_turn2)
        return {task_name + 'turn_1': details_turn1}, {task_name + 'turn_2': details_turn2}
    
    def update_results(self,
                       task2details:Dict[str, Dict[str, Any]]
                    )->None:
        brief_file_path = os.path.join(self.output_dir, self.brief_filename)
        detailed_file_path = os.path.join(self.output_dir, self.detailed_filename)
        
        for task, value in task2details.items():
            output_brief = []
            output_detailed = []
            
            for item in value:
                output_brief.append(requestoutput_to_dict(item, mode='brief'))
                output_detailed.append(requestoutput_to_dict(item, mode='detailed'))
                
            with open(brief_file_path + '_' + task + ".jsonl", 'w', encoding='utf-8') as file:
                for item in output_brief:
                    json_record = json.dumps(item, ensure_ascii=False)
                    file.write(json_record + '\n')

            with open(detailed_file_path + '_' + task + ".jsonl", 'w', encoding='utf-8') as file:
                for item in output_detailed:
                    json_record = json.dumps(item, ensure_ascii=False)
                    file.write(json_record + '\n')

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
    dict_configs, infer_configs = read_eval_cfgs('test_mt_bench')
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    
    eval_module = TestBenchmark(dict_configs, infer_configs)

    eval_module.eval()

if __name__ == '__main__':
    main()
