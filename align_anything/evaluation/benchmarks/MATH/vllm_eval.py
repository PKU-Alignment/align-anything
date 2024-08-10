import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'

from align_anything.evaluation.base_vllm import BaseEvaluatorVLLM
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple
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

    def get_answer(self, data):
        return data['solution']

    def set_fewshot_dataset(self, dataset):
        few_shot_examples = json.load(open("few_shot.json", encoding='utf-8'))['math']['ocp']

        formatted_data = []
        for example in few_shot_examples:
            formatted_data.append({
                'problem': example['problem'],
                'solution': example['solution']
            })

        self.few_shot_data = Dataset.from_dict({
            'problem': [item['problem'] for item in formatted_data],
            'solution': [item['solution'] for item in formatted_data]
        })

    def build_example_prompt(self, data, with_answer=True):
        problem = data['problem']
        solution = f'Solution: {data["solution"]}' if with_answer else 'Solution: '
        return f"{problem}\n{solution}"

    def build_prompt(self, data):
        prompt = f"The following are math questions (with answers).\n\n"
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)
        if len(few_shot_examples) == 0:
            question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
        else:
            few_shots = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['problem']))
            ]
            question = []
            for item in data:
                request = {}
                for key, value in item.items():
                    request[key] = value
                examples = few_shots + [self.build_example_prompt(request, False)]
                question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(examples)) + template.assistant_prompt.format(output=""))
        
        return question
        '''
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=item['question']) + template.assistant_prompt.format(output="") for item in data]

        return question
        '''

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        answers = [self.get_answer(item) for item in data]

        return {
            "prompt": prompts,
            "answer": answers,
        }

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
    dict_configs, infer_configs = read_eval_cfgs('test_math')
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    print(dict_configs)
    
    eval_module = TestBenchmark(dict_configs, infer_configs)

    eval_module.eval()

if __name__ == '__main__':
    main()
 