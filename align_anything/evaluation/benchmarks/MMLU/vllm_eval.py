import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

from align_anything.evaluation.base_generator_vllm import BaseGeneratorVLLM
from align_anything.evaluation.base_data_loader import BaseDataLoader
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple
from align_anything.utils.template_registry import get_template_class


class MMLUDataLoader(BaseDataLoader):

    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return chr(65 + data['answer'])

    def set_fewshot_dataset(self, dataset):
        return dataset['dev']

    def build_example_prompt(self, data, with_answer=True):
        choices = '\n'.join([f'{label}: {data["choices"][ord(label) - 65]}' for label in self.candidate_labels])
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = f"The following are multiple choice questions (with answers).\n\n"
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)
        if len(few_shot_examples) == 0:
            question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
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
                question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(examples)) + template.assistant_prompt.format(output=""))
        
        return question
    
def main():
    dict_configs, infer_configs = read_eval_cfgs('test_mmlu')
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    
    dataloader = MMLUDataLoader(dict_configs)
    test_data = dataloader.load_dataset()
    eval_module = BaseGeneratorVLLM(dict_configs, infer_configs)
    eval_module.eval(test_data)

if __name__ == '__main__':
    main()
