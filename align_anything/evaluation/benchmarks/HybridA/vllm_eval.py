import argparse
import json
from align_anything.evaluation.inference.vllm_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from datasets import load_dataset
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.vllm_inference import update_results
from align_anything.evaluation.eval_logger import EvalLogger

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

class HybridQAGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data: Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)
        
        output_dir = eval_configs.output_dir
        model_id = self.model_cfgs.model_id
        detailed_filename = f'{model_id}_detailed'
        brief_filename = f'{model_id}_brief'
        update_results(output_dir, brief_filename, detailed_filename, task2details)
        
        return task2details
    
def evaluator(raw_output: List[InferenceOutput], dataloader: HybridQADataLoader, task: str):
    dataset = load_dataset(dataloader.task_dir, task)[dataloader.split]
    correct_answers = []
    responses = []
    true_cases = []
    false_cases = []
    cnt_sum = 0
    cnt_match = 0
    cnt_fail = 0

    for instance in dataset:
        correct_answers.append(
            {
                'question': instance['question'],
                'context': f"Table: {instance['table']}\nText: {instance['text']}",
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        responses.append(
            {
                'prompt': item.prompt,
                'response': item.response[0]
            }
        )
    for correct_answer in correct_answers:
        cnt_sum += 1
        flag_fail = True
        for response in responses:
            if correct_answer['question'] in response['prompt']:
                flag_fail = False
                eval_case = {
                    'question': correct_answer['question'],
                    'context': correct_answer['context'],
                    'correct_answer': correct_answer['answer'],
                    'response': response['response']
                }
                if judge_answer(correct_answer['answer'], response['response']):
                    cnt_match += 1
                    eval_case['result'] = True
                    true_cases.append(eval_case)
                else:
                    eval_case['result'] = False
                    false_cases.append(eval_case)
                break
        if flag_fail:
            cnt_fail += 1

    return cnt_match, cnt_sum, true_cases, false_cases

def judge_answer(correct_answer: str, response: str):
    return correct_answer.strip().lower() == response.strip().lower()

def main():
    parser = argparse.ArgumentParser(description='Evaluation Configuration')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--custom_cfgs', type=str, help='Any additional config settings.')

    args = parser.parse_args()

    cfgs_dict = read_eval_cfgs(args.cfg)
    if args.custom_cfgs:
        custom_cfgs = json.loads(args.custom_cfgs)
        cfgs_dict = update_dict(cfgs_dict, custom_cfgs_to_dict(custom_cfgs))

    cfgs = dict_to_namedtuple(cfgs_dict)
    dataloader = HybridQADataLoader(cfgs)
    inferencer = HybridQAGeneratorVLLM(cfgs)

    data = dataloader.load_data()
    raw_output = inferencer.eval(data, cfgs.eval_cfgs)
    cnt_match, cnt_sum, true_cases, false_cases = evaluator(raw_output, dataloader, cfgs.data_cfgs.task)
    log = EvalLogger(cfgs.eval_cfgs.output_dir, cfgs.model_cfgs.model_id)
    log.log_summary(cnt_match, cnt_sum, true_cases, false_cases)

if __name__ == '__main__':
    main()
