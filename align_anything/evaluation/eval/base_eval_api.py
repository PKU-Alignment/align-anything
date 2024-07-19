from typing import List
from abc import abstractmethod
from align_anything.evaluation.outputs import Arena_input, EvalOutput
# from eval.gpt_evaluation.utils import batch_request_openai


class BaseEval_API:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError


'''
reward model : 
'''

class GPT_eval(BaseEval_API):
    def __init__(self,
                    system_prompt: str,
                    model: str,
                    api_key: str,
                    base_url: str,
                    cache_dir: str,
                    num_workers: int,
                    **kwargs):
        self.system_prompt = system_prompt
        self.model = model
        self.kwargs = kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    @staticmethod
    def batch_request_openai(
        type: str,
        inputs: List,
        openai_api_keys: str,
        openai_base_url: str,
        openai_models: List[str],
        num_workers: int,
        cache_dir: str,
    ) -> List:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_keys, base_url=openai_base_url)
        outputs = []
        for input in inputs:
            response = client.chat.completions.create(
                model=openai_models,
                messages=input,
                max_tokens=1024,
                temperature=0.7,
                top_p=1
            )
            outputs.append(response.choices[0].message)
        return outputs
            

    def evaluate(self, inputs : Arena_input | List[Arena_input]) -> List[EvalOutput]:
        if not isinstance(inputs, list):
            inputs = [inputs,]
        print(inputs)
        processed_inputs = []
        for input in inputs:
            print(input)
            prompt = "[CONTEXT] " + input.prompt + "\n\n" + "[RESPONSE1] " + input.response1 + "\n\n" + "[RESPONSE2] " + input.response2
            gpt_input = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            processed_inputs.append(gpt_input)
        responses = GPT_eval.batch_request_openai(
            type="Arena",
            inputs=processed_inputs,
            openai_api_keys=self.api_key,
            openai_base_url=self.base_url,
            openai_models=self.model,
            num_workers=self.num_workers,
            cache_dir=self.cache_dir,
        )
        return [EvalOutput(evalEngine="gpt_evaluation", input=input, raw_output=response) for input, response in zip(inputs, responses)]
        