from typing import List
from abc import abstractmethod
from align_anything.evaluation.outputs import Arena_input, EvalOutput, InferenceOutput
from utils import batch_request_openai

class BaseEval:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError
    
class Reward_Single_eval_deepspeed(BaseEval):
    def __init__(self, 
                 judge_prompt: str,
                 model_name_or_path: str,
                 num_workers: int = 1,
                 cache_dir : str = None,
                **kwargs):
        self.judge_prompt = judge_prompt
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.num_workers = num_workers
        self.cache_dir = cache_dir
            
    def evaluate(self, inputs : InferenceOutput | List[InferenceOutput]) -> List[EvalOutput]:
        raise NotImplementedError

class Reward_Single_eval_vllm(BaseEval):
    def __init__(self, 
                 judge_prompt: str,
                 model_name_or_path: str,
                 num_workers: int = 1,
                 cache_dir : str = None,
                **kwargs):
        self.judge_prompt = judge_prompt
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.num_workers = num_workers
        self.cache_dir = cache_dir        

    def evaluate(self, inputs : InferenceOutput | List[InferenceOutput]) -> List[EvalOutput]:
        raise NotImplementedError


class API_Single_Eval(BaseEval):
    def __init__(self,
                    judge_prompt: str,
                    model: str = 'deepseek-chat',
                    num_workers: int = 1,
                    cache_dir : str = None,
                    api_key: str = None,
                    base_url: str = None,
                    **kwargs):
        self.judge_prompt = judge_prompt
        self.model = model
        self.kwargs = kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.num_workers = num_workers
        self.cache_dir = cache_dir
            
    def evaluate(self, inputs : InferenceOutput | List[InferenceOutput]) -> List[EvalOutput]:
        if not isinstance(inputs, list):
            inputs = [inputs,]
        processed_inputs = []
        for input in inputs:
            prompt = "[CONTEXT] " + input.prompt + "\n\n" + "[RESPONSE] " + input.response
            gpt_input = [
                {'role': 'system', 'content': self.judge_prompt},
                {'role': 'user', 'content': prompt}
            ]
            processed_inputs.append(gpt_input)
        responses = batch_request_openai(
            type="Arena",
            inputs=processed_inputs,
            num_workers=self.num_workers,
            model = self.model,
            cache_dir=self.cache_dir,
            kwargs=self.kwargs
        )
        return [EvalOutput(evalEngine="gpt_evaluation", input=input, raw_output=response) for input, response in zip(inputs, responses)]
        

class API_Pair_Eval(BaseEval):
    def __init__(self,
                    judge_prompt: str,
                    model: str = 'deepseek-chat',
                    num_workers: int = 1,
                    cache_dir : str = None,
                    api_key: str = None,
                    base_url: str = None,
                    **kwargs):
        self.judge_prompt = judge_prompt
        self.model = model
        self.kwargs = kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.num_workers = num_workers
        self.cache_dir = cache_dir
            
    def evaluate(self, inputs : Arena_input | List[Arena_input]) -> List[EvalOutput]:
        if not isinstance(inputs, list):
            inputs = [inputs,]
        processed_inputs = []
        for input in inputs:
            prompt = "[CONTEXT] " + input.prompt + "\n\n" + "[RESPONSE1] " + input.response1 + "\n\n" + "[RESPONSE2] " + input.response2
            gpt_input = [
                {'role': 'system', 'content': self.judge_prompt},
                {'role': 'user', 'content': prompt}
            ]
            processed_inputs.append(gpt_input)
        responses = batch_request_openai(
            type="Arena",
            inputs=processed_inputs,
            num_workers=self.num_workers,
            model = self.model,
            cache_dir=self.cache_dir,
            kwargs=self.kwargs
        )
        return [EvalOutput(evalEngine="gpt_evaluation", input=input, raw_output=response) for input, response in zip(inputs, responses)]
        
if __name__ == "__main__":
    # judger = API_Single_Eval(judge_prompt="judge how good it is, in a [0,10] scores. Response should start with 'SCORE[X]',where X is the socres", model='deepseek-chat', num_workers=2, temperature=0.5)
    # print(judger.evaluate(InferenceOutput(prompt="what is 1+2+3", response="1+2+3=1")))
    
    judger = API_Pair_Eval(judge_prompt="judge which response is better,response should start with '[1]' or '[2]'", model='deepseek-chat', num_workers=2, temperature=0.7)
    print(judger.evaluate(Arena_input(prompt="what is 1+2+3", response1="1+2+3=1", response2="1+2+3=6")))