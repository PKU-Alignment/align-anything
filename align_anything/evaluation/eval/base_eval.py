from typing import List
from abc import abstractmethod
from align_anything.evaluation.outputs import ArenaInput, EvalOutput, InferenceOutput, InferenceInput, SingleInput
from align_anything.evaluation.inference.base_inference import vllm_Inference
from utils import batch_request_openai,filter_out_exception
from openai.types.chat.chat_completion import ChatCompletion

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
            
    def evaluate(self, inputs : SingleInput | List[SingleInput]) -> List[EvalOutput]:
        raise NotImplementedError

class vllm_Eval(BaseEval):
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

    def _evaluate(self, processed_inputs : List[str]) -> List[str]:
        raise NotImplementedError
        inferencor = vllm_Inference(
            model_name_or_path=self.model_name_or_path,
            **self.kwargs
        )
        responses = inferencor.inference(processed_inputs)
        return responses
    
    def evaluate(self, inputs : SingleInput | List[SingleInput]) -> List[EvalOutput]:
        raise NotImplementedError

class Reward_Single_eval_vllm(vllm_Eval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, inputs : SingleInput | List[SingleInput]) -> List[EvalOutput]:
        if not isinstance(inputs, list):
            inputs = [inputs,]
        processed_inputs = []
        for input in inputs:
            text = "CONTEXT: " + input.prompt + "\n\n" + "RESPONSE: " + input.response
            prompt = self.judge_prompt + "\n\n" + text
            # TODO: fix it to template rather than hard code

            processed_inputs.append(prompt)
        responses = self._evaluate(processed_inputs) 
        results = [EvalOutput(evalEngine="vllm_evaluation", input=input, raw_output=response) for input, response in zip(inputs, responses)]
        return filter_out_exception(results)

# this function should not be in this file, move it and fix it in the right place
def template_function_example(input):
    assert isinstance(input, ArenaInput)
    return "test:Human: {prompt}\nAssistant 1: {response1}\nAssistant 2: {response2}".format(prompt=input.prompt, response1=input.response1, response2=input.response2)

class API_Eval(BaseEval):
    def __init__(self,
                    judge_prompt: str,
                    model: str = 'deepseek-chat',
                    num_workers: int = 1,
                    cache_dir : str = None,
                    api_key: str = None,
                    base_url: str = None,
                    template_function = None,
                    **kwargs):
        self.judge_prompt = judge_prompt
        self.model = model
        self.kwargs = kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.template_function = template_function

    def _evaluate(self, processed_inputs : List[List[dict]]) -> List[ChatCompletion | Exception]:
        print(processed_inputs)
        responses = batch_request_openai(
            type="Arena",
            inputs=processed_inputs,
            num_workers=self.num_workers,
            model = self.model,
            cache_dir=self.cache_dir,
            kwargs=self.kwargs
        )
        return responses


class API_Single_Eval(API_Eval):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, inputs : SingleInput | List[SingleInput]) -> List[EvalOutput]:
        if not isinstance(inputs, list):
            inputs = [inputs,]
        processed_inputs = []
        for input in inputs:
            gpt_input = input.build_gpt_input(judge_prompt=self.judge_prompt, template_function=self.template_function)
            processed_inputs.append(gpt_input)
        responses = self._evaluate(processed_inputs)
        results = [EvalOutput(evalEngine="gpt_evaluation", input=input, raw_output=response) for input, response in zip(inputs, responses)]
        return filter_out_exception(results)
        

class API_Pair_Eval(API_Eval):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.template_function is None:
            self.template_function = template_function_example

    def evaluate(self, inputs : ArenaInput | List[ArenaInput]) -> List[EvalOutput]:
        if not isinstance(inputs, list):
            inputs = [inputs,]
        processed_inputs = []
        for input in inputs:
            gpt_input = input.build_gpt_input(judge_prompt=self.judge_prompt, template_function=self.template_function)
            processed_inputs.append(gpt_input)
        responses = self._evaluate(processed_inputs)
        results = [EvalOutput(evalEngine="gpt_evaluation", input=input, raw_output=response) for input, response in zip(inputs, responses)]
        return filter_out_exception(results)
        
if __name__ == "__main__":
    
    import os
    os.environ["OPENAI_API_KEY"] = "sk-xxx"
    # judger = API_Single_Eval(judge_prompt="judge how good it is, in a [0,10] scores. Response should start with 'SCORE[X]',where X is the socres", model='deepseek-chat', num_workers=2, temperature=0.5)
    # print(judger.evaluate(InferenceOutput(prompt="what is 1+2+3", response="1+2+3=1")))
    
    judger = API_Pair_Eval(judge_prompt="judge which response is better,response should start with '[1]' or '[2]'", model='deepseek-chat', num_workers=2, temperature=0.7, template_function=template_function_example)
    print(judger.evaluate(ArenaInput(prompt="what is 1+2+3", response1="1+2+3=1", response2="1+2+3=6")))