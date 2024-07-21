from typing import List
from abc import abstractmethod
from align_anything.evaluation.outputs import Arena_input, EvalOutput, InferenceOutput, InferenceInput

class BaseInference:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def inference(self, **kwargs):
        raise NotImplementedError

class API_Inference:
    def __init__(self,
                    system_prompt: str,
                    model: str = 'deepseek-chat',
                    num_workers: int = 1,
                    cache_dir : str = None,
                    api_key: str = None,
                    base_url: str = None,
                    **kwargs):
        self.system_prompt = system_prompt
        self.model = model
        self.kwargs = kwargs
        self.api_key = api_key
        self.base_url = base_url
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def inference(self, inputs : str | List[str]) -> List[InferenceOutput]:
        if not isinstance(inputs, List):
            inputs = [inputs]
        raise NotImplementedError

class vllm_Inference(BaseInference):
    def __init__(self, 
                 model_name_or_path: str,
                **kwargs):
        ##这里没有用config，只是一个参考，但还是用config比较好
        self.sp = {}
        self.model_name_or_path = model_name_or_path
        self.sp_n = kwargs.get('n', 1)
        self.sp_top_k = kwargs.get('top_k', 10)
        self.sp_top_p = kwargs.get('top_p', 0.95)
        self.sp_temperature = kwargs.get('temperature', 1)
        self.sp_frequency_penalty = kwargs.get('frequency_penalty', 1.2)
        self.sp_prompt_logprobs = kwargs.get('prompt_logprobs', 1)
        self.sp_logprobs = kwargs.get('logprobs', 1)
        self.max_length = kwargs.get('max_length', 32)

        self.llm_tokenizer_mode = kwargs.get('tokenizer_mode', 'auto')
        self.llm_trust_remote_code = kwargs.get('trust_remote_code', False)
        self.llm_gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.3)

        
            
    def inference(self, inputs : InferenceInput | List[InferenceInput]) -> List[InferenceOutput]:
        if not isinstance(inputs, List):
            inputs = [inputs]
        inputs = [input.text for input in inputs] #这里的input是InferenceInput类
        num_gpu = 1
        from vllm import LLM, SamplingParams
        self.samplingparams = SamplingParams(
            n=self.sp_n,
            top_k=self.sp_top_k,
            top_p=self.sp_top_p,
            temperature=self.sp_temperature,
            max_tokens=self.max_length,
            frequency_penalty=self.sp_frequency_penalty,
            prompt_logprobs=self.sp_prompt_logprobs,
            logprobs=self.sp_logprobs
        )
        self.llm = LLM(
            model=self.model_name_or_path,
            tokenizer_mode=self.llm_tokenizer_mode,
            tensor_parallel_size=num_gpu,
            trust_remote_code=self.llm_trust_remote_code,
            gpu_memory_utilization=self.llm_gpu_memory_utilization
        )
        responses = self.llm.generate(
            prompts=inputs,
            sampling_params=self.samplingparams
        )
        return [InferenceOutput.from_vllm_output(response) for response in responses]

if __name__ == "__main__":
    inferencor = vllm_Inference(model_name_or_path="gpt2") 
    print(inferencor.inference(
        [InferenceInput(text="what is 1+2+3"),
        InferenceInput(text="what is 1+2+3"),
        InferenceInput(text="what is 1+2+3")]
    ))