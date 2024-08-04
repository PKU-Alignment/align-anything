from typing import List
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from align_anything.evaluation.outputs import EvalOutput
import json
import os
import logging
from dataclasses import dataclass
import time
import openai
from hashlib import sha256
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage

def create_openai_chat_completion(client : OpenAI,messages: List[dict], parameters: dict, key: str, retry_steps = 0) -> tuple[ChatCompletion | Exception, str]:
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=parameters['model'],
            max_tokens=parameters.get('max_tokens', 2048),
            temperature=parameters.get('temperature', 1),
            top_p=parameters.get('top_p', 1),
            frequency_penalty=parameters.get('frequency_penalty', 0),
            presence_penalty=parameters.get('presence_penalty', 0),
            stop=parameters.get('stop', None),
            stream=parameters.get('stream', False),
            logprobs=parameters.get('logprobs', False),
            top_logprobs=parameters.get('top_logprobs', None),
        )
        return (response, messages, key, retry_steps)
    except Exception as e:
        return (e, messages, key, retry_steps)


def batch_request_openai(
    type: str,
    inputs: List,
    model: str,
    num_workers: int = 1 , 
    cache_dir: str = "/home/yangyaodong/projects/wangkaile/align-anything-kaile/align_anything/evaluation/benchmarks/mt_bench/cache",
    openai_api_keys: str = None,
    openai_base_url: str = None,
    kwargs: dict = {},
    MAX_RETRY_STEPS = 5
) -> List:
    
    def generate_key(input):
        key = f"{type}_{model}_{input}_{kwargs}_{openai_api_keys}_{openai_base_url}"
        return sha256(str(key).encode()).hexdigest()
    
    def load_cache(key: str):
        cache_path = os.path.join(cache_dir, f"{key}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return eval(f.read())
        return None
    def write_cache(key: str, response):
        cache_path = os.path.join(cache_dir, f"{key}.txt")
        with open(cache_path, "w") as f:
            f.write(str(response))
        return None

    # Load environment variables
    if openai_api_keys is None:
        openai_api_keys = os.getenv("OPENAI_API_KEY")
    if openai_base_url is None:
        openai_base_url = os.getenv("OPENAI_API_BASE_URL")
    parameters = {
        'model': model,
    }
    parameters.update(kwargs)
    #print(openai_base_url)
    client = OpenAI(api_key=openai_api_keys, base_url=openai_base_url)
    
    cache_dict = {}
    full_keys = [generate_key(input) for input in inputs]

    # Load cache
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        for key in full_keys:
            response = load_cache(key)
            if response is not None:
                cache_dict[key] = response

    # Filter out cached inputs
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for id, key, input in zip(range(len(inputs)), full_keys, inputs):
            if key in cache_dict:
                continue
            future = executor.submit(create_openai_chat_completion, client, input, parameters, key, 0)
            futures.append(future)

        print('Appending all todo futures',len(futures))
        while len(futures) > 0:
            new_futures = []
            for future in as_completed(futures):
                response, input, key, retry_steps = future.result()
                if isinstance(response, Exception):
                    response = 'Exception("' + str(response) + '")'
                    if retry_steps < MAX_RETRY_STEPS:
                        future = executor.submit(create_openai_chat_completion, client, input, parameters, key, retry_steps + 1)
                        new_futures.append(future)
                # key = generate_key(input)
                cache_dict[key] = response
                # Save cache
                if cache_dir is not None:
                    write_cache(key, response)
            time.sleep(1)
            futures = new_futures
            print('One round of futures done',len(futures),'with new futures to go',len(new_futures))
    outputs = [cache_dict[key] for key in full_keys]
    return outputs

def clean_cache(cache_dir: str):
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))
    return None

def filter_out_exception(inputs: List[EvalOutput]) -> List[EvalOutput]:
    return [input for input in inputs if not isinstance(input.raw_output, Exception)]

if __name__ == "__main__":
    # clean_cache("./cache")
    inputs = [
            [{'role': 'system', 'content': 'you are a helpful agent'},
            {'role': 'user', 'content': 'Describe Paris for me.'}],
            
        ]
    outputs = batch_request_openai(
        type="Arena",
        model= 'gpt-4o',
        inputs=inputs,
        num_workers=4,
        openai_api_keys = None ,
        openai_base_url = None,
        cache_dir="/home/yangyaodong/projects/wangkaile/align-anything-kaile/align_anything/evaluation/eval/.cache",
    )
    print(outputs)
    exit()
    results = [EvalOutput(evalEngine="gpt_evaluation", input=input, raw_output=response) for input, response in zip(inputs, outputs)]
    results = filter_out_exception(results)
    print(len(results))