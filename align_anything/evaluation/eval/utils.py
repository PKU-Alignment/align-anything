from typing import List
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import os
def batch_request_openai(
    type: str,
    inputs: List,
    model: str,
    num_workers: int = 1 , 
    cache_dir: str = None,
    openai_api_keys: str = None,
    openai_base_url: str = None,
    kwargs: dict = {}
) -> List:
    if openai_api_keys is None:
        openai_api_keys = os.getenv("OPENAI_API_KEY")
    if openai_base_url is None:
        openai_base_url = os.getenv("OPENAI_API_BASE_URL")
    parameters = {
        'model': model,
    }
    parameters.update(kwargs)
    client = OpenAI(api_key=openai_api_keys, base_url=openai_base_url)
    outputs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for input in inputs:
            future = executor.submit(client.chat.completions.create, 
                                        model = parameters['model'],
                                        messages=input,
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
            futures.append(future)
        
        for future in futures:
            response = future.result()
            outputs.append(response.choices[0].message)
    
    return outputs


if __name__ == "__main__":
    output = batch_request_openai(
        type="Arena",
        inputs=[
            [{'role': 'system', 'content': 'judge which response is better'},
            {'role': 'user', 'content': '[A]This is a test prompt[B]This is another test prompt'}],
        ],
        num_workers=2,
        parameters={
            'model': 'deepseek-chat',
        }
    )
    print(output)