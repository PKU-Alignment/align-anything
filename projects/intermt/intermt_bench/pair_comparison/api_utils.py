"""
API utilities for making requests to various AI models with caching mechanism.
"""

import os
import json
import time
import hashlib
import logging
import base64
import io
from typing import Any, Callable, List, Optional, Union

import ray
import urllib3
from urllib3.util.retry import Retry
from PIL import Image
from tqdm import tqdm

from config import API_KEY, MAX_RETRIES, NUM_WORKERS, API_BASE_URL

def encode_image(image_path: str) -> str:
    """
    Open an image from various sources (base64, URL, local path) and encode it as base64.
    
    Args:
        image_path: Path or data of the image to encode
        
    Returns:
        Base64 encoded image string
    """
    if image_path == '' or image_path is None or image_path == [] or image_path == '[]':
        return None
    try:
        if image_path.startswith('data:image'):
            # base64 image
            from io import BytesIO
            image_data = image_path.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        elif image_path.startswith('http://') or image_path.startswith('https://'):
            # URL image
            import requests
            from io import BytesIO
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Local path image
            image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error opening image: {e}")
    
    img = image
    img.thumbnail((512, 512))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64

def process_message_content(content_list):
    """
    Process message content, converting images to base64 format
    
    Args:
        content_list: List of message content
        
    Returns:
        Processed message content list
    """
    processed_content = []
    
    for item in content_list:
        if item['type'] == 'text':
            processed_content.append({"type": "text", "text": item['text']})
        elif item['type'] == 'image':
            image_base64 = encode_image(item['image'])
            if image_base64:
                processed_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    }
                })
    
    return processed_content

@ray.remote(num_cpus=1)
def api_call(
    system_content: str,
    message_content: List[dict],
    model_name: str,
    temperature: float = 0.5,
    post_process: Callable = lambda x: x,
) -> Any:
    """
    Make an API call to an LLM model with retry mechanism.
    
    Args:
        system_content: System prompt
        message_content: Processed message content list
        model_name: Model name to use for generation
        temperature: Temperature for generation
        post_process: Function to process the response
        
    Returns:
        Processed model response
    """
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': message_content},
        ]

    # API endpoint
    openai_api = API_BASE_URL  # Or your preferred API endpoint
    
    params = {
        'model': model_name,
        'messages': messages,
        'temperature': temperature,
    }
    url = openai_api + '/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': API_KEY,
        'Connection': 'close',
    }

    # Set up retry mechanism
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['POST'],
        raise_on_redirect=False,
        raise_on_status=False,
    )
    http = urllib3.PoolManager(retries=retry_strategy)
    encoded_data = json.dumps(params).encode('utf-8')
    
    max_try = MAX_RETRIES
    while max_try > 0:
        try:
            response = http.request('POST', url, body=encoded_data, headers=headers)
            if response.status == 200:
                response_data = json.loads(response.data.decode('utf-8'))
                response_text = response_data['choices'][0]['message']['content']
                logging.info(f"API call succeeded for model {model_name}")
                break
            else:
                err_msg = f'API error, status code: {response.status}, response: {response.data.decode("utf-8")}'
                logging.error(err_msg)
                time.sleep(3)
                max_try -= 1
                continue
        except Exception as e:
            logging.error(f"Exception during API call: {str(e)}")
            time.sleep(3)
            max_try -= 1
            continue
    else:
        logging.error(f"API Failed after {MAX_RETRIES} attempts")
        response_text = 'API Failed'

    return post_process(response_text)

def generate_hash_uid(to_hash: Union[dict, tuple, list, str]) -> str:
    """
    Generate a unique hash for caching based on input data.
    
    Args:
        to_hash: Data to hash
        
    Returns:
        Hash string
    """
    json_string = json.dumps(to_hash, sort_keys=True)
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()
    return hash_uid

def batch_api_call(
    system_contents: List[str],
    message_contents: List[List[dict]],
    model_name: str,
    temperature: float = 0.5,
    num_workers: int = NUM_WORKERS,
    post_process: Callable = lambda x: x,
    cache_dir: str = './cache',
) -> List[Any]:
    """
    Make batch API calls with caching.
    
    Args:
        system_contents: List of system prompts
        message_contents: List of processed message contents
        model_name: Model name to use
        temperature: Temperature for generation
        num_workers: Number of parallel workers
        post_process: Function to process responses
        cache_dir: Directory for caching results
        
    Returns:
        List of processed responses
    """
    if len(system_contents) != len(message_contents):
        raise ValueError('Length of system_contents and message_contents should be equal.')
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers)

    contents = list(enumerate(zip(system_contents, message_contents)))
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
    
    # Generate hashes for caching
    hashes = []
    for idx, (sys_content, msg_content) in enumerate(zip(system_contents, message_contents)):
        to_hash = {
            "system_content": sys_content,
            "message_content": json.dumps(msg_content),
            "model_name": model_name,
            "temperature": temperature
        }
        hashes.append(generate_hash_uid(to_hash))
    
    # Check cache and prepare tasks
    not_finished = []
    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break

        # Start new tasks up to num_workers
        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            hash_id = hashes[index]
            cache_path = os.path.join(cache_dir, f'{hash_id}.json')
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    results[index] = result
                    bar.update(1)
                    continue
                except Exception:
                    # If cache file is corrupt, remove it and proceed
                    os.remove(cache_path)
            
            # Start a new task
            future = api_call.remote(
                content[0], content[1], model_name, temperature, post_process
            )
            not_finished.append([index, future, hash_id])

        if len(not_finished) == 0:
            continue

        # Check for completed tasks
        if not_finished:
            indices, futures, hash_ids = zip(*not_finished)
            finished, pending = ray.wait(list(futures), timeout=1.0)
            
            # Process completed tasks
            for i, task in enumerate(finished):
                idx = indices[futures.index(task)]
                hash_id = hash_ids[futures.index(task)]
                
                results[idx] = ray.get(task)
                
                # Save to cache
                cache_path = os.path.join(cache_dir, f'{hash_id}.json')
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(results[idx], f, ensure_ascii=False, indent=4)
            
            # Update not_finished list
            not_finished = [[idx, fut, hid] for idx, fut, hid in not_finished if fut not in finished]
            
            bar.update(len(finished))
    
    bar.close()
    
    # Ensure all results are collected
    # assert all(result is not None for result in results)
    
    return results 

if __name__ == "__main__":
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init()
    
    # Single API call example
    system_prompt = "You are a helpful AI assistant."
    user_query = "I want to buy a new car, what should I consider?"
    model_name = "gpt-3.5-turbo"
    
    # Remote API call and wait for result
    result_ref = api_call.remote(system_prompt, user_query, model_name)
    result = ray.get(result_ref)
    print("Single call result:", result)
    
    # Batch API call example
    system_prompts = [
        "You are a helpful AI assistant.",
        "You are a professional scientist."
    ]
    user_queries = [
        "I want to buy a new car, what should I consider?",
        "What is the capital of France?"
    ]
    
    # Optional: API call with images
    image_path = "path/to/image.jpg"
    encoded_image = encode_image(image_path)
    
    # Batch API call
    results = batch_api_call(
        system_contents=system_prompts,
        user_contents=user_queries,
        model_name=model_name,
        image_urls=[None, None],  # No images used
        temperature=0.7,
        cache_dir="./my_cache"
    )
    
    # Print results
    for i, result in enumerate(results):
        print(f"Batch {i+1}:", result) 