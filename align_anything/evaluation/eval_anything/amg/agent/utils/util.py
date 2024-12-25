from typing import List, Optional, Union

import hashlib
import json

def filter_suffix(response: Union[str, List[str]],
                  suffixes: Optional[List[str]] = None) -> str:
    """Filter response with suffixes.

    Args:
        response (Union[str, List[str]]): generated responses by LLMs.
        suffixes (str): a list of suffixes to be deleted.

    Return:
        str: a clean response.
    """
    if suffixes is None:
        return response
    batched = True
    if isinstance(response, str):
        response = [response]
        batched = False
    processed = []
    for resp in response:
        for item in suffixes:
            # if response.endswith(item):
            #     response = response[:len(response) - len(item)]
            if item in resp:
                resp = resp.split(item)[0]
        processed.append(resp)
    if not batched:
        return processed[0]
    return processed

def generate_hash_uid(to_hash: dict | tuple | list | str) -> str:
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    return hash_object.hexdigest()