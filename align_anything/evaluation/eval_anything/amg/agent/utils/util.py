# Copyright 2024 PKU-Alignment Team and Lagent Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import hashlib
import json
from typing import List, Optional, Union


def filter_suffix(response: Union[str, List[str]], suffixes: Optional[List[str]] = None) -> str:
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
