# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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

from typing import List

import requests
import torch


class RemoteRewardModel:
    """
    RemoteRewardModel
    remote reward model client, interact with remote reward service via HTTP API

    Args:
        endpoint (str): URL of the remote reward model API
        timeout (int): API request timeout (seconds)
        retry_times (int): Number of retries on request failure
    """

    def __init__(self, endpoint: str, timeout: int = 100, retry_times: int = 3):
        self.endpoint = endpoint
        self.timeout = timeout
        self.retry_times = retry_times
        self.headers = {'Content-Type': 'application/json'}

    def score(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Get reward scores for given prompts and responses

        Args:
            prompts: List of input prompts
            responses: List of corresponding responses
        Returns:
            A tensor containing the reward scores
        """
        assert len(prompts) == len(
            responses
        ), 'The number of prompts and responses must be the same'

        request_json = {
            'prompts': prompts,
            'responses': responses,
        }
        # Golden responses is not used in the remote reward model, but will be provided when load reward model server
        for attempt in range(self.retry_times):
            try:
                print(f'Sending request to {self.endpoint}')
                response = requests.post(
                    self.endpoint,
                    json=request_json,
                    headers=self.headers,
                    verify=False,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    # Assume the API returns format is {"rewards": [...]}
                    rewards = torch.tensor(response.json()['rewards'])
                    return rewards
                else:
                    print(f'API request failed, status code: {response.status_code}')

            except Exception as e:
                print(
                    f'Remote reward API request error (attempt {attempt+1}/{self.retry_times}): {e}'
                )
                import time

                time.sleep(1)

        raise RuntimeError(f'Failed to get rewards from API, tried {self.retry_times} times')
