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

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from eval_anything.models.base_model import BaseModel
from eval_anything.utils.cached_requests import cached_requests
from eval_anything.utils.data_type import InferenceInput, InferenceOutput


class APILM(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], infer_cfgs: Dict[str, Any]):
        super().__init__(model_cfgs)
        self.infer_cfgs = infer_cfgs
        self.api_key = os.getenv('API_KEY')
        self.api_base = os.getenv('API_BASE')
        self.num_workers = os.getenv('NUM_WORKERS', 32)

    def init_model(self):
        pass

    def generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        return self._parallel_generation(inputs)

    def _parallel_generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        if not inputs:
            return []

        if len(inputs) == 1:
            return [self._single_generation(inputs[0])]

        results = {}
        max_workers = min(len(inputs), 32)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._single_generation, input_item): idx
                for idx, input_item in enumerate(inputs)
            }

            for future in tqdm(
                as_completed(future_to_index), total=len(inputs), desc='Generating responses'
            ):
                idx = future_to_index[future]
                result = future.result()
                results[idx] = result

        return [results[i] for i in range(len(inputs))]

    def _single_generation(self, input: InferenceInput) -> InferenceOutput:
        response = cached_requests(
            input.conversation,
            model=self.model_cfgs.model_name_or_path,
            max_completion_tokens=self.infer_cfgs.model_max_length,
            temperature=self.infer_cfgs.temperature,
            top_p=self.infer_cfgs.top_p,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        inference_output = InferenceOutput(
            task=input.task,
            ref_answer=input.ref_answer,
            uuid=input.uuid,
            response=response,
            engine='api',
        )
        return inference_output
