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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from tqdm import tqdm

from eval_anything.evaluate_tools.metrics import BaseMetric
from eval_anything.utils.data_type import EvaluationResult
from eval_anything.utils.register import MetricRegistry
from eval_anything.utils.utils import check_correctness, estimate_pass_at_k


@MetricRegistry.register('pass_rate')
class PassAtK(BaseMetric):
    # TODO Support k>1 pass@k, need multiple rounds of inference
    def calculate(
        self,
        evaluation_results: List[EvaluationResult],
        judge_method: str,
        k: int = 1,
        n_workers: int = 4,
        timeout: float = 3.0,
        **kwargs,
    ) -> dict[str, float]:
        # Set tokenizer parallelism at the start
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Initialize results for each extractor
        pass_at_k = {extractor: {} for extractor in evaluation_results[0].extracted_result.keys()}

        # Process each extractor separately
        for extractor in evaluation_results[0].extracted_result.keys():
            results = defaultdict(list)

            # Process results in parallel with progress tracking
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                completion_id = Counter()
                n_samples = 0

                for sample in evaluation_results:
                    task_id = sample.ground_truth.get('task_id', '')
                    completion = sample.extracted_result[extractor]
                    args = (sample.ground_truth, completion, timeout, completion_id[task_id])
                    future = executor.submit(check_correctness, *args)
                    futures.append(future)
                    completion_id[task_id] += 1
                    n_samples += 1

                assert len(completion_id) == len(
                    evaluation_results
                ), 'Some problems are not attempted.'

                self.logger.log('info', f'Running test suites for {extractor}...')
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    results[result['task_id']].append((result['completion_id'], result))

            # Calculate pass@k for this extractor
            total, correct = [], []
            for result in results.values():
                result.sort()
                passed = [r[1]['passed'] for r in result]
                total.append(len(passed))
                correct.append(sum(passed))
            total = np.array(total)
            correct = np.array(correct)

            # Calculate pass@k score if we have enough samples
            if (total >= k).all():
                pass_at_k[extractor] = estimate_pass_at_k(total, correct, k).mean()

        return pass_at_k

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)
