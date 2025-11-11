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

# ref: https://github.com/OpenSafetyLab/SALAD-BENCH/blob/main/saladbench/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def eval_auprc(targets, preds, print_=False):
    """Evaluate AUPRC.

    Args:
        targets (List[int]): Ground truth labels.
        preds (List[int]): Predicted labels.

    Returns:
        float: AUPRC.
    """
    precision, recall, thresholds = precision_recall_curve(targets, preds)
    auprc = auc(recall, precision)
    if print_:
        print(f'AUPRC: {np.round(auprc, 4)}')
    # TODO: do not round return values
    return auprc


def eval_f1(targets, preds, print_=False):
    """Evaluate F1 score.

    Args:
        targets (List[int]): Ground truth labels.
        preds (List[int]): Predicted labels.

    Returns:
        Tuple[float, float, float, float]: Accuracy, Precision, Recall, F1 score.
    """
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    if print_:
        print(f'Accuracy: {np.round(accuracy, 4)}')
        print(f'Precision: {np.round(precision, 4)}')
        print(f'Recall: {np.round(recal, 4)}')
        print(f'F1: {np.round(f1, 4)}')
    # TODO: do not round return values
    # TODO: move print to outside
    # TODO: consider using a dict or namedtuple
    return (accuracy, precision, recal, f1)
