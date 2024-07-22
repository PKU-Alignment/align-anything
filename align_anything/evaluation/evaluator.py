# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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
import re
import tempfile
from typing import Dict, List


class GSM8KEvaluator:
    _decimal_separator = re.compile(r'(?<=\d),(?=\d)')
    _extract_numbers = re.compile(r'[-+]?\d*\.\d+|\d+')

    def score(self, prediction: str, reference: str) -> bool:
        prediction = self.data_process_for_prediction(prediction)
        reference = self.data_process_for_reference(reference)
        return prediction == reference

    def data_process_for_reference(self, text: str) -> str:
        return text.split('#### ')[1].replace(',', '')

    def data_process_for_prediction(self, text: str) -> str:
        pred = self._decimal_separator.sub('', text)
        pred = text.split('\n\n')[0].replace(',', '')
        numbers = self._extract_numbers.findall(pred)
        if numbers:
            # remove trailing zeros
            number = re.sub(r'(?<=\d)\.0*$', '', numbers[-1])
            return number
        else:
            return pred
