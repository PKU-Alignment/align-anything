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


try:
    from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioPreTrainedModel
    Qwen2Audio_AVALIABLE = True
    class AccustomedQwen2AudioModel(Qwen2AudioForConditionalGeneration):

        @classmethod
        def pretrain_class(cls) -> Qwen2AudioPreTrainedModel:
            return Qwen2AudioPreTrainedModel
    
except ImportError:
    Qwen2Audio_AVALIABLE = False
    print("Qwen2Audio is currently not available.")
