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

EVALUATOR_REGISTRY = {}


def register_evaluator(evaluator_name):
    def decorator(cls):
        EVALUATOR_REGISTRY[evaluator_name] = cls
        return cls

    return decorator


def get_template_class(evaluator_name: str, cfgs: tuple, ds_cfgs: tuple):
    evaluator_class = EVALUATOR_REGISTRY.get(evaluator_name)
    if evaluator_class is None:
        raise ValueError(f"Template '{evaluator_name}' not found.")
    return evaluator_class(cfgs, ds_cfgs)
