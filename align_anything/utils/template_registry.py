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

TEMPLATE_REGISTRY = {}
EVAL_TEMPLATE_REGISTRY = {}


def register_template(template_name):
    def decorator(cls):
        TEMPLATE_REGISTRY[template_name] = cls
        return cls

    return decorator


def register_eval_template(template_name):
    def decorator(cls):
        EVAL_TEMPLATE_REGISTRY[template_name] = cls
        return cls

    return decorator


def get_template_class(template_name: str):
    template_class = TEMPLATE_REGISTRY.get(template_name)
    if template_class is None:
        raise ValueError(f"Template '{template_name}' not found.")
    return template_class()


def get_eval_template_class(template_name: str):
    template_class = EVAL_TEMPLATE_REGISTRY.get(template_name)
    if template_class is None:
        raise ValueError(f"Template '{template_name}' not found.")
    return template_class()
