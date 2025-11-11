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

"""
注册器，实现metric和template的注册
"""


class MetricRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_cls):
            cls._registry[name] = metric_cls
            metric_cls._registered_name = name
            return metric_cls

        return decorator

    @classmethod
    def get_metric(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)


class TemplateRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(template_cls):
            cls._registry[name] = template_cls
            template_cls.__name__ = name
            return template_cls

        return decorator

    @classmethod
    def get_template(cls, name: str, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Template '{name}' is not registered!")
        return cls._registry[name](*args, **kwargs)


class BenchmarkRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(benchmark_cls):
            cls._registry[name] = benchmark_cls
            benchmark_cls._registered_name = name
            return benchmark_cls

        return decorator

    @classmethod
    def get_benchmark(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Benchmark '{name}' is not registered!")
        return cls._registry[name]


class MMDatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(mm_dataset_cls):
            cls._registry[name] = mm_dataset_cls
            mm_dataset_cls._registered_name = name
            return mm_dataset_cls

        return decorator

    @classmethod
    def get_mm_dataset(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered!")
        return cls._registry[name]


class MMDataManagerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(mm_data_manager_cls):
            cls._registry[name] = mm_data_manager_cls
            mm_data_manager_cls._registered_name = name
            return mm_data_manager_cls

        return decorator

    @classmethod
    def get_mm_data_manager(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"DataManager '{name}' is not registered!")
        return cls._registry[name]


class PromptBuilderRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(prompt_builder_cls):
            cls._registry[name] = prompt_builder_cls
            prompt_builder_cls._registered_name = name
            return prompt_builder_cls

        return decorator

    @classmethod
    def get_prompt_builder(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Prompt builder '{name}' is not registered!")
        return cls._registry[name]


class DataloaderRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dataloader_cls):
            cls._registry[name] = dataloader_cls
            dataloader_cls._registered_name = name
            return dataloader_cls

        return decorator

    @classmethod
    def get_dataloader(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Dataloader '{name}' is not registered!")
        return cls._registry[name]


class AnswerExtractorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(extractor_cls):
            cls._registry[name] = extractor_cls
            extractor_cls._registered_name = name
            return extractor_cls

        return decorator

    @classmethod
    def get_extractor(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Answer extractor '{name}' is not registered!")
        return cls._registry[name]


class JudgeRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(judge_cls):
            cls._registry[name] = judge_cls
            judge_cls._registered_name = name
            return judge_cls

        return decorator

    @classmethod
    def get_judge(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Judge method '{name}' is not registered!")
        return cls._registry[name]
