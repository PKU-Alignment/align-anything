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

from typing import Type

from .action_executor import ActionExecutor
from .base_action import TOOL_REGISTRY, BaseAction, tool_api
from .builtin_actions import FinishAction, InvalidAction, NoAction
from .modality_generator import ModalityGenerator
from .parser import BaseParser, JsonParser, TupleParser


__all__ = [
    'BaseAction',
    'ActionExecutor',
    'InvalidAction',
    'FinishAction',
    'NoAction',
    'BINGMap',
    'BaseParser',
    'ModalityGenerator',
    'JsonParser',
    'TupleParser',
    'tool_api',
    'list_tools',
    'get_tool_cls',
    'get_tool',
]


def list_tools(with_class: bool = False):
    """List available tools.

    Args:
        with_class (bool): whether to return the action class along
            with its name. Defaults to ``False``.

    Returns:
        list: all action names
    """
    return list(TOOL_REGISTRY.items()) if with_class else list(TOOL_REGISTRY.keys())


def get_tool_cls(specifier: str) -> Type[BaseAction]:
    """Get the action class.

    Args:
        specifier (:class:`str`): tool name

    Returns:
        Type[BaseAction]: action class
    """
    return TOOL_REGISTRY.get_class(specifier)


def get_tool(specifier: str, *args, **kwargs) -> BaseAction:
    """Instantiate an action.

    Args:
        specifier (str): tool name
        args: positional arguments passed to the action's ``__init__`` method
        kwargs: keyword arguments passed to the action's ``__init__`` method

    Returns:
        :class:`BaseAction`: action object
    """
    return TOOL_REGISTRY.get(specifier, *args, **kwargs)
