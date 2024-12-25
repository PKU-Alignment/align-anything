from typing import Type

from .action_executor import ActionExecutor
from .base_action import TOOL_REGISTRY, BaseAction, tool_api
from .parser import BaseParser, JsonParser, TupleParser
from .builtin_actions import FinishAction, InvalidAction, NoAction
from .modality_generator import ModalityGenerator

__all__ = [
    'BaseAction', 'ActionExecutor', 'InvalidAction', 'FinishAction',
    'NoAction', 'BINGMap', 'BaseParser', 'ModalityGenerator',
    'JsonParser', 'TupleParser', 'tool_api', 'list_tools', 'get_tool_cls', 'get_tool'
]


def list_tools(with_class: bool = False):
    """List available tools.

    Args:
        with_class (bool): whether to return the action class along
            with its name. Defaults to ``False``.

    Returns:
        list: all action names
    """
    return list(TOOL_REGISTRY.items()) if with_class else list(
        TOOL_REGISTRY.keys())


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
