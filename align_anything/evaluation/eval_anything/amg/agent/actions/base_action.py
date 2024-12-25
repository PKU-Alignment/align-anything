import inspect
import logging
import re
from abc import ABCMeta
from copy import deepcopy
from functools import wraps
from typing import Callable, Optional, Type, get_args, get_origin


try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from class_registry import AutoRegister, ClassRegistry
from griffe import Docstring


try:
    from griffe import DocstringSectionKind
except ImportError:
    from griffe.enumerations import DocstringSectionKind

from ..schema import ActionReturn, ActionStatusCode
from .parser import BaseParser, JsonParser, ParseError


logging.getLogger('griffe').setLevel(logging.ERROR)

TOOL_REGISTRY = ClassRegistry('__tool_name__', unique=True)


def tool_api(
    func: Optional[Callable] = None,
    *,
    explode_return: bool = False,
    returns_named_value: bool = False,
    **kwargs,
):
    """Turn functions into tools. It will parse typehints as well as docstrings
    to build the tool description and attach it to functions via an attribute
    ``api_description``.

    Examples:

        .. code-block:: python

            # typehints has higher priority than docstrings
            from typing import Annotated

            @tool_api
            def add(a: Annotated[int, 'augend'], b: Annotated[int, 'addend'] = 1):
                '''Add operation

                Args:
                    x (int): a
                    y (int): b
                '''
                return a + b

            print(add.api_description)

    Args:
        func (Optional[Callable]): function to decorate. Defaults to ``None``.
        explode_return (bool): whether to flatten the dictionary or tuple return
            as the ``return_data`` field. When enabled, it is recommended to
            annotate the member in docstrings. Defaults to ``False``.

            .. code-block:: python

                @tool_api(explode_return=True)
                def foo(a, b):
                    '''A simple function

                    Args:
                        a (int): a
                        b (int): b

                    Returns:
                        dict: information of inputs
                            * x: value of a
                            * y: value of b
                    '''
                    return {'x': a, 'y': b}

                print(foo.api_description)

        returns_named_value (bool): whether to parse ``thing: Description`` in
            returns sections as a name and description, rather than a type and
            description. When true, type must be wrapped in parentheses:
            ``(int): Description``. When false, parentheses are optional but
            the items cannot be named: ``int: Description``. Defaults to ``False``.

    Returns:
        Callable: wrapped function or partial decorator

    Important:
        ``return_data`` field will be added to ``api_description`` only
        when ``explode_return`` or ``returns_named_value`` is enabled.
    """

    def _detect_type(string):
        field_type = 'STRING'
        if 'list' in string:
            field_type = 'Array'
        elif 'str' not in string:
            if 'float' in string:
                field_type = 'FLOAT'
            elif 'int' in string:
                field_type = 'NUMBER'
            elif 'bool' in string:
                field_type = 'BOOLEAN'
        return field_type

    def _explode(desc):
        kvs = []
        desc = '\nArgs:\n' + '\n'.join(
            ['    ' + item.lstrip(' -+*#.') for item in desc.split('\n')[1:] if item.strip()]
        )
        docs = Docstring(desc).parse('google')
        if not docs:
            return kvs
        if docs[0].kind is DocstringSectionKind.parameters:
            for d in docs[0].value:
                d = d.as_dict()
                if not d['annotation']:
                    d.pop('annotation')
                else:
                    d['type'] = _detect_type(d.pop('annotation').lower())
                kvs.append(d)
        return kvs

    def _parse_tool(function):
        # remove rst syntax
        docs = Docstring(re.sub(':(.+?):`(.+?)`', '\\2', function.__doc__ or '')).parse(
            'google', returns_named_value=returns_named_value, **kwargs
        )
        desc = dict(
            name=function.__name__,
            description=docs[0].value if docs[0].kind is DocstringSectionKind.text else '',
            parameters=[],
            required=[],
        )
        args_doc, returns_doc = {}, []
        for doc in docs:
            if doc.kind is DocstringSectionKind.parameters:
                for d in doc.value:
                    d = d.as_dict()
                    d['type'] = _detect_type(d.pop('annotation').lower())
                    args_doc[d['name']] = d
            if doc.kind is DocstringSectionKind.returns:
                for d in doc.value:
                    d = d.as_dict()
                    if not d['name']:
                        d.pop('name')
                    if not d['annotation']:
                        d.pop('annotation')
                    else:
                        d['type'] = _detect_type(d.pop('annotation').lower())
                    returns_doc.append(d)

        sig = inspect.signature(function)
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            parameter = dict(
                name=param.name,
                type='STRING',
                description=args_doc.get(param.name, {}).get('description', ''),
            )
            annotation = param.annotation
            if annotation is inspect.Signature.empty:
                parameter['type'] = args_doc.get(param.name, {}).get('type', 'STRING')
            else:
                if get_origin(annotation) is Annotated:
                    annotation, info = get_args(annotation)
                    if info:
                        parameter['description'] = info
                while get_origin(annotation):
                    annotation = get_args(annotation)
                parameter['type'] = _detect_type(str(annotation))
            desc['parameters'].append(parameter)
            if param.default is inspect.Signature.empty:
                desc['required'].append(param.name)

        return_data = []
        if explode_return:
            return_data = _explode(returns_doc[0]['description'])
        elif returns_named_value:
            return_data = returns_doc
        if return_data:
            desc['return_data'] = return_data
        return desc

    if callable(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper.api_description = _parse_tool(func)
        return wrapper

    def decorate(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper.api_description = _parse_tool(func)
        return wrapper

    return decorate


class ToolMeta(ABCMeta):
    """Metaclass of tools."""

    def __new__(mcs, name, base, attrs):
        is_toolkit, tool_desc = True, dict(
            name=attrs.setdefault('__tool_name__', name),
            description=Docstring(attrs.get('__doc__', '')).parse('google')[0].value,
        )
        for key, value in attrs.items():
            if callable(value) and hasattr(value, 'api_description'):
                api_desc = getattr(value, 'api_description')
                if key == 'run':
                    tool_desc['parameters'] = api_desc['parameters']
                    tool_desc['required'] = api_desc['required']
                    if api_desc['description']:
                        tool_desc['description'] = api_desc['description']
                    if api_desc.get('return_data'):
                        tool_desc['return_data'] = api_desc['return_data']
                    is_toolkit = False
                else:
                    tool_desc.setdefault('api_list', []).append(api_desc)
        if not is_toolkit and 'api_list' in tool_desc:
            raise KeyError('`run` and other tool APIs can not be implemented ' 'at the same time')
        if is_toolkit and 'api_list' not in tool_desc:
            is_toolkit = False
            if callable(attrs.get('run')):
                run_api = tool_api(attrs['run'])
                api_desc = run_api.api_description
                tool_desc['parameters'] = api_desc['parameters']
                tool_desc['required'] = api_desc['required']
                if api_desc['description']:
                    tool_desc['description'] = api_desc['description']
                if api_desc.get('return_data'):
                    tool_desc['return_data'] = api_desc['return_data']
                attrs['run'] = run_api
            else:
                tool_desc['parameters'], tool_desc['required'] = [], []
        attrs['_is_toolkit'] = is_toolkit
        attrs['__tool_description__'] = tool_desc
        return super().__new__(mcs, name, base, attrs)


class BaseAction(metaclass=AutoRegister(TOOL_REGISTRY, ToolMeta)):
    """Base class for all actions.

    Args:
        description (:class:`Optional[dict]`): The description of the action.
            Defaults to ``None``.
        parser (:class:`Type[BaseParser]`): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (:class:`bool`): Whether the action is enabled. Defaults to
            ``True``.

    Examples:

        * simple tool

        .. code-block:: python

            class Bold(BaseAction):
                '''Make text bold'''

                def run(self, text: str):
                    '''
                    Args:
                        text (str): input text

                    Returns:
                        str: bold text
                    '''
                    return '**' + text + '**'

            action = Bold()

        * toolkit with multiple APIs

        .. code-block:: python

            class Calculator(BaseAction):
                '''Calculator'''

                @tool_api
                def add(self, a, b):
                    '''Add operation

                    Args:
                        a (int): augend
                        b (int): addend

                    Returns:
                        int: sum
                    '''
                    return a + b

                @tool_api
                def sub(self, a, b):
                    '''Subtraction operation

                    Args:
                        a (int): minuend
                        b (int): subtrahend

                    Returns:
                        int: difference
                    '''
                    return a - b

            action = Calculator()
    """

    def __init__(
        self,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ):
        self._description = deepcopy(description or self.__tool_description__)
        self._name = self._description['name']
        self._parser = parser(self)
        self._enable = enable

    def __call__(self, inputs: str, name='run') -> ActionReturn:
        fallback_args = {'inputs': inputs, 'name': name}
        if not hasattr(self, name):
            return ActionReturn(
                fallback_args,
                type=self.name,
                errmsg=f'invalid API: {name}',
                state=ActionStatusCode.API_ERROR,
            )
        try:
            inputs = self._parser.parse_inputs(inputs, name)
        except ParseError as exc:
            return ActionReturn(
                fallback_args, type=self.name, errmsg=exc.err_msg, state=ActionStatusCode.ARGS_ERROR
            )
        try:
            outputs = getattr(self, name)(**inputs)
        except Exception as exc:
            return ActionReturn(
                inputs, type=self.name, errmsg=str(exc), state=ActionStatusCode.API_ERROR
            )
        if isinstance(outputs, ActionReturn):
            action_return = outputs
            if not action_return.args:
                action_return.args = inputs
            if not action_return.type:
                action_return.type = self.name
        else:
            result = self._parser.parse_outputs(outputs)
            action_return = ActionReturn(inputs, type=self.name, result=result)
        return action_return

    @property
    def name(self):
        return self._name

    @property
    def enable(self):
        return self._enable

    @property
    def is_toolkit(self):
        return self._is_toolkit

    @property
    def description(self) -> dict:
        """Description of the tool."""
        return self._description

    def __repr__(self):
        return f'{self.description}'

    __str__ = __repr__
