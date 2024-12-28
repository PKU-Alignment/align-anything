from typing import Optional

from agent.actions.base_action import BaseAction, tool_api
from agent.actions.parser import BaseParser
from agent.schema import ActionReturn, ActionStatusCode, ActionValidCode


class InvalidAction(BaseAction):
    """This is a invalid action class, which is used to return error message
    when the action is invalid.

    Args:
        err_msg (str): The error message. Defaults to 'The action is invalid,
            please check the action name'.

    Returns:
        ActionReturn: The action return.
    """

    def __init__(
        self,
        err_msg: str = 'The action is invalid, please check the action name.',
        description: Optional[dict] = None,
        parser=BaseParser,
    ) -> None:
        super().__init__(description, parser, enable=False)
        self._err_msg = err_msg

    @tool_api
    def run(self, err_msg: Optional[str] = None) -> ActionReturn:
        """Return the error message.

        Args:
            err_msg (str, optional): The error message. If err_msg is not None,
                it will be returned, otherwise the default error message will
                be returned. Defaults to None.
        """
        action_return = ActionReturn(
            url=None,
            args=dict(text=err_msg),
            errmsg=err_msg or self._err_msg,
            type=self.name,
            valid=ActionValidCode.INVALID,
            state=ActionStatusCode.API_ERROR,
        )
        return action_return


class NoAction(BaseAction):
    """This is a no action class, which is used to return error message when
    the response does not follow the format.

    Args:
        err_msg (str): The error message. Defaults to
            'Please follow the format'.
    """

    def __init__(
        self,
        err_msg: str = 'Please follow the format',
        description: Optional[dict] = None,
        parser=BaseParser,
    ):
        super().__init__(description, parser, enable=False)
        self._err_msg = err_msg

    @tool_api
    def run(self, err_msg: Optional[str] = None) -> ActionReturn:
        """Return the error message.

        Args:
            err_msg (str, optional): The error message. If err_msg is not None,
                it will be returned, otherwise the default error message will
                be returned. Defaults to None.

        Returns:
            ActionReturn: The action return.
        """
        action_return = ActionReturn(
            url=None,
            args=dict(text=err_msg),
            type=self.name,
            errmsg=err_msg or self._err_msg,
            valid=ActionValidCode.INVALID,
            state=ActionStatusCode.API_ERROR,
        )
        return action_return


class FinishAction(BaseAction):
    """This is a finish action class, which is used to return the final
    result."""

    def __init__(self, description: Optional[dict] = None, parser=BaseParser):
        super().__init__(description, parser, enable=True)

    @tool_api
    def run(self, response: str) -> ActionReturn:
        """Return the final result.

        Args:
            response (str): The final result.

        Returns:
            ActionReturn: The action return.
        """
        action_return = ActionReturn(
            url=None,
            args=dict(text=response),
            result=[dict(type='text', content=response)],
            type=self.name,
            valid=ActionValidCode.FINISH,
            state=ActionStatusCode.SUCCESS,
        )
        return action_return
