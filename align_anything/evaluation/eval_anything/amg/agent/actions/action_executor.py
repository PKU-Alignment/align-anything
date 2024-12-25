from typing import Dict, List, Union

from agent.schema import ActionReturn, ActionValidCode

from .base_action import BaseAction
from .builtin_actions import FinishAction, InvalidAction, NoAction


class ActionExecutor:
    """The action executor class.

    Args:
        actions (Union[BaseAction, List[BaseAction]]): The action or actions.
        invalid_action (BaseAction, optional): The invalid action. Defaults to
            InvalidAction().
        no_action (BaseAction, optional): The no action.
            Defaults to NoAction().
        finish_action (BaseAction, optional): The finish action. Defaults to
            FinishAction().
        finish_in_action (bool, optional): Whether the finish action is in the
            action list. Defaults to False.
    """

    def __init__(
        self,
        actions: Union[BaseAction, List[BaseAction]],
        invalid_action: BaseAction = InvalidAction(),
        no_action: BaseAction = NoAction(),
        finish_action: BaseAction = FinishAction(),
        finish_in_action: bool = False,
    ):
        if isinstance(actions, BaseAction):
            actions = [actions]

        for action in actions:
            assert isinstance(
                action, BaseAction
            ), f'action must be BaseAction, but got {type(action)}'
        if finish_in_action:
            actions.append(finish_action)
        self.actions = {action.name: action for action in actions}
        self.invalid_action = invalid_action
        self.no_action = no_action
        self.finish_action = finish_action

    def get_actions_info(self) -> List[Dict]:
        actions = []
        for action_name, action in self.actions.items():
            if not action.enable:
                continue
            if action.is_toolkit:
                for api in action.description['api_list']:
                    api_desc = api.copy()
                    api_desc['name'] = f"{action_name}.{api_desc['name']}"
                    actions.append(api_desc)
            else:
                action_desc = action.description.copy()
                actions.append(action_desc)
        return actions

    def is_valid(self, name: str):
        return name in self.actions and self.actions[name].enable

    def action_names(self, only_enable: bool = True):
        if only_enable:
            return [k for k, v in self.actions.items() if v.enable]
        else:
            return list(self.actions.keys())

    def add_action(self, action: BaseAction):
        assert isinstance(action, BaseAction), f'action must be BaseAction, but got {type(action)}'
        self.actions[action.name] = action

    def del_action(self, name: str):
        if name in self.actions:
            del self.actions[name]

    def __call__(self, name: str, command: str) -> ActionReturn:
        action_name, api_name = name.split('.') if '.' in name else (name, 'run')
        if not self.is_valid(action_name):
            if name == self.no_action.name:
                action_return = self.no_action(command)
            elif name == self.finish_action.name:
                action_return = self.finish_action(command)
            else:
                action_return = self.invalid_action(command)
        else:
            action_return = self.actions[action_name](command, api_name)
            action_return.valid = ActionValidCode.OPEN
        return action_return
