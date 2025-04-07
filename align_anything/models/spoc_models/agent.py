# Copyright 2024 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class AbstractAgent(ABC):
    """
    Abstract base class for an agent.

    This class provides a template for agents that can be used in different environments.
    It declares methods that should be implemented by any concrete agent class.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the agent's state.

        This method should be implemented by each subclass to reset the agent's internal state.
        It is typically called at the beginning of each episode.
        """

    @abstractmethod
    def get_action_list(self) -> List[str]:
        """
        Get the list of possible actions.

        This method should be implemented by each subclass to return a list of possible actions
        that the agent can take in the environment.

        Returns:
            List[str]: A list of action names.
        """

    @abstractmethod
    def get_action(self, observations: Dict[str, Any], goal: str) -> Tuple[str, Any]:
        """
        Decide on the action to take based on the observations and goal.

        This method should be implemented by each subclass to decide on the action to take
        based on the current observations from the environment and the goal.

        Args:
            observations (Dict[str, Any]): The current observations from the environment.
            goal (Any): The current goal.

        Returns:
            Tuple[str, Any]: The chosen action and the action probabilities.
        """
