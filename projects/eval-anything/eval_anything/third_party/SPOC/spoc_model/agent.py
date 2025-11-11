# Copyright 2024 Allen Institute for AI
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
    def generate(self, observations: Dict[str, Any], goal: str) -> Tuple[str, Any]:
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
