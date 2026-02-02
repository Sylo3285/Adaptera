"""Tool registration system."""

from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class Tool:
    """A tool that can be called by an agent.
    
    Args:
        name (str): The name of the tool.
        func (Callable[..., Any]): The function that implements the tool.
        description (str): A description of the tool.
    """
    name: str
    func: Callable[..., Any]
    description: str

    def __call__(self, *args, **kwargs) -> Any:
        """Call the tool function."""
        return self.func(*args, **kwargs)
