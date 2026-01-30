"""Tool registration system."""

from typing import Any, Callable, Dict, List
from dataclasses import dataclass


@dataclass
class Tool:
    """A tool that can be called by an agent."""
    name: str
    func: Callable[..., Any]
    description: str

    def __call__(self, *args, **kwargs) -> Any:
        """Call the tool function."""
        return self.func(*args, **kwargs)
