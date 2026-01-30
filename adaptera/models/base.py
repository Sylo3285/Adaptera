"""Abstract base interface for LLM models."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the prompt.

        TODO: Implement generation logic.
        """
        pass

    def stream(self, prompt: str):
        """Optional streaming generation.

        TODO: Implement streaming if supported.
        """
        pass