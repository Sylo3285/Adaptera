"""Adaptera: A local-first LLM orchestration library."""

from .chains.agent import Agent
from .chains.multiagent import MultiAgent
from .memory.core import VectorDB
from .model.core import AdapteraHFModel, AdapteraLMSModel
from .tools.core import Tool
from .about import about

__all__ = ["Agent", "MultiAgent", "VectorDB", "AdapteraHFModel","AdapteraLMSModel", "Tool"]
__version__ = "0.1.3"