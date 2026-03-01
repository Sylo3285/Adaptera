from typing import Any, List, Optional
from adaptera.model.core import AdapteraModel
from adaptera.tools.core import Tool

class Agent:
    """
    A planning based agent which plans , uses tools to solve the problem

    Args:
        llm_name (str): Name of the agent.
        llm (AdapteraModel): The language model to use.
        tools (Optional[List[Tool]]): A list of tools the agent can use.
        max_iterations (int): Maximum number of Thought/Action cycles before stopping.
        description (str): Optional description of the agent's purpose.
        CORE_SYSTEM_PROMPT (str): Optional custom system prompt for the agent.
        SAFETY (bool): If True, prevents overriding system prompt with custom CORE_SYSTEM_PROMPT.
    """

    def __init__(self,llm_name: str, llm: AdapteraModel, tools: Optional[List[Tool]] = None, max_iterations: int = 5 , description:str=None,CORE_SYSTEM_PROMPT:str=None,SAFETY:bool=True):
        self.name = llm_name
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.max_iterations = max_iterations
        self.description = description
        self.CORE_SYSTEM_PROMPT = CORE_SYSTEM_PROMPT
        self.SAFETY = SAFETY # Safety flag for system prompt usage
        if self.CORE_SYSTEM_PROMPT and self.SAFETY:
            print("Using custom CORE_SYSTEM_PROMPT for Agent. This OVERRIDES the default tool description prompt and can lead to unexpected behavior if the format is not followed.")
        
        # ANSI Color Constants
        self.COLOR_AGENT = "\033[38;5;208m" # New Color
        self.COLOR_THOUGHT = "\033[94m"  # Blue
        self.COLOR_ACTION = "\033[93m"   # Yellow
        self.COLOR_OBSERVATION = "\033[92m" # Green
        self.COLOR_ERROR = "\033[91m"    # Red
        self.COLOR_FINAL = "\033[95m"    # Magenta
        self.COLOR_RESET = "\033[0m"