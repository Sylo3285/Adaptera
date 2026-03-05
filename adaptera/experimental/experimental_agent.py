import torch
from typing import Dict, List, Optional
from adaptera.model.core import AdapteraModel
from adaptera.tools.core import Tool
from adaptera.experimental.agent_helpers.planner import Planner

class ToolGate:
    """
    A gate that decides whether ANY tool should be used for a user query
    using the main LLM for semantic understanding.
    """
    def __init__(self, tools: Dict[str, "Tool"], llm:AdapteraModel):
        """
        tools: dict of Tool objects
        llm: a callable that takes a prompt and returns a string
        """
        self.tools = tools
        self.llm = llm

    def tool_needed(self, user_input: str) -> bool:
        """
        Returns True if ANY tool is needed for the user input.
        Uses the LLM to determine if a tool is required.
        """
        # Build a dynamic prompt listing all tools and their descriptions
        tool_list_str = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        prompt = f"""
            You are an assistant that decides whether a user query requires any tool.
            Available tools:
            {tool_list_str}

            User query: "{user_input}"

            Answer "Yes" if any tool is needed, "No" if no tools are needed.

            ONLY RESPOND WITH "Yes" or "No" NO OTHER ANSWERS ARE ACCEPTED.
        """

        # Call the LLM
        response = self.llm.generate(prompt,min_new_tokens=1,max_new_tokens=3)
        response_lower = response.strip().lower()
        return "yes" in response_lower

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

    def __init__(self,llm_name: str, llm: AdapteraModel,planning_model:AdapteraModel, tools: Optional[List[Tool]] = None, max_iterations: int = 5 , description:str=None,CORE_SYSTEM_PROMPT:str=None,SAFETY:bool=True):
        self.name = llm_name
        self.llm = llm
        self.planning_model = planning_model

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

        self.temp_memory_bank = [] # A temporary state to store and fetch information after each step
        self.planner = Planner(tools=tools, planning_model=self.planning_model,verbose=True)
    
    def run(self,user_input:str):

        if not self.tools:
            print(f"{self.COLOR_AGENT}No tools available. Processing input directly with LLM...{self.COLOR_RESET}")
            response = self.llm.generate(user_input)
            print(f"{self.COLOR_FINAL}Final Response: {response}{self.COLOR_RESET}")
            return response

        # Check if any tool is needed for the user input

        gate = ToolGate(tools= self.tools,llm=self.llm)

        if not gate.tool_needed(user_input):
            print(f"{self.COLOR_AGENT}No tools needed for the input. Processing directly with LLM...{self.COLOR_RESET}")
            response = self.llm.generate(user_input)
            print(f"{self.COLOR_FINAL}Final Response: {response}{self.COLOR_RESET}")
            return response

        print(f"{self.COLOR_AGENT}Tool needed for the input. Planning...{self.COLOR_RESET}")
        plan = self.planner.plan(user_input)
        print(plan)


        """while True:
            #plan things out using planner
            next_step = self.planner.next_pending()
            self.planner.update_status(next_step, "in_progress")
            next_step = self.planner.state.todos[next_step] if next_step is not None else None
            
            #schedule the plan using a Scheduler or call the scheduler internally inside the planner to sort out the steps.

            #Execute the plan with the executor

            #Verify each execution and update the planner state accordingly

            #Termination condition - either all steps completed or max iterations reached"""