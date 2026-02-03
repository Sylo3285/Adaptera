from typing import Any, List, Optional
from adaptera.model.core import AdapteraModel
from adaptera.tools.core import Tool


class Agent:
    """
    A strict ReAct-style agent.
    Cycles through Thought -> Action -> Action Input -> Observation until Final Answer.

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

    def _get_system_prompt(self) -> str:
        if not self.tools:
            return "You are a helpful assistant. Answer the following question directly.\n\nQuestion: "

        tool_names = ", ".join(self.tools.keys())
        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        return f"""You are a helpful assistant. Answer the following questions as best you can.
You have access to the following tools ONLY:

{tool_desc}

### Rules:
1. ONLY use tools listed above. If you do not need a tool to answer the question, or if no relevant tool is available, go straight to "Final Answer:".
2. DO NOT hallucinate tools. If you use a tool, it MUST be one of: [{tool_names}].
3. If you have the information already, or it's a creative task (like writing a poem), do not use a tool. Just answer.

### Format:
Question: the input question you must answer
Thought: logical reasoning to determine if a tool is needed.
Action: the tool name (ONLY if needed), MUST be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
... (Thought/Action/Action Input/Observation can repeat)
Final Answer: the final answer to the original input question

Begin!

Question: """

    def run(self, task: str,
        min_new_tokens: int = 16,
        max_new_tokens: int = 128,
        top_p: float = 0.9,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k_memory: int = 5,
        **kwargs,) -> str:
        
        prompt = self.CORE_SYSTEM_PROMPT + task + "\n" if self.CORE_SYSTEM_PROMPT and not self.SAFETY else self._get_system_prompt() + task + "\n"
        
        if not self.tools:
            # Direct generation if no tools are available
            response = self.llm.generate(prompt, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_p=top_p, top_k_memory=top_k_memory, **kwargs)
            if response.startswith(prompt):
                response = response[len(prompt):]
            return response.strip()

        intermediate_steps = ""
        for i in range(self.max_iterations):
            current_prompt = prompt + intermediate_steps + "Thought: "
            
            response = self.llm.generate(current_prompt, max_new_tokens=512)
            if response.startswith(current_prompt):
                response = response[len(current_prompt):]
            
            # Manually handle stopping if "Observation:" is generated
            if "Observation:" in response:
                response = response.split("Observation:")[0].strip()
            
            full_output = "Thought: " + response
            print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_THOUGHT}{full_output}{self.COLOR_RESET}")

            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer

            # Parse Action and Action Input
            if "Action:" in response and "Action Input:" in response:
                try:
                    action_part = response.split("Action:")[1].split("Action Input:")[0].strip()
                    action_input = response.split("Action Input:")[1].split("Observation:")[0].strip()
                    # Strip leading/trailing quotes if model adds them
                    action_input = action_input.strip("\"'")
                    
                    print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_ACTION}Action: {action_part}{self.COLOR_RESET}")
                    print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_ACTION}Action Input: {action_input}{self.COLOR_RESET}")

                    if action_part in self.tools:
                        observation = self.call_tool(action_part, action_input)
                        print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_OBSERVATION}Observation: {observation}{self.COLOR_RESET}")
                        intermediate_steps += response + f"\nObservation: {observation}\n"
                    else:
                        error_msg = f"Tool '{action_part}' is not available. Please answer directly or use available tools: [{', '.join(self.tools.keys())}]."
                        print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_ERROR}{error_msg}{self.COLOR_RESET}")
                        intermediate_steps += response + f"\nObservation: {error_msg}\n"
                        
                except Exception as e:
                    error_msg = "Error parsing Action/Action Input. Use the format: Action: [tool] then Action Input: [input]."
                    print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_ERROR}{error_msg}{self.COLOR_RESET}")
                    intermediate_steps += response + f"\nObservation: {error_msg}\n"
            else:
                # If neither Final Answer nor Action is present, nudge the model
                error_msg = "Please provide either an 'Action:' or a 'Final Answer:'."
                print(f"{self.COLOR_AGENT}{self.name}{self.COLOR_RESET}: {self.COLOR_ERROR}{error_msg}{self.COLOR_RESET}")
                intermediate_steps += response + f"\nObservation: {error_msg}\n"

        return "Reached maximum iterations without a final answer."

    def call_tool(self, tool_name: str, tool_input: str) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        try:
            # If the tool input is comma-separated, try to split it
            if "," in tool_input:
                args = [arg.strip().strip("\"'") for arg in tool_input.split(",")]
                # Convert to numbers if possible
                processed_args = []
                for arg in args:
                    try:
                        if "." in arg:
                            processed_args.append(float(arg))
                        else:
                            processed_args.append(int(arg))
                    except ValueError:
                        processed_args.append(arg)
                return tool.func(*processed_args)
            
            #if the tool input is empty call it directly
            elif tool_input.strip() == "" or tool_input is None or tool_input.lower() == "none" or tool_input.lower() == "None":
                return tool.func()
            else:
                clean_input = tool_input.strip().strip("\"'")
                try:
                    if "." in clean_input:
                        val = float(clean_input)
                    else:
                        val = int(clean_input)
                    return tool.func(val)
                except ValueError:
                    return tool.func(clean_input)
        except Exception as e:
            return f"Error executing tool: {str(e)}"

