from typing import List
from adaptera.model.core import AdapteraModel
from adaptera.chains.agent import Agent
from adaptera.tools.core import Tool

# ANSI color codes for debug logs
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET = "\033[0m"


class MultiAgent:
    """
    A multi-agent coordinator that reasons like a ReAct agent.
    Coordinator uses Thought → Action → Observation → scratchpad
    to decide which agent to delegate to or when to output final answer.
    """

    def __init__(self, agents: List[Agent], coordinator_model: AdapteraModel = None):
        if not agents:
            raise ValueError("At least one agent must be provided.")
        if not coordinator_model:
            raise ValueError("A coordinator model must be provided.")
        
        self.agents = agents
        self.descriptions = [
            {"name": f"Agent_{i+1}", "description": a.description or "No description"}
            for i, a in enumerate(self.agents)
        ]

        tools = [
            Tool(name = "call_model", func=self.call_sub_agent, description="Calls one of the sub agents with the provided input and returns the observation, Inputs are in the format 'agent_name, agent_input'.")
        ]
        self.tool = {tool.name: tool for tool in tools} if tools else {}
        self.COORDINATOR_SYSTEM_PROMPT = self._build_coordinator_prompt()

        self.coordinator_agent = Agent(
            "Coordinator_Agent",
            coordinator_model,
            tools=tools,
            CORE_SYSTEM_PROMPT=self.COORDINATOR_SYSTEM_PROMPT,
            SAFETY=False
        )

    
    def _build_coordinator_prompt(self) -> str:
        agent_desc = "\n".join(
            [f"- {desc['name']}: {desc['description']}" for desc in self.descriptions]
        )
        prompt = f"""
            You are a Coordinator agent managing multiple specialized agents. Each agent has specific capabilities described below:
            {agent_desc}

            Available Tools:
            - {"\n".join([f"- {t.name}: {t.description}" for t in self.tool.values()])}

            ### Rules:
            1. ONLY use tools listed above. If you do not need a tool to answer the question, or if no relevant tool is available, go straight to "Final Answer:".
            2. DO NOT hallucinate tools. If you use a tool, it MUST be one of: [{"\n".join([f"- {t.name}: {t.description}" for t in self.tool.values()])}].
            3. ALWAYS DELEGATE TASKS TO SUB MODELS
            4. ALWAYS use call_model tool with these two STRING inputs: 'agent_name, agent_input' where both are strings.
            5. YOU MUST CALL EXACTLY ONE AGENT AT A TIME.
            
            ### Format:
            Question: the input question you must answer
            Thought: logical reasoning to determine if an agent is needed.
            Action: the tool name, MUST be one of [{"\n".join([f"- {t.name}: {t.description}" for t in self.tool.values()])}]]
            Action Input: 
                        'agent_name, agent_input
                        Example:
                        Action Input: Agent_1, What is 2+2?'

            Observation: the result of the tool
            ... (Thought/Action/Action Input/Observation can repeat)
            Final Answer: the final answer to the original input question

            Begin!

            Question: 
            """
        
        return prompt
    

    def call_sub_agent(self, agent_name: str, agent_input: str) -> str:
        for i, desc in enumerate(self.descriptions):
            if desc["name"] == agent_name:
                agent = self.agents[i]

                print(Colors.CYAN + f"[MultiAgent] Delegating to {agent_name} with input: {agent_input}" + Colors.RESET)
                return agent.run(agent_input)
        raise ValueError(f"Agent '{agent_name}' not found.")
    
    def run(self, task: str, **gen_kwargs) -> str:
        """
        Runs the multi-agent coordinator.
        All routing and delegation happens via tools.
        """
        return self.coordinator_agent.run(task,max_new_tokens=64)
