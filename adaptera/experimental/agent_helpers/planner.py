from typing import List, Literal, TypedDict, Optional
import json
from adaptera.tools.core import Tool
from adaptera.model.core import AdapteraModel
import regex as re




class Todo(TypedDict):
    id: str
    task: str
    tool: str
    args: str
    depends_on: List[str]
    status: Literal["pending", "in_progress", "completed"]


class PlanningState:
    def __init__(self):
        self.todos: List[Todo] = []


class Planner:
    def __init__(self, tools: list[Tool], planning_model: AdapteraModel, verbose: bool = False):
        self.tool_map = {tool.name: tool.description for tool in tools}
        self.tools = [
            {"name": tool.name, "description": tool.description}
            for tool in tools
        ]

        self.planning_model = planning_model
        self.state = PlanningState()
        self.verbose = verbose

    # -------------------------
    # State Management
    # -------------------------

    def add_todo(self, task: str, tool: Optional[str], args: Optional[str]):
        self.state.todos.append({
            "task": task,
            "tool": tool,
            "args": args,
            "status": "pending"
        })
    
    def update_status(self, index: int, status: Literal["pending", "in_progress", "completed"]):
        if 0 <= index < len(self.state.todos):
            self.state.todos[index]["status"] = status
        else:
            raise IndexError("Todo index out of range.")

    def reset(self):
        self.state.todos.clear()

    def next_pending(self) -> Optional[int]:
        for i, todo in enumerate(self.state.todos):
            if todo["status"] == "pending":
                return i
        return None

    def all_completed(self) -> bool:
        return all(todo["status"] == "completed" for todo in self.state.todos)

    # -------------------------
    # Prompt
    # -------------------------

    def generate_planner_prompt(self, user_query: str) -> str:
        tools_str = json.dumps(self.tools, indent=2)

        example_json = {
            "steps": [
                {
                    "id": "step1",
                    "task": "Add 5 and 3",
                    "tool": "add",
                    "args": "5,3",
                    "depends_on": []
                },
                {
                    "id": "step2",
                    "task": "Multiply result by 2",
                    "tool": "multiply",
                    "args": "RESULT_FROM_step1,2",
                    "depends_on": ["step1"]
                }
            ]
        }
        
        example_str = json.dumps(example_json, indent=2)
        return f"""You are a task planner.

Break the user query into ordered steps using the available tools.
PLEASE USE PEMDAS LOGIC WHEN BREAKING DOWN THE TASKS and ARRANGING THEM IN PROPER ORDER.
DO NOT WRITE THE BROKEN DOWN STEPS IN THE OUTPUT, ONLY FILL IN THE EXPECTED JSON FORMAT.
YOU MUST NOT REPLY WITH ANYTHING ELSE.

USE THE GIVEN TOOLS AVAILABLE TO YOU.
Available Tools:
{tools_str}

Output strictly in this JSON format:

{example_str}


Rules:
- Every step MUST use one of the provided tools.
- tool_name MUST exactly match a provided tool.
- Arguments MUST be a comma-separated string with exactly two values.
- null is NOT allowed.
- Do not describe the action abstractly. Use the tool.
- Output only one JSON object.
- YOU MUST STOP IMMEDIATELY AFTER '{"}"}' AND DO NOT ADD ANY EXTRA TEXT AFTER IT.
- Output **only one JSON object**. Do not write anything else. Do not add extra {"{}"} or text. Stop immediately after the closing brace.
- VALUES OF args MUST ONLY BE "RESULT_FROM_STEP_X" if there is a previous step related to it or some new args extracted from the user query.
- ONE STEP CAN ONLY DEPEND ON OTHER IF AND ONLY IF IT USES THE RESULT OF THAT STEP AS ARGUMENTS. DO NOT DEPEND ON A STEP IF YOU ARE NOT USING ITS RESULT.
- DO NOT SOLVE ANYTHING , YOUR JOB IS JUST TO PASS THE RAW ARGUMENTS INTO THE JSON FORMAT
Return ONLY the JSON object.

User Query:
{user_query}

JSON:
"""

    # -------------------------
    # Balanced JSON Extraction
    # -------------------------

    def _extract_first_json(self, text: str) -> Optional[str]:
        brace_count = 0
        start = None

        for i, ch in enumerate(text):
            if ch == "{":
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    return text[start:i + 1]

        return None

    # -------------------------
    # Plan
    # -------------------------

    def plan(self, user_query: str) -> dict:
        prompt = self.generate_planner_prompt(user_query)

        response = self.planning_model.generate(
            prompt,
            do_sample=False,
            max_new_tokens=500,
        )

        if self.verbose:
            print("Planner raw response:\n", response)

        json_str = self._extract_first_json(response)
        if not json_str:
            raise ValueError("Planner failed: No JSON found.")

        try:
            plan = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Planner JSON decode error: {e}")

        validated_plan = self._validate(plan)

        self.reset()

        for step in validated_plan:
            self.state.todos.append({
                **step,
                "status": "pending"
            })

        return validated_plan

    # -------------------------
    # Validation
    # -------------------------

    def _validate(self, plan: dict) -> dict:
        if "steps" not in plan:
            raise ValueError("Missing 'steps' key.")

        validated = []
        ids = set()

        for step in plan["steps"]:
            step_id = step["id"]

            if step_id in ids:
                raise ValueError("Duplicate step id.")

            ids.add(step_id)

            if step["tool"] not in self.tool_map:
                raise ValueError(f"Unknown tool {step['tool']}")

            validated.append(step)

        return validated