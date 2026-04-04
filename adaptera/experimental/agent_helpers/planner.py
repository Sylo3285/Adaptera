from typing import List, Literal, TypedDict, Optional
import json
from adaptera.tools.core import Tool
from adaptera.model.core import AdapteraHFModel, AdapteraLMSModel


# -------------------------
# Types
# -------------------------

class Todo(TypedDict):
    id: str
    task: str
    tool: str
    args: str
    depends_on: List[str]
    status: Literal["pending", "completed"]


class PlanningState:
    def __init__(self):
        self.todos: List[Todo] = []


# -------------------------
# Planner
# -------------------------

class Planner:
    def __init__(
        self,
        tools: List[Tool],
        planning_model: AdapteraHFModel | AdapteraLMSModel,
        formatting_model: AdapteraHFModel | AdapteraLMSModel,
        verbose: bool = False
    ):
        # Add finish tool
        self.tools = [
            {"name": tool.name, "description": tool.description}
            for tool in tools
        ] + [
            {"name": "finish", "description": "Marks the task as complete"}
        ]

        self.tool_map = {t["name"]: t["description"] for t in self.tools}

        self.planning_model = planning_model
        self.formatting_model = formatting_model

        self.state = PlanningState()
        self.verbose = verbose

    # -------------------------
    # State
    # -------------------------

    def reset(self):
        self.state.todos.clear()

    def is_finished(self) -> bool:
        return len(self.state.todos) > 0 and self.state.todos[-1]["tool"] == "finish"

    # -------------------------
    # Prompts
    # -------------------------

    def generate_planning_prompt(self, user_query: str) -> str:
        history_str = json.dumps(self.state.todos, indent=2)

        return f"""
            You are a planner.

            Your job is to think step-by-step and decide the NEXT action.

            User Query:
            {user_query}

            Current Steps:
            {history_str}

            Instructions:
            - Think freely
            - Decide the NEXT step only
            - Do NOT output JSON
            - Be concise
            - Describe the plan in detail for the formatter later on YOU MUST NOT SOLVE THE PROBLEMS , JUST TELL THE FORMATTER WHICH VALUE GOES WHERE

            Output:
        """

    def generate_formatting_prompt(self, plan_text: str) -> str:
        tools_str = json.dumps(self.tools, indent=2)

        return f"""
            You are a JSON compiler.

            Your ONLY job is to convert the given plan into JSON.

            You are NOT allowed to:
            - add new steps
            - remove steps
            - explain anything
            - output multiple JSON blocks

            ----------------------

            PLAN:
            {plan_text}

            ----------------------

            AVAILABLE TOOLS:
            {tools_str}

            ----------------------

            OUTPUT RULES:

            - Output EXACTLY one JSON object
            - Format:
            {{
            "id": "stepX",
            "task": "...",
            "tool": "...",
            "args": "a,b",
            "depends_on": []
            }}

            - Choose ONLY the NEXT step from the plan
            - DO NOT generate multiple steps
            - DO NOT wrap in markdown
            - DO NOT explain anything
            - DO NOT write text before or after JSON

            If you fail → output is INVALID.

            OUTPUT:
        """

    # -------------------------
    # JSON Extraction
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
    # Validation
    # -------------------------

    def _validate_step(self, step: dict):
        required_keys = ["id", "task", "tool", "args", "depends_on"]

        for key in required_keys:
            if key not in step:
                raise ValueError(f"Missing key: {key}")

        if step["tool"] not in self.tool_map:
            raise ValueError(f"Unknown tool: {step['tool']}")

        existing_ids = {t["id"] for t in self.state.todos}
        if step["id"] in existing_ids:
            raise ValueError("Duplicate step id")

        for dep in step["depends_on"]:
            if dep not in existing_ids:
                raise ValueError(f"Invalid dependency: {dep}")

        if not isinstance(step["args"], str) or "," not in step["args"]:
            raise ValueError("Args must be comma-separated string")

        if len(step["args"].split(",")) != 2:
            raise ValueError("Args must contain exactly two values")

    # -------------------------
    # Core Step Planning
    # -------------------------

    def _plan_next(self, user_query: str) -> dict:
        # ---- STEP 1: Reasoning ----
        planning_prompt = self.generate_planning_prompt(user_query)

        if type(self.planning_model) == AdapteraHFModel:
            plan_text = self.planning_model.generate(
                planning_prompt,
                do_sample=True,
                max_new_tokens=200,
            )
        elif type(self.planning_model) == AdapteraLMSModel:
            plan_text = self.planning_model.generate(planning_prompt)

        if self.verbose:
            print("\n[PLAN SCRATCHPAD]\n", plan_text)

        # ---- STEP 2: Formatting (with retry) ----
        formatting_prompt = self.generate_formatting_prompt(plan_text)

        for attempt in range(2):
            if type(self.formatting_model) == AdapteraHFModel:
                response = self.formatting_model.generate(
                    formatting_prompt,
                    do_sample=False,
                    max_new_tokens=150,
                )
            elif type(self.formatting_model) == AdapteraLMSModel:
                response = self.formatting_model.generate(formatting_prompt)


            if self.verbose:
                print("\n[FORMATTED OUTPUT]\n", response)

            json_str = self._extract_first_json(response)

            if json_str:
                try:
                    step = json.loads(json_str)
                    self._validate_step(step)
                    return step
                except Exception:
                    pass

            # Retry hint
            formatting_prompt += "\nYour previous output was invalid. Fix it."

        raise ValueError("Formatter failed after retries.")

    # -------------------------
    # Public API
    # -------------------------

    def plan(self, user_query: str, max_steps: int = 10) -> List[Todo]:
        self.reset()

        for i in range(max_steps):
            step = self._plan_next(user_query)

            # enforce sequential IDs (override model stupidity)
            step["id"] = f"step{i+1}"

            self.state.todos.append({
                **step,
                "status": "completed"
            })

            if step["tool"] == "finish":
                break

        return self.state.todos