from typing import List, Literal, TypedDict
from adaptera.tools.core import Tool


class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]


class PlanningState:
    def __init__(self):
        self.todos: List[Todo] = []


class Planner:
    def __init__(self, tools: list[Tool]):
        self.tools = []
        for tool in tools:
            self.tools.append({
                "name": tool.name,
                "description": tool.description
            })

        self.state = PlanningState()

    def add(self, content: str) -> None:
        self.state.todos.append({
            "content": content,
            "status": "pending"
        })

    def update_status(
        self,
        index: int,
        new_status: Literal["pending", "in_progress", "completed"]
    ) -> None:
        if 0 <= index < len(self.state.todos):
            self.state.todos[index]["status"] = new_status
        else:
            raise IndexError("Todo item index out of range.")

    def next_pending(self) -> int | None:
        for i, todo in enumerate(self.state.todos):
            if todo["status"] == "pending":
                return i
        return None

    def all_completed(self) -> bool:
        return all(todo["status"] == "completed" for todo in self.state.todos)

    def reset(self):
        self.state.todos.clear()

    def generate_prompt(self) -> str:
        prompt = f"""
        You are the task decision engine for an agent.

        Current task: {self.state.todos[self.next_pending()]['content']}
        Current task status: {self.state.todos[self.next_pending()]['status']}

        Available tools: {self.tools}

        Decide:
        - Should a tool be used?
        - Which tool?
        - What is the input?
        - Optional: Can the task be completed directly without a tool?

        Return a JSON object like:
        {{
        "use_tool": true,
        "tool_name": "add",
        "tool_input": "1,2",
        "complete_directly": false
        }}
    """