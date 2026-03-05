class DAGScheduler:
    def __init__(self, planner, tools, llm, verbose=False):
        self.planner = planner
        self.tools = tools
        self.llm = llm
        self.memory = {}
        self.verbose = verbose

    def _resolve_args(self, args: str):
        for step_id, result in self.memory.items():
            placeholder = f"RESULT_FROM_{step_id}"
            args = args.replace(placeholder, str(result))
        return args

    def _get_executable_nodes(self):
        executable = []

        for step in self.planner.state.todos:
            if step["status"] != "pending":
                continue

            if all(
                dep in self.memory
                for dep in step["depends_on"]
            ):
                executable.append(step)

        return executable

    def run(self, max_iterations=50):
        iterations = 0

        while True:
            if iterations > max_iterations:
                raise RuntimeError("Max DAG iterations reached.")

            executable = self._get_executable_nodes()

            if not executable:
                break

            for step in executable:
                step["status"] = "in_progress"

                try:
                    args = self._resolve_args(step["args"])
                    tool = self.tools[step["tool"]]
                    result = tool.run(args)

                    self.memory[step["id"]] = result
                    step["status"] = "completed"

                    if self.verbose:
                        print(f"Executed {step['id']} → {result}")

                except Exception as e:
                    step["status"] = "pending"
                    raise RuntimeError(f"{step['id']} failed: {str(e)}")

            iterations += 1

        # Cycle detection
        unfinished = [
            s for s in self.planner.state.todos
            if s["status"] != "completed"
        ]

        if unfinished:
            raise RuntimeError("DAG execution incomplete. Possible cycle.")

        return self.memory