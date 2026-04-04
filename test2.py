from adaptera.model.core import AdapteraHFModel , AdapteraLMSModel
from adaptera.tools.core import Tool
from adaptera.experimental.experimental_agent import Agent

def test_planner():
    """model = AdapteraHFModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit"
    )"""

    model = AdapteraLMSModel()


    def add(a,b):
        "Adds 2 numbers together"
        print(f"Adding {a} and {b} via tool call")
        return a + b

    def subtract(a,b):
        "Subtracts b from a"
        print(f"Subtracting {b} from {a} via tool call")
        return a - b
    
    def multiply(a,b):
        "Multiplies 2 numbers together"
        print(f"Multiplying {a} and {b} via tool call")
        return a * b
    
    tools1 = [
        Tool(name="add", func=add, description="returns sum of 2 numbers. Input should be in the format: 'a,b' where a and b are numbers."),
        Tool(name="subtract", func=subtract, description="Subtracts b from a. Input should be in the format: 'a,b' where a and b are numbers."),
        Tool(name="multiply", func=multiply, description="Multiplies 2 numbers together. Input should be in the format: 'a,b' where a and b are numbers.")
    ]

    agent = Agent(llm_name="TestAgent", llm=model,planning_model=model, tools=tools1, description="An Agent for addition and subtraction tasks.")
    query = "What is the sum of 5 and 3 then multiplied by the difference between 10 and 4?"
    response = agent.run(query)
    print("Final response from agent:", response)

test_planner()