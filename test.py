from adaptera import AdapteraModel,VectorDB,Agent,Tool
from adaptera.experimental.multiagent import MultiAgent

"""
This is a test script to demonstrate a multi-agent system using Adaptera.
This was used in the development of this project and attempts to show best approaches.

Things in this script may change or be removed in future versions.
Things also may not be documented fully here as this is a test script.
Certain parts of this script may be experimental and not part of the main Adaptera package.

"""

db = VectorDB()

model = AdapteraModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit",
    vector_db=db
    )

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

def divide(a,b):
    "Divides a by b"
    print(f"Dividing {a} by {b} via tool call")
    if b == 0:
        return "Error: Division by zero"
    return a / b

tools1 = [
    Tool(name="add", func=add, description="Adds two numbers together. Input should be in the format: 'a,b' where a and b are numbers."),
    Tool(name="subtract", func=subtract, description="Subtracts b from a. Input should be in the format: 'a,b' where a and b are numbers.")
] 

tools2 = [
    Tool(name="multiply", func=multiply, description="Multiplies two numbers together. Input should be in the format: 'a,b' where a and b are numbers."),
    Tool(name="divide", func=divide, description="Divides a by b. Input should be in the format: 'a,b' where a and b are numbers."),
]

agent1 = Agent(
    "AddSubtract_Agent",
    model, 
    tools=tools1,
    description = "An agent for only addition and subtraction tasks."
)

agent2 = Agent(
    "MultiplyDivide_Agent",
    model, 
    tools=tools2,
    description = "An agent only for multiplication and division tasks."
)

#create a multi agent system

MOA = MultiAgent(agents=[agent1,agent2],coordinator_model=model)
response = MOA.run("What is the result of (15 + 5) * (10 - 2)?")
print("Final Response from MultiAgent System:")
print(response)