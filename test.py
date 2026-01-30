from adaptera.model.core import AdapteraModel
from adaptera.memory.core import VectorDB
from adaptera.chains.agent import Agent
from adaptera.tools.registry import Tool

db = VectorDB()
#db.add("Hello, this is a memory entry", metadata=["greeting"])


model = AdapteraModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit", 
    vector_db=db
    )

def add(a,b):
    "Adds 2 numbers together"
    return a + b

tools = [
    Tool(name="add", func=add, description="Adds two numbers together. Input should be in the format: 'a,b' where a and b are numbers.")
]

agent = Agent(model)#,tools=tools)
del model

print(agent.run("What is the sum of 2 and 5"))
print(agent.run("What is the definition of an apple?"))
print(agent.run("Write a poem about apples"))