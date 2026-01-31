from adaptera import AdapteraModel,VectorDB,Agent,Tool

db = VectorDB()
"""db.add("Hello, this is a memory entry")
db.add("You were made by Sylo.")
"""
model = AdapteraModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="8bit",
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

tools = [
    Tool(name="add", func=add, description="Adds two numbers together. Input should be in the format: 'a,b' where a and b are numbers."),
    Tool(name="subtract", func=subtract, description="Subtracts b from a. Input should be in the format: 'a,b' where a and b are numbers.")
]

agent = Agent(model, tools=tools)
del model

print(agent.run("What is the sum of 2 and 5",max_new_tokens=50,min_new_tokens=10))
print(agent.run("What is the result of subtracting 10 from 25",max_new_tokens=50,min_new_tokens=10))
print(agent.run("What is an apple? Give me a long answer.",max_new_tokens=50,min_new_tokens=10))
print(agent.run("What is 15 plus 30 minus 10?",max_new_tokens=50,min_new_tokens=10))
print(agent.run("who are you?",max_new_tokens=50,min_new_tokens=10))

print(agent.run("who made you?",max_new_tokens=50,min_new_tokens=10))