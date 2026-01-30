from adaptera import AdapteraModel,VectorDB,Agent,Tool

db = VectorDB()
#db.add("Hello, this is a memory entry", metadata=["greeting"])


model = AdapteraModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit"
    )

def add(a,b):
    "Adds 2 numbers together"
    return a + b

tools = [
    Tool(name="add", func=add, description="Adds two numbers together. Input should be in the format: 'a,b' where a and b are numbers.")
]

agent = Agent(model)#,tools=tools)
del model

print(agent.run("What is the sum of 2 and 5",max_new_tokens=50,min_new_tokens=10))
print(agent.run("What is an apple? Give me a short answer.",max_new_tokens=50,min_new_tokens=10))