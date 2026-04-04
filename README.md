# Adaptera 🌌

A local-first LLM orchestration library with native support for Hugging Face, PEFT/LoRA, QLoRA — without hiding the model and giving advanced users the full control.

Adaptera is a thin orchestration layer — not an abstraction barrier.
---

## Why Adaptera?

Use Adaptera if you:
- Want full control over your model (no hidden layers)
- Prefer explicit agent logic over auto-magic chains
- Are working locally with Hugging Face / quantized models

Avoid Adaptera if you:
- Want plug-and-play integrations (use LangChain)
- Need production-ready pipelines out of the box

> **Note:**
## ⚠️ Status

- Early development — APIs may change
- Currently focusing on agentic systems (v0.1.3)

## ⚠️ LM Studio Support

- Inference only
- Agent integration coming soon

## 🤝 Contributing

- Contributions welcome
- Please avoid low-quality / spam PRs
---
## Features

- **Local-First**: Built for running LLMs on your own hardware efficiently.
- **Native PEFT/QLoRA**: Seamless integration with Hugging Face's PEFT for efficient model loading.
- **Persistent Memory**: Vector-based memory using FAISS with automatic text embedding (SLM).
- **Strict ReAct Agents**: Deterministic agent loops using JSON-based tool calls.
- **Model Transparency**: Easy access to the underlying Hugging Face model and tokenizer.

## Installation

### Using python
```bash
pip install adaptera
```

### Using Anaconda/Miniforge
```bash
conda activate < ENV NAME >
pip install adaptera
```

*(Note: Requires Python 3.12+)*

## Quick Start

```python
#setup imports and optional vector DB
from adaptera import Agent, AdapteraHFModel, VectorDB, Tool

#Optional
db = VectorDB()
db.add("Information to be added into the vector db to be checked, it will automatically searched up by the model when generating responses")
```

```python
model = AdapteraHFModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit",
    vector_db=db #optional
)

model.generate("What is an apple?")

# OR

model = AdapteraLMSModel()
model.generate("What is an apple")
```

```python

#define functions for the agent
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
```

```python
#Setup and run the agent
agent = Agent(
    "AddSubtract_Agent",
    model, 
    tools=tools,
    description = "An agent for only addition and subtraction tasks."
)

agent.run("what is 1+1?")
```

```python
# MULTI AGENT setup
# The coordinator model routes tasks between agents based on their descriptions.
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
print("Final Response from MultiAgent System: ",response)
```

## Project Structure

- `adaptera/chains/`: Agentic workflows and ReAct implementations.
- `adaptera/model/`: Hugging Face model loading and generation wrappers.
- `adaptera/memory/`: FAISS-backed persistent vector storage.
- `adaptera/tools/`: Tool registry and definition system.
- `adaptera/experimental/`: Experimental features do not use these in production , these will be changed / deleted later on.

## Non-goals

This library does not aim to be a full ML framework or replace existing tools like LangChain. It focuses on providing a clean, minimal interface for local-first LLM orchestration.

## Notes for devlopers
in order to complie adaptera into a python package , please run:
```bash
python -m build
```