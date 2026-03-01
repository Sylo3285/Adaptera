# Adaptera 🌌

A local-first LLM orchestration library with native support for Hugging Face, PEFT/LoRA, QLoRA — without hiding the model and giving advanced users the full control.

---
> **Note:** This project is in its early development phase and may undergo significant changes. However, the core goal of providing local LLM processing will remain consistent. Once the agentic part of the module is stable, we will work on making a fine-tuner for it so that this library can be used as a quick way of prototyping local agentic models.
> 
> Feel free to contribute, please do not spam pull requests. Any and all help is deeply appreciated.
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
from adaptera import Agent, AdapteraModel, VectorDB, Tool
db = VectorDB()
```

```python
model = AdapteraModel(
    model_name ="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit",
    vector_db=db
)

model.generate("What is an apple?")
```

```python
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
agent = Agent(
    "AddSubtract_Agent",
    model, 
    tools=tools,
    description = "An agent for only addition and subtraction tasks."
)

agent.run("what is 1+1?")
```

## Project Structure

- `adaptera/chains/`: Agentic workflows and ReAct implementations.
- `adaptera/model/`: Hugging Face model loading and generation wrappers.
- `adaptera/memory/`: FAISS-backed persistent vector storage.
- `adaptera/tools/`: Tool registry and definition system.
- `adaptera/experimental/`: Experimental features 

## Non-goals

This library does not aim to be a full ML framework or replace existing tools like LangChain. It focuses on providing a clean, minimal interface for local-first LLM orchestration.
