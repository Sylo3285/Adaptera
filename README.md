# Adaptera 🌌

A local-first LLM orchestration library with native support for Hugging Face, PEFT/LoRA, QLoRA, and API models — without hiding the model.

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
---
> **As of now**
- Python module for this has not been released yet, install using github like this:
---
### Using pip
```bash
# Clone the repository
git clone https://github.com/Sylo3285/Adaptera
cd Adaptera

# Install dependencies in editable mode
pip install .
```

### Using Anaconda/Miniforge
```bash
conda activate < ENV NAME >

# Clone the repository
git clone https://github.com/Sylo3285/Adaptera
cd Adaptera

# Install dependencies in editable mode
pip install .
```

*(Note: Requires Python 3.12+)*

## Quick Start

```python
from adaptera import Agent, AdapteraModel, VectorDB, Tool

# 1. Initialize Vector Memory
db = VectorDB(index_file="memory.index")

# 2. Load a Model (with 4-bit quantization)
model = AdapteraModel(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    quantization="4bit",
    vector_db=db
)

# 3. Define Tools
def add(a, b):
    """Adds two numbers together"""
    return a + b

tools = [
    Tool(name="add", func=add, description="Adds two numbers together. Input: 'a,b'")
]

# 4. Create and Run Agent
agent = Agent(model, tools=tools)
print(agent.run("What is 15 + 27?"))
```

## Project Structure

- `adaptera/chains/`: Agentic workflows and ReAct implementations.
- `adaptera/model/`: Hugging Face model loading and generation wrappers.
- `adaptera/memory/`: FAISS-backed persistent vector storage.
- `adaptera/tools/`: Tool registry and definition system.

## Non-goals

This library does not aim to be a full ML framework or replace existing tools like LangChain. It focuses on providing a clean, minimal interface for local-first LLM orchestration.
