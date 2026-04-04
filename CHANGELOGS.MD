# This file contains logs for each update of adaptera:

---

# V0.1
- Added core model
- Added core agent (ReACT loop)
- Added tools class for the agent sub-calling
- Added VectorDB using Faiss

# V0.1.1
- Fixed Readme for PyPI

# V0.1.2
- Added experimental multiagent system (experimental/multiagent.py)
- Added experimental branching into the files under **adaptera/experimental/**

# V0.1.3 (CURRENT IN DEVLOPEMENT)
- Moved multiagent to adaptera/chains making it officially a part of agent system.
- Added **EXPERIMENTAL** DAG for agent inside experimental/experimental_agent.py.
- Will start to improve the multiagent.py inside the new experimental/experimental_multiagent.py after the experimental/experimental_agent.py is stable
- Added LM Studio Model support
- Added about.py which will guide people to a discord server for community help if needed and to my youtube channel can be used via-
```python
import adaptera
adaptera.about()
```
- Depreciated AdapteraModel in exchange for AdapteraHFModel and AdapteraLMSModel
- Added experimental agent and multiagent inside the experimental/ dir 