# TO DO on STREAM-

- Add an about section      |✅
- Update the Readme.md file |✅

- Optimize/Overhaul agent class|❌ (IN PROGRESS)
- Add Stricter parsing      |❌ (IN PROGRESS)

# Things that have beem done:
- Added LM Studio Model support
- Added about.py which will guide people to a discord server for community help if needed and to my youtube channel can be used via import adaptera ; adaptera.about()
- Depreciated AdapteraModel
- Added experimental agent and multiagent inside the experimental/ dir



# other plans

```bash
User Goal
   ↓
Planner Module
   ↓
Task Graph (DAG)
   ↓
Execution Engine
   ↓
Observation Aggregator
   ↓
Verifier
   ↓
Replan if necessary


Agent
 ├── Planner
 ├── Executor
 ├── Verifier
 └── Scheduler
```