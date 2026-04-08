---
title: Orchid Env - Map-Reduce Orchestrator Evaluation
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - multi-agent
  - big-data
---

# Orchid Env - Map-Reduce Orchestrator Evaluation Environment

Orchid Env is a sophisticated OpenEnv environment designed to evaluate RL agents on their ability to **orchestrate multi-agent workflows**. Instead of solving tasks directly, the agent acts as an **Orchestrator** that must divide a massive dataset into chunks, delegate processing to sub-agents (running in isolated Daytona sandboxes), and synthesize the results.

## Motivation
Real-world AI engineering often involves managing context window limits and parallelizing data processing. Orchid Env benchmarks an agent's "wisdom" in task decomposition, delegation efficacy, and load balancing across multiple execution environments.

## Environment Details

### Action Space
**OrchidAction**:
- `chunking_strategy` (str): Natural language explanation of the breakdown logic.
- `sub_agents` (List[SubAgentConfig]): 
    - `role_prompt`: Instructions for the sub-agent.
    - `start_line` / `end_line`: The slice of the dataset to process.
    - `python_code`: The script the sub-agent runs to extract data from its chunk.
- `synthesis_code` (str): Python script to reduce the list of sub-agent outputs into a final answer.

### Observation Space
**OrchidObservation**:
- `task_description` / `dataset_lines`: Context for the current "Big Data" task.
- `correctness_score` (float): Accuracy of the final synthesized output (0.0 - 1.0).
- `decomposition_score` (float): Efficiency of chunking (penalizes overlap, missing lines, and sub-agent overhead).
- `prompt_score` (float): Heuristic quality of the delegation instructions.
- `reward` (float): Weighted RL signal.
- `done` (bool): True when all tasks in the episode are complete.

### Task Bank (10,000+ line System Logs)
| Task ID | Objective | Difficulty | Grader Type |
| :--- | :--- | :--- | :--- |
| `extract_anomalies_easy` | Extract 'EASTER_EGG' error codes. | Easy | Jaccard Similarity |
| `count_critical_medium` | Count total 'CRITICAL' logs in the file. | Medium | Distance Penalty |
| `extract_timestamps_hard` | Extract precise timestamps for all anomalies. | Hard | Chronological List |

## Reward Function
The environment provides a dense, multi-objective reward:
- **Correctness (50%)**: Rewards partial progress (e.g., finding 2/3 items).
- **Decomposition Efficiency (30%)**: Rewards optimal chunk sizes (~2000 lines) and load balancing. Penalizes overlapping or skipped data.
- **Prompt Quality (20%)**: Rewards clear, actionable sub-agent instructions.
- **Trajectory Decay**: `-0.05 * step_count` to penalize inefficient trials.

## Setup & Usage

### 1. Installation
```bash
git clone <repo-url>
cd orchid_env
uv sync
```

### 2. Build and Run Server (Docker)
The environment uses a lightweight Python 3.11-slim image:
```bash
docker build -t orchid_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 -e DAYTONA_API_KEY="your_key" orchid_env-env:latest
```

### 3. Running the Baseline Inference
To test an agent (using Gemini or any OpenAI-compatible API):
```bash
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-1.5-flash"
export API_KEY="your_api_key"

uv run python inference.py
```

## Quick Start Client Example
```python
from client import OrchidEnv
from models import OrchidAction, SubAgentConfig

async with OrchidEnv(base_url="http://localhost:8000") as env:
    obs = await env.reset()
    # Define a Map-Reduce plan
    action = OrchidAction(
        chunking_strategy="Split 10k lines into 5 agents",
        sub_agents=[SubAgentConfig(start_line=0, end_line=2000, python_code="...", role_prompt="...")],
        synthesis_code="print(sub_outputs)"
    )
    result = await env.step(action)
    print(f"Orchestration Reward: {result.reward}")
```
