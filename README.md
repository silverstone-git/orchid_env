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
  - map-reduce
---

# Orchid Env: Map-Reduce Orchestrator Evaluation Environment

Orchid Env is a specialized **OpenEnv** environment designed to benchmark Large Language Models on their ability to act as **Orchestrators** for complex, large-context data processing tasks.

Instead of directly solving a problem, the agent must design an execution pipeline:
1.  **Partition** a massive dataset (10,000+ lines) into chunks.
2.  **Delegate** processing to multiple sub-agents, providing each with custom Python extraction logic.
3.  **Synthesize** (Reduce) the outputs from all agents into a final, structured answer.

## 🏗 Architecture & Execution Flow

The environment leverages **Daytona SDK** to provide secure, isolated sandboxes for code execution.

### The Map-Reduce Pipeline
When an agent submits an `OrchidAction`:
1.  **Map Phase**: The environment loops through the list of `sub_agents`. For each agent:
    -   It slices the `dataset.log` according to `start_line` and `end_line`.
    -   It injects the slice into a variable called `chunk_data`.
    -   It executes the agent's `python_code` in the Daytona sandbox.
2.  **Reduce Phase**:
    -   All strings printed/returned by sub-agents are collected into a list called `sub_outputs`.
    -   The agent's `synthesis_code` is executed, which has access to `sub_outputs`.
3.  **Synthesis**: The final output of the synthesis script is what the environment grades.

## 🎯 Task Bank

All tasks utilize a synthetic 10,003-line system log file (`server/mock_system.log`) containing various log levels and embedded "Easter Egg" anomalies.

| Task ID | Description | Difficulty | Grader Logic |
| :--- | :--- | :--- | :--- |
| `extract_anomalies_easy` | Extract specific 'EASTER_EGG' string values. | Easy | Jaccard Similarity |
| `count_critical_medium` | Count total 'CRITICAL' logs in the dataset. | Medium | Distance-based Penalty |
| `extract_timestamps_hard` | Extract timestamps of anomalies in chronological order. | Hard | Sequential List Match |

## 🏆 Multi-Objective Reward Function

The reward signal is designed to guide the agent toward "wise" orchestration, not just correct answers.

### 1. Correctness (50%)
- Uses deterministic graders to provide **partial credit**.
- For lists: $Score = \frac{|Predicted \cap Truth|}{|Predicted \cup Truth|}$
- For counts: $Score = 1.0 - \frac{|Diff|}{Truth}$

### 2. Decomposition Efficiency (30%)
- **Chunking Quality**: Penalizes overlapping lines or lines skipped by the orchestrator.
- **Overhead Control**: Penalizes spawning too many agents (the "ideal" is ~1 agent per 2000 lines).
- **Load Balancing**: Encourages even distribution of data across sub-agents.

### 3. Prompt Quality (20%)
- A heuristic check for clear, actionable delegation instructions in the `role_prompt`.

### 4. Trajectory Penalty
- A decay of `-0.05` per step is applied to penalize brute-force trials or infinite loops.

## 🚀 Setup & Usage

### Prerequisites
- [Daytona API Key](https://www.daytona.io/)
- Python 3.10+ and `uv`

### Running the Server (Local)
```bash
export DAYTONA_API_KEY="your_key"
uv run python -m server.app
```

### Running the Server (Docker)
```bash
docker build -t orchid_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 -e DAYTONA_API_KEY="your_key" orchid_env-env:latest
```

### Gradio Web UI
When running the server with `ENABLE_WEB_INTERFACE=true`, a Gradio UI is available at `http://localhost:8000/web`.

#### How to use the UI:
1. **Click "Reset"** at the bottom of the page to start the episode and provision the sandbox.
2. The observation window will update with the first task (`extract_anomalies_easy`).
3. Fill in the **Action** form with the following example to achieve a perfect orchestration score:

**`chunking_strategy`**:
```text
Dividing 10,003 lines evenly across 5 agents to maximize parallel throughput.
```

**`sub_agents`** (Paste this exact JSON array):
```json
[
  {
    "role_prompt": "Extract EASTER_EGG_ERROR_CODE anomalies from logs using regex.",
    "start_line": 0,
    "end_line": 2000,
    "python_code": "import re, json; codes = re.findall(r'EASTER_EGG_ERROR_CODE:\\s*(0x[0-9A-Fa-f]+)', chunk_data); print(json.dumps(codes))"
  },
  {
    "role_prompt": "Extract EASTER_EGG_ERROR_CODE anomalies from logs using regex.",
    "start_line": 2000,
    "end_line": 4000,
    "python_code": "import re, json; codes = re.findall(r'EASTER_EGG_ERROR_CODE:\\s*(0x[0-9A-Fa-f]+)', chunk_data); print(json.dumps(codes))"
  },
  {
    "role_prompt": "Extract EASTER_EGG_ERROR_CODE anomalies from logs using regex.",
    "start_line": 4000,
    "end_line": 6000,
    "python_code": "import re, json; codes = re.findall(r'EASTER_EGG_ERROR_CODE:\\s*(0x[0-9A-Fa-f]+)', chunk_data); print(json.dumps(codes))"
  },
  {
    "role_prompt": "Extract EASTER_EGG_ERROR_CODE anomalies from logs using regex.",
    "start_line": 6000,
    "end_line": 8000,
    "python_code": "import re, json; codes = re.findall(r'EASTER_EGG_ERROR_CODE:\\s*(0x[0-9A-Fa-f]+)', chunk_data); print(json.dumps(codes))"
  },
  {
    "role_prompt": "Extract EASTER_EGG_ERROR_CODE anomalies from logs using regex.",
    "start_line": 8000,
    "end_line": 10003,
    "python_code": "import re, json; codes = re.findall(r'EASTER_EGG_ERROR_CODE:\\s*(0x[0-9A-Fa-f]+)', chunk_data); print(json.dumps(codes))"
  }
]
```

**`synthesis_code`**:
```python
import json
final_list = []
for out in sub_outputs:
    try:
        final_list.extend(json.loads(out))
    except:
        pass
print(final_list)
```

4. **Click "Step"** to execute your pipeline. You will see a breakdown of your Correctness, Decomposition, and Prompt scores!

### Baseline Inference
Evaluate an agent (e.g., Gemma) using the OpenAI-compatible endpoint:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="google/gemma-4-31B-it"
export API_KEY="your OPENAI_API_KEY or HF_TOKEN"

uv run python inference.py
```

## 🛠 Troubleshooting

### "Total disk limit exceeded" (Daytona)
If you see this error, it means orphaned sandboxes from previous crashed runs are filling your 30GB limit.
1.  Visit the [Daytona Dashboard](https://app.daytona.io/dashboard/).
2.  Manually delete or archive old workspaces.
3.  The environment now uses `auto_delete_interval=1` to prevent this in the future.

### "No module named pytest"
The environment uses standard Python libraries for Map-Reduce. If your `synthesis_code` requires a specific library, ensure it is available or installable via `pip` inside the sandbox logic.
