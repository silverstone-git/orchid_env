---
title: Orchid Env - Code Fix Evaluation
emoji: 💿
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - code-generation
---

# Orchid Env - Code Fix Evaluation Environment

Orchid Env is a real-world task simulation environment for evaluating LLM-based RL agents on code-fixing tasks. It provides a series of broken Python functions and requires the agent to submit code fixes. These fixes are evaluated inside isolated **Daytona** sandboxes using **Pytest**.

## Motivation
Evaluating coding agents requires more than just static analysis. Orchid Env provides a dynamic "Gymnasium-style" interface where agents receive immediate feedback from test executions, allowing for reinforcement learning and iterative improvement in a secure, isolated environment.

## Environment Details

### Action Space
**OrchidAction**:
- `task_id` (str): ID of the task being attempted.
- `code_submission` (str): The full Python code fix submitted by the agent.
- `agent_id` (str): Identifier for the agent (e.g., model name).

### Observation Space
**OrchidObservation**:
- `task_id` (str): ID of the **next** task to be attempted.
- `task_description` (str): Human-readable description of the next task.
- `broken_code` (str): The original broken code for the next task.
- `execution_output` (str): Full pytest output from the **previous** task's execution.
- `tests_passed` (int): Number of tests passed in the previous task.
- `tests_total` (int): Total number of tests in the previous task.
- `score` (float): Correctness of the previous task (passed/total).
- `reward` (float): The RL reward signal for the previous step.
- `done` (bool): True if all tasks in the bank have been completed.

### Tasks & Difficulty
| Task ID | Description | Difficulty |
| :--- | :--- | :--- |
| `fix_off_by_one` | Fix a range loop that skips the last element. | Easy |
| `fix_type_error` | Ensure a function raises `ZeroDivisionError` correctly. | Medium |
| `fix_logic_bug` | Remove a spurious character from a palindrome checker. | Easy |
| `fix_missing_return` | Add a missing return statement in a recursive function. | Medium |

### Reward Function
The environment provides a dense reward signal:
- **Base Reward**: `tests_passed / tests_total` (0.0 to 1.0).
- **Zero Progress Penalty**: `-0.2` if no tests pass.
- **Runtime Error Penalty**: `-0.3` if the code generates a Python `Traceback`.
- **Step Penalty**: `-0.05 * step_count` to discourage inefficient/random trials.
- **Total Reward**: Clamped between `-1.0` and `1.0`.

## Baseline Scores
Evaluated using `gemini-3.1-flash-lite-preview`:
- **Average Correctness**: 1.00 (All tasks solved)
- **Total Reward**: ~3.50 (After step penalties and deductions)

## Setup & Usage

### Prerequisites
- [Daytona API Key](https://www.daytona.io/)
- Python 3.10+ and `uv`

### Installation
```bash
git clone <repo-url>
cd orchid_env
uv sync
```

### Running the Server
```bash
export DAYTONA_API_KEY="your_api_key_here"
uv run python -m server.app
```
The server will start at `http://localhost:8000`.

### Running the Baseline Inference
To run the evaluation using Gemini's OpenAI-compatible endpoint:
```bash
export GEMINI_API_KEY="your_api_key"
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-1.5-flash"
export API_KEY="${GEMINI_API_KEY}"

uv run python inference.py
```

## Quick Start Client Example
```python
import asyncio
from client import OrchidEnv
from models import OrchidAction

async def main():
    async with OrchidEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        print(f"Task: {obs.observation.task_description}")
        
        result = await env.step(OrchidAction(
            task_id=obs.observation.task_id,
            code_submission="def sum_list(nums): return sum(nums)"
        ))
        print(f"Reward: {result.reward}")

if __name__ == "__main__":
    asyncio.run(main())
```
