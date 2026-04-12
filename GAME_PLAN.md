# GAME_PLAN: The "Data Forge" Orchestrator Game

## 1. Concept & RL Framing
We will reframe the Big Data Orchestrator benchmark into a turn-based programming and architecture puzzle game called **"Data Forge"**. 

Instead of treating the entire task bank as a single continuous episode where the agent gets one shot per task, **one episode = one task**. 

The agent (the "Orchestrator") is given a massive dataset and a specific extraction goal. It has a limited number of "deployments" (attempts) to write and execute the perfect Map-Reduce pipeline. 

If the pipeline crashes or returns the wrong answer, the agent receives detailed diagnostic feedback (tracebacks, chunk overlap warnings, partial credit scores) and can try again until it runs out of attempts. This creates a true **RL loop with trial-and-error debugging**, perfectly matching the `WordGameEnvironment` pattern from `notebook.md`.

## 2. Type Definitions (Dataclasses)
To match the clean, framework-agnostic style of `notebook.md`, we will drop Pydantic for the core environment interface and use pure Python `@dataclass`. This ensures the inference script sees the environment exactly as an RL agent would during training.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class SubAgentDeploy:
    """A worker agent deployed to process a specific chunk of data."""
    start_line: int
    end_line: int
    role_prompt: str
    python_code: str

@dataclass
class OrchestratorAction:
    """The player submits a full Map-Reduce architecture."""
    chunking_strategy: str
    sub_agents: List[SubAgentDeploy]
    synthesis_code: str

@dataclass
class OrchestratorObservation:
    """What the player sees after deploying their architecture."""
    done: bool
    reward: Optional[float]
    
    # Task Context (remains static during the episode)
    task_description: str
    dataset_path: str
    dataset_lines: int
    dataset_sample: str
    
    # Game State (evolves during the episode)
    attempts_remaining: int
    
    # Feedback from the last deployment
    message: str                 # High-level feedback ("Pipeline crashed!", "Partial success.")
    execution_output: str        # The actual stdout/stderr from the synthesis code
    sub_agent_errors: int        # Number of workers that threw exceptions
    
    # Partial Credit Breakdown (from the Grader)
    correctness_score: float
    decomposition_score: float
    prompt_score: float

    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestratorState:
    """Internal episode metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    max_attempts: int = 5
```

## 3. The RL Loop & Grading Mechanics

The `step(action: OrchestratorAction)` function will execute the map-reduce job and calculate the scores just like the current environment, but the transition dynamics change to support an iterative game loop:

1. **Map-Reduce Execution**: The environment spawns local processes for each `SubAgentDeploy`, gathers outputs, and runs `synthesis_code`.
2. **Partial Grading**: The environment calculates:
   - `correctness_score`: Regex/difflib matching against ground truth.
   - `decomposition_score`: Penalizes overlapping line numbers, missing lines, or severe sub-agent imbalances.
   - `prompt_score`: Keyword overlap between task description and agent prompts.
3. **Win/Loss/Continue Condition**:
   - **Win**: If `correctness_score == 1.0` (or `> 0.95` depending on float strictness). The game ends (`done=True`), and the agent receives the full weighted `reward` (up to 1.0) and a congratulatory `message`.
   - **Loss**: If `attempts_remaining == 0` after a failed attempt. The game ends (`done=True`), and the agent receives a final `reward` based on their final attempt.
   - **Continue**: If `correctness_score < 1.0` and attempts remain. The agent loses an attempt (`attempts_remaining -= 1`). The game continues (`done=False`). The `reward` is `0.0` for the step, but the `message` and `execution_output` contain the tracebacks and hints needed to fix the code on the next turn.

## 4. Why This Design Matches `notebook.md`
- **Clean Contracts**: Exposes raw dataclasses, making it trivial for RL frameworks (like PPO, GRPO, or OpenEnv's base clients) to wrap the environment.
- **Iterative State**: The state genuinely evolves. In the previous `orchid_env` version, `step()` immediately advanced to an entirely new, unrelated task. Now, `step()` advances the *debugging state* of a single task, allowing RL agents to learn how to fix their own code based on environmental feedback.
- **Inference Compatibility**: The inference script will interact with a standard Gym-like interface (`obs = env.reset()`, `obs = env.step(action)`), allowing for easy integration with standard training loops.