# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Orchid Env RL Evaluation Environment.

An RL environment where child agents are evaluated on real coding-fix tasks.
An orchestrator manages the task queue; each step() evaluates a submitted
code fix inside an isolated Daytona sandbox and returns a reward signal.

Flow::

    obs = env.reset()               # assigned first task (description + broken code)
    while not obs.done:
        fix = agent.solve(obs)      # child agent produces a code fix
        obs = env.step(fix)         # run fix in sandbox, score it, advance task

Reward = tests_passed / tests_total  (0.0 – 1.0 per task)
"""

import base64
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from daytona import Daytona, DaytonaConfig
from dotenv import load_dotenv

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OrchidAction, OrchidObservation
except ImportError:
    from models import OrchidAction, OrchidObservation

load_dotenv()


# ---------------------------------------------------------------------------
# Task bank
# ---------------------------------------------------------------------------

@dataclass
class BigDataTask:
    id: str
    description: str
    dataset_path: str
    dataset_lines: int
    ground_truth: str
    task_type: str # 'list' or 'count'

TASK_BANK: List[BigDataTask] = [
    BigDataTask(
        id="extract_anomalies_easy",
        description=(
            "EASY: You are given a massive system log file. "
            "Extract all occurrences of 'EASTER_EGG_ERROR_CODE' values. "
            "Output must be a python list of the string codes found, e.g., ['0x99A', '0x99B']."
        ),
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="['0x99A', '0x99B', '0x99A']",
        task_type="list"
    ),
    BigDataTask(
        id="count_critical_medium",
        description=(
            "MEDIUM: You are given a massive system log file. "
            "Count the exact total number of 'CRITICAL' level logs in the entire file. "
            "Output must be just the integer count."
        ),
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="3",
        task_type="count"
    ),
    BigDataTask(
        id="extract_timestamps_hard",
        description=(
            "HARD: You are given a massive system log file. "
            "Extract the exact timestamps (e.g., 'Feb 01 12:00:00') of every 'CRITICAL' log. "
            "Output must be a python list of strings in chronological order."
        ),
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="['Feb 01 12:00:00', 'Feb 05 14:30:00', 'Feb 10 09:15:00']",
        task_type="list"
    )
]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OrchidEnvironment(Environment):
    """
    RL environment for evaluating code-fixing agents.

    The orchestrator (this class) manages a FIFO queue of CodingTasks.
    On each step() a child agent submits a code fix which is executed inside
    an isolated Daytona sandbox together with the task's pytest suite.
    The reward equals the fraction of tests that pass (0.0 – 1.0).

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: allows multiple independent WebSocket
            sessions, each with their own environment instance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_queue: List[BigDataTask] = []
        self._current_task: Optional[BigDataTask] = None
        self._sandbox = None
        self._daytona: Daytona
        self._agent_scores: Dict[str, float] = {}
        self._episode_done: bool = False
        self._init_daytona()

    # ------------------------------------------------------------------
    # Daytona helpers
    # ------------------------------------------------------------------

    def _init_daytona(self) -> None:
        config = DaytonaConfig(api_key=os.getenv("DAYTONA_API_KEY", ""))
        self._daytona = Daytona(config)  # always set; never None after __init__

    def _create_sandbox(self) -> None:
        try:
            params = CreateSandboxFromSnapshotParams(
                auto_stop_interval=1,
                auto_delete_interval=1
            )
            self._sandbox = self._daytona.create(params)
            self._provision_logs()
        except Exception as e:
            print(f"FAILED TO CREATE SANDBOX: {e}")
            self._sandbox = None

    def _provision_logs(self) -> None:
        # Push the dataset to the sandbox
        if self._sandbox and self._current_task and os.path.exists(self._current_task.dataset_path):
            with open(self._current_task.dataset_path, "r") as f:
                log_content = f.read()

            # Write to the sandbox
            encoded_logs = base64.b64encode(log_content.encode()).decode()
            try:
                self._sandbox.process.code_run(
                    f"import base64; "
                    f"with open('dataset.log', 'w') as f: "
                    f"f.write(base64.b64decode('{encoded_logs}').decode())"
                )
            except Exception as e:
                print(f"FAILED TO PROVISION LOGS: {e}")


    def _destroy_sandbox(self) -> None:
        if self._sandbox is not None and self._daytona is not None:
            try:
                self._daytona.delete(self._sandbox)
            except Exception:
                pass
            self._sandbox = None

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> OrchidObservation:  # type: ignore[override]
        """
        Start a new episode.
        """
        self._destroy_sandbox()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_queue = list(TASK_BANK)
        self._agent_scores = {}
        self._episode_done = False
        self._current_task = self._task_queue.pop(0) if self._task_queue else None
        
        # Provision the sandbox once per episode
        self._create_sandbox()
        return self._next_task_observation()

    def step(self, action: OrchidAction) -> OrchidObservation:  # type: ignore[override]
        self._state.step_count += 1

        if self._episode_done or self._current_task is None:
            return OrchidObservation(
                task_id="",
                task_description="Episode complete.",
                done=True,
                reward=0.0
            )

        # If sandbox failed during reset, try one last time to create it
        if self._sandbox is None:
            self._create_sandbox()

        task = self._current_task
        agent_id = action.agent_id or "default"

        # Evaluate the execution graph inside the sandbox
        execution_output, correctness = self._evaluate_in_sandbox(action, task.ground_truth, task.task_type)

        # ... (reward calculation logic remains same) ...
        
        # Calculate Decomposition Efficiency
        total_lines_processed = 0
        overlap_penalty = 0
        covered_lines = set()
        
        for sa in action.sub_agents:
            chunk_size = sa.end_line - sa.start_line
            total_lines_processed += chunk_size
            for i in range(sa.start_line, sa.end_line):
                if i in covered_lines:
                    overlap_penalty += 1
                covered_lines.add(i)
                
        total_agents = len(action.sub_agents)
        dataset_lines = task.dataset_lines
        
        # Ideal agents logic: Roughly 1 agent per 2000 lines
        ideal_agents = max(1, dataset_lines // 2000)
        agent_count_penalty = abs(total_agents - ideal_agents) * 0.1
        
        # Overlap penalty (0.0 to 1.0 scale)
        overlap_ratio = overlap_penalty / max(1, dataset_lines)
        
        # Missing lines penalty
        missing_lines = dataset_lines - len(covered_lines)
        missing_ratio = missing_lines / max(1, dataset_lines)
        
        decomposition_score = 1.0 - agent_count_penalty - overlap_ratio - missing_ratio
        decomposition_score = max(0.0, min(1.0, decomposition_score))
        
        # Mock Prompt Score (heuristic for now, checks for relevant keywords)
        prompt_score = 0.0
        if action.sub_agents:
            total_prompt_score = 0
            for sa in action.sub_agents:
                p_lower = sa.role_prompt.lower()
                if "extract" in p_lower or "find" in p_lower: total_prompt_score += 0.5
                if "easter_egg" in p_lower or "error" in p_lower: total_prompt_score += 0.5
            prompt_score = total_prompt_score / total_agents

        # Calculate Final Reward
        reward = (0.5 * correctness) + (0.3 * decomposition_score) + (0.2 * prompt_score)
        
        # Step decay to discourage endless looping
        reward -= 0.05 * self._state.step_count

        self._agent_scores[agent_id] = self._agent_scores.get(agent_id, 0.0) + reward

        feedback = (
            f"[{agent_id}] Map-Reduce Execution:\n"
            f"  - Sub-agents spawned: {total_agents} (Ideal: {ideal_agents})\n"
            f"  - Decomposition Score: {decomposition_score:.2f} (Overlap: {overlap_penalty}, Missing: {missing_lines})\n"
            f"  - Prompt Quality: {prompt_score:.2f}\n"
            f"  - Correctness: {correctness:.2f}\n"
            f"Total Step Reward: {reward:.2f}\n"
            f"Output:\n{execution_output}"
        )

        # Advance Task
        done = False
        completed_task_id = self._current_task.id
        if self._task_queue:
            self._current_task = self._task_queue.pop(0)
            # Re-provision logs for the new task inside the SAME sandbox
            self._provision_logs()
        else:
            self._current_task = None
            self._episode_done = True
            done = True

        return OrchidObservation(
            task_id=self._current_task.id if self._current_task else "",
            task_description=self._current_task.description if self._current_task else "",
            dataset_path=self._current_task.dataset_path if self._current_task else "",
            dataset_lines=self._current_task.dataset_lines if self._current_task else 0,
            execution_output=execution_output,
            correctness_score=correctness,
            decomposition_score=decomposition_score,
            prompt_score=prompt_score,
            score=reward,
            feedback=feedback,
            done=done,
            reward=reward,
            metadata={
                "agent_id": agent_id,
                "agent_scores": self._agent_scores,
                "step": self._state.step_count,
                "completed_task_id": completed_task_id,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Orchestrator helpers
    # ------------------------------------------------------------------

    def _next_task_observation(self) -> OrchidObservation:
        """Assign the next task from the queue and return the opening observation."""
        if not self._current_task:
            self._episode_done = True
            return OrchidObservation(
                task_id="",
                task_description="No tasks available.",
                done=True,
                reward=0.0,
            )

        return OrchidObservation(
            task_id=self._current_task.id,
            task_description=self._current_task.description,
            dataset_path=self._current_task.dataset_path,
            dataset_lines=self._current_task.dataset_lines,
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    def _evaluate_in_sandbox(self, action: OrchidAction, ground_truth: str, task_type: str) -> tuple[str, float]:
        if self._sandbox is None:
            return "No sandbox available.", 0.0

        sub_outputs = []
        
        # 1. Map Phase: Execute all sub-agents sequentially (or virtually concurrently)
        for idx, sub_agent in enumerate(action.sub_agents):
            agent_runner = (
                "import sys, json\n"
                "with open('dataset.log', 'r') as f:\n"
                f"    lines = f.readlines()[{sub_agent.start_line}:{sub_agent.end_line}]\n"
                "chunk_data = ''.join(lines)\n\n"
                f"{sub_agent.python_code}\n"
            )
            try:
                response = self._sandbox.process.code_run(agent_runner)
                sub_outputs.append(response.result.strip())
            except Exception as e:
                sub_outputs.append(f"Error in SubAgent {idx}: {str(e)}")

        # 2. Reduce Phase: Run the synthesis code
        escaped_outputs = base64.b64encode(repr(sub_outputs).encode()).decode()
        
        synth_runner = (
            "import base64, ast, sys\n"
            f"sub_outputs = ast.literal_eval(base64.b64decode('{escaped_outputs}').decode())\n\n"
            f"{action.synthesis_code}\n"
        )
        
        try:
            response = self._sandbox.process.code_run(synth_runner)
            final_output = response.result.strip()
        except Exception as e:
            final_output = f"Error in Synthesis: {str(e)}"
            
        # 3. Evaluate Correctness (Partial Progress Grader)
        correctness = 0.0
        try:
            if "Error" in final_output or "Traceback" in final_output:
                correctness = -0.5
            else:
                import ast
                truth_val = ast.literal_eval(ground_truth)
                try:
                    pred_val = ast.literal_eval(final_output)
                except:
                    pred_val = final_output # fallback to string
                
                if task_type == "list" and isinstance(truth_val, list) and isinstance(pred_val, list):
                    # Jaccard-like overlap for partial credit
                    truth_set = set(truth_val)
                    pred_set = set(pred_val)
                    if not truth_set and not pred_set:
                        correctness = 1.0
                    else:
                        intersection = truth_set.intersection(pred_set)
                        union = truth_set.union(pred_set)
                        correctness = len(intersection) / len(union)
                elif task_type == "count":
                    # Distance-based partial credit for counts
                    try:
                        pred_count = int(pred_val)
                        truth_count = int(truth_val)
                        diff = abs(pred_count - truth_count)
                        correctness = max(0.0, 1.0 - (diff / max(1, truth_count)))
                    except:
                        correctness = 0.0
                else:
                    # Fallback strict match
                    if str(truth_val).strip() == str(pred_val).strip():
                        correctness = 1.0
        except Exception as e:
            print(f"Grader exception: {e}")
            correctness = 0.0
            
        return final_output, correctness
