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
class CodingTask:
    id: str
    description: str
    broken_code: str
    test_code: str


TASK_BANK: List[CodingTask] = [
    CodingTask(
        id="fix_off_by_one",
        description=(
            "Fix the off-by-one error in `sum_list`. "
            "It should return the sum of ALL elements in the list."
        ),
        broken_code=(
            "def sum_list(nums):\n"
            "    total = 0\n"
            "    for i in range(len(nums) - 1):  # BUG: skips last element\n"
            "        total += nums[i]\n"
            "    return total"
        ),
        test_code=(
            "def test_sum_empty():\n"
            "    assert sum_list([]) == 0\n\n"
            "def test_sum_single():\n"
            "    assert sum_list([5]) == 5\n\n"
            "def test_sum_multiple():\n"
            "    assert sum_list([1, 2, 3, 4]) == 10\n\n"
            "def test_sum_negative():\n"
            "    assert sum_list([-1, -2, 3]) == 0\n"
        ),
    ),
    CodingTask(
        id="fix_type_error",
        description=(
            "Fix `divide` so it raises `ZeroDivisionError` instead of "
            "returning an error string when b == 0."
        ),
        broken_code=(
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        return 'Error: division by zero'  # BUG: should raise\n"
            "    return a / b"
        ),
        test_code=(
            "import pytest\n\n"
            "def test_normal():\n"
            "    assert divide(10, 2) == 5.0\n\n"
            "def test_zero_raises():\n"
            "    with pytest.raises(ZeroDivisionError):\n"
            "        divide(5, 0)\n\n"
            "def test_float_result():\n"
            "    assert divide(7, 2) == 3.5\n"
        ),
    ),
    CodingTask(
        id="fix_logic_bug",
        description=(
            "Fix `is_palindrome` — it incorrectly returns False for palindromes "
            "because a spurious character is appended before comparing."
        ),
        broken_code=(
            "def is_palindrome(s):\n"
            "    s = s.lower()\n"
            "    return s == s[::-1] + 'x'  # BUG: spurious 'x' appended\n"
        ),
        test_code=(
            "def test_palindrome():\n"
            "    assert is_palindrome('racecar') is True\n\n"
            "def test_non_palindrome():\n"
            "    assert is_palindrome('hello') is False\n\n"
            "def test_single_char():\n"
            "    assert is_palindrome('a') is True\n\n"
            "def test_case_insensitive():\n"
            "    assert is_palindrome('Madam') is True\n"
        ),
    ),
    CodingTask(
        id="fix_missing_return",
        description=(
            "Fix `factorial` — recursive calls are made but the return value "
            "is discarded, so the function always returns None for n > 1."
        ),
        broken_code=(
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    n * factorial(n - 1)  # BUG: missing `return`\n"
        ),
        test_code=(
            "def test_zero():\n"
            "    assert factorial(0) == 1\n\n"
            "def test_one():\n"
            "    assert factorial(1) == 1\n\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n\n"
            "def test_ten():\n"
            "    assert factorial(10) == 3628800\n"
        ),
    ),
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
        self._task_queue: List[CodingTask] = []
        self._current_task: Optional[CodingTask] = None
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
            self._sandbox = self._daytona.create()
            # Ensure pytest is installed in the sandbox's Python environment
            self._sandbox.process.code_run(
                'import subprocess, sys; '
                'subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], capture_output=True)'
            )
        except Exception as e:
            print(f"Failed to create sandbox or install pytest: {e}")
            self._sandbox = None

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

        Destroys any existing sandbox, resets state, loads the full task queue,
        creates a fresh Daytona sandbox, and returns the first task.

        Returns:
            OrchidObservation describing the first coding task.
        """
        self._destroy_sandbox()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_queue = list(TASK_BANK)
        self._agent_scores = {}
        self._episode_done = False
        self._create_sandbox()
        return self._next_task_observation()

    def step(self, action: OrchidAction) -> OrchidObservation:  # type: ignore[override]
        self._state.step_count += 1

        if self._episode_done or self._current_task is None:
            return OrchidObservation(
                task_id="",
                task_description="Episode complete. Call reset() to start a new episode.",
                broken_code="",
                execution_output="",
                tests_passed=0,
                tests_total=0,
                score=0.0,
                feedback="Episode done.",
                done=True,
                reward=0.0,
            )

        task = self._current_task
        agent_id = action.agent_id or "default"

        execution_output, tests_passed, tests_total = self._evaluate_in_sandbox(
            action.code_submission, task.test_code
        )

        correctness = tests_passed / tests_total if tests_total > 0 else 0.0

        reward = correctness

        if tests_passed == 0:
            reward -= 0.2

        if execution_output and "Traceback" in execution_output:
            reward -= 0.3

        reward -= 0.05 * self._state.step_count

        reward = max(-1.0, min(1.0, reward))

        self._agent_scores[agent_id] = self._agent_scores.get(agent_id, 0.0) + reward

        feedback = (
            f"[{agent_id}] Task '{task.id}': {tests_passed}/{tests_total} tests passed "
            f"(score={correctness:.2f}, reward={reward:.2f})\n"
            f"Output:\n{execution_output}"
        )

        done = self._advance_task()

        next_task_id = ""
        next_task_desc = ""
        next_broken_code = ""

        if not done and self._current_task:
            next_task_id = self._current_task.id
            next_task_desc = self._current_task.description
            next_broken_code = self._current_task.broken_code

        return OrchidObservation(
            task_id=next_task_id,
            task_description=next_task_desc,
            broken_code=next_broken_code,
            execution_output=execution_output,
            tests_passed=tests_passed,
            tests_total=tests_total,
            score=correctness,
            feedback=feedback,
            done=done,
            reward=reward,
            metadata={
                "agent_id": agent_id,
                "agent_scores": self._agent_scores,
                "step": self._state.step_count,
                "completed_task_id": task.id,
                "correctness": correctness,
            },
        )

    @property
    def state(self) -> State:
        """Current episode state (episode_id + step_count)."""
        return self._state

    # ------------------------------------------------------------------
    # Orchestrator helpers
    # ------------------------------------------------------------------

    def _next_task_observation(self) -> OrchidObservation:
        """Assign the next task from the queue and return the opening observation."""
        if not self._task_queue:
            self._episode_done = True
            self._current_task = None
            return OrchidObservation(
                task_id="",
                task_description="No tasks available.",
                broken_code="",
                execution_output="",
                tests_passed=0,
                tests_total=0,
                score=0.0,
                feedback="Task queue empty.",
                done=True,
                reward=0.0,
            )

        self._current_task = self._task_queue.pop(0)
        return OrchidObservation(
            task_id=self._current_task.id,
            task_description=self._current_task.description,
            broken_code=self._current_task.broken_code,
            execution_output="",
            tests_passed=0,
            tests_total=0,
            score=0.0,
            feedback="Task assigned. Submit your fix via step().",
            done=False,
            reward=0.0,
        )

    def _advance_task(self) -> bool:
        """Pop the next task. Returns True when the episode is finished."""
        if self._task_queue:
            self._current_task = self._task_queue.pop(0)
            return False
        self._current_task = None
        self._episode_done = True
        return True

    # ------------------------------------------------------------------
    # Sandbox evaluation
    # ------------------------------------------------------------------

    def _evaluate_in_sandbox(
        self, submission: str, test_code: str
    ) -> tuple[str, int, int]:
        """
        Combine submission + tests, run pytest inside Daytona, parse results.

        The combined source is base64-encoded before being sent to the sandbox
        to avoid shell-escaping issues with arbitrary code.

        Returns:
            (output, tests_passed, tests_total)
        """
        if self._sandbox is None:
            return "No sandbox available.", 0, 0

        combined = submission + "\n" + test_code
        encoded = base64.b64encode(combined.encode()).decode()

        runner = (
            "import base64, tempfile, subprocess\n"
            f"src = base64.b64decode('{encoded}').decode()\n"
            "with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:\n"
            "    f.write(src)\n"
            "    fname = f.name\n"
            "res = subprocess.run(\n"
            "    ['python', '-m', 'pytest', fname, '-v', '--tb=short'],\n"
            "    capture_output=True, text=True\n"
            ")\n"
            "print(res.stdout)\n"
            "print(res.stderr)\n"
        )

        try:
            response = self._sandbox.process.code_run(runner)
            output = response.result
        except Exception as e:
            return f"Sandbox error: {e}", 0, 0

        tests_passed, tests_total = self._parse_pytest(output)
        return output, tests_passed, tests_total

    @staticmethod
    def _parse_pytest(output: str) -> tuple[int, int]:
        """Extract (passed, total) counts from pytest -v output."""
        passed = len(re.findall(r" PASSED", output))
        failed = len(re.findall(r" FAILED", output))
        errors = len(re.findall(r" ERROR", output))
        return passed, passed + failed + errors

