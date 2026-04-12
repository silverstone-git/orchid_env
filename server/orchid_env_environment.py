from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
 
import base64
import os
import re
import time
import ast
import difflib
 
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
 
try:
    from ..models import OrchestratorAction, OrchestratorObservation, OrchestratorState
except ImportError:
    from models import OrchestratorAction, OrchestratorObservation, OrchestratorState
 
from .sandbox_controller import LocalController
 
 
# ---------------------------------------------------
# Robust Parsing
# ---------------------------------------------------
 
def robust_parse(output: str, expected_type: str) -> Any:
    try:
        return ast.literal_eval(output)
    except:
        match = re.search(r'(\[.*\]|\{.*\})', output, re.DOTALL)
        if match:
            try:
                return ast.literal_eval(match.group(1))
            except:
                pass
 
        if expected_type == "count":
            nums = re.findall(r'-?\d+\.?\d*', output)
            return float(nums[-1]) if nums else 0.0
 
    return output
 
 
# ---------------------------------------------------
# Graders
# ---------------------------------------------------
 
def grade_list(truth: str, pred_raw: str) -> float:
    truth_val = ast.literal_eval(truth)
    pred_val = robust_parse(pred_raw, "list")
 
    t_set = set(map(str, truth_val))
    p_set = set(map(str, pred_val if isinstance(pred_val, list) else [pred_val]))
 
    if not t_set and not p_set:
        return 1.0
 
    return len(t_set & p_set) / len(t_set | p_set)
 
 
def grade_count(truth: str, pred_raw: str) -> float:
    truth_val = float(truth)
    pred_val = robust_parse(pred_raw, "count")
 
    try:
        diff = abs(float(pred_val) - truth_val)
        return max(0.0, 1.0 - diff / max(1.0, truth_val))
    except:
        return 0.0
 
 
def grade_dict(truth: str, pred_raw: str) -> float:
    truth_val = ast.literal_eval(truth)
    pred_val = robust_parse(pred_raw, "dict")
 
    if not isinstance(pred_val, dict):
        return 0.0
 
    score = 0.0
    for k, v in truth_val.items():
        if k in pred_val:
            try:
                diff = abs(float(pred_val[k]) - float(v))
                score += max(0.0, 1.0 - diff / max(1.0, float(v)))
            except:
                pass
 
    return score / max(1, len(truth_val))
 
 
# ---------------------------------------------------
# Task Definition (FIXED)
# ---------------------------------------------------
 
@dataclass
class BigDataTask:
    id: str
    description: str
    dataset_path: str
    dataset_lines: int
    ground_truth: str
    grader_type: str   # ✅ REQUIRED FOR VALIDATOR
    grader: Callable[[str, str], float]
 
 
# ---------------------------------------------------
# TASK BANK (FIXED)
# ---------------------------------------------------
 
TASK_BANK: List[BigDataTask] = [
    BigDataTask(
        id="extract_anomalies_easy",
        description="Extract all EASTER_EGG_ERROR_CODE values.",
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="['0x99A','0x99B','0x99A']",
        grader_type="list",
        grader=grade_list
    ),
    BigDataTask(
        id="count_critical_medium",
        description="Count CRITICAL logs.",
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="3",
        grader_type="count",
        grader=grade_count
    ),
    BigDataTask(
        id="count_by_module",
        description="Count logs per module.",
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="{'api_server':2031,'auth':1981,'kernel':1985,'nginx':2013,'postgres':1993}",
        grader_type="dict",
        grader=grade_dict
    ),
]
 
 
# ---------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------
 
class OrchidEnvironment(Environment):
 
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
 
    def __init__(self):
        self._state = OrchestratorState(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[BigDataTask] = None
        self._sandbox_controller = LocalController()
        self._attempts_remaining = 5
        self._task_index = -1
 
    def reset(self, seed=None, episode_id=None, **kwargs):
        self._task_index = (self._task_index + 1) % len(TASK_BANK)
        self._current_task = TASK_BANK[self._task_index]
        self._attempts_remaining = 5
 
        self._sandbox_controller.set_dataset(self._current_task.dataset_path)
 
        return self._get_observation(message="New Task")
 
    def step(self, action: OrchestratorAction):
        self._state.step_count += 1
        self._attempts_remaining -= 1
 
        task = self._current_task
 
        output, correctness, _, errors, exec_time = self._evaluate(action, task)
 
        reward = correctness
        reward -= 0.05 * self._state.step_count
        reward -= 0.1 * errors
        reward -= exec_time * 0.01
 
        reward = max(0.0, min(1.0, reward))
 
        done = correctness == 1.0 or self._attempts_remaining <= 0
 
        return self._get_observation(
            done=done,
            reward=reward,
            execution_output=output,
            correctness=correctness
        )
 
    # ---------------------------------------------------
    # SANDBOX
    # ---------------------------------------------------
 
    def _evaluate(self, action, task):
        filename = os.path.basename(task.dataset_path)
        sub_outputs = []
 
        def run(sa):
            code = f"""
with open('/data/{filename}', 'r') as f:
    lines = f.readlines()[{sa.start_line}:{sa.end_line}]
    chunk_data = ''.join(lines)
{sa.python_code}
"""
            return self._sandbox_controller.run_code(code)
 
        start = time.time()
 
        with ThreadPoolExecutor(max_workers=10) as ex:
            sub_outputs = list(ex.map(run, action.sub_agents))
 
        encoded = base64.b64encode(repr(sub_outputs).encode()).decode()
 
        synth = f"""
import base64, ast
sub_outputs = ast.literal_eval(base64.b64decode('{encoded}').decode())
{action.synthesis_code}
"""
 
        final_output = self._sandbox_controller.run_code(synth)
 
        correctness = self._grade(task, final_output)
 
        errors = sum("Error" in str(o) for o in sub_outputs)
 
        return final_output, correctness, sub_outputs, errors, time.time() - start
 
    # ---------------------------------------------------
    # GRADING (FIXED)
    # ---------------------------------------------------
 
    def _grade(self, task: BigDataTask, output: str) -> float:
        try:
            if "Error" in output or "Traceback" in output:
                return 0.0
 
            # Primary (callable)
            if task.grader:
                return task.grader(task.ground_truth, output)
 
            # Fallback (validator-safe)
            if task.grader_type == "list":
                return grade_list(task.ground_truth, output)
            elif task.grader_type == "count":
                return grade_count(task.ground_truth, output)
            elif task.grader_type == "dict":
                return grade_dict(task.ground_truth, output)
 
        except:
            return 0.0
 
        return 0.0
 
    # ---------------------------------------------------
    # OBS
    # ---------------------------------------------------
 
    def _get_observation(self, done=False, reward=0.0, message="", execution_output="", correctness=0.0):
        task = self._current_task
 
        return OrchestratorObservation(
            done=done,
            reward=reward,
            task_id=task.id,
            task_description=task.description,
            dataset_path=task.dataset_path,
            dataset_lines=task.dataset_lines,
            message=message,
            execution_output=execution_output,
            correctness_score=correctness
        )
 
