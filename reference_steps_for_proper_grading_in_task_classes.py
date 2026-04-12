
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
 
try:
    from ..models import OrchidAction, OrchidObservation
except ImportError:
    from models import OrchidAction, OrchidObservation
 
from .sandbox_controller import LocalController
import base64, os, ast, time
from concurrent.futures import ThreadPoolExecutor
 
 
# -----------------------------
# Grader functions (per-task)
# -----------------------------
 
def grade_list(truth, pred):
    truth_set = set(map(str, truth))
    pred_set = set(map(str, pred if isinstance(pred, list) else [pred]))
    if not truth_set and not pred_set:
        return 1.0
    return len(truth_set & pred_set) / len(truth_set | pred_set)
 
 
def grade_count(truth, pred):
    try:
        diff = abs(float(pred) - float(truth))
        return max(0.0, 1.0 - diff / max(1.0, float(truth)))
    except:
        return 0.0
 
 
def grade_dict(truth, pred):
    if not isinstance(pred, dict):
        return 0.0
    score = 0.0
    for k, v in truth.items():
        if k in pred:
            try:
                diff = abs(float(pred[k]) - float(v))
                score += max(0.0, 1.0 - diff / max(1.0, float(v)))
            except:
                pass
    return score / max(1, len(truth))
 
 
# -----------------------------
# Task definition
# -----------------------------
 
@dataclass
class BigDataTask:
    id: str
    description: str
    dataset_path: str
    dataset_lines: int
    ground_truth: str
    grader: Callable[[Any, Any], float]
 
 
TASK_BANK: List[BigDataTask] = [
    BigDataTask(
        id="extract_anomalies_easy",
        description="Extract all occurrences of EASTER_EGG_ERROR_CODE.",
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="['0x99A','0x99B','0x99A']",
        grader=grade_list
    ),
    BigDataTask(
        id="count_critical_medium",
        description="Count CRITICAL logs.",
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="3",
        grader=grade_count
    ),
    BigDataTask(
        id="count_by_module",
        description="Count logs per module.",
        dataset_path="server/mock_system.log",
        dataset_lines=10003,
        ground_truth="{'api_server':2031,'auth':1981,'kernel':1985,'nginx':2013,'postgres':1993}",
        grader=grade_dict
    ),
]
 
 
# -----------------------------
# Environment
# -----------------------------
 
class OrchidEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
 
    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_queue: List[BigDataTask] = []
        self._current_task: Optional[BigDataTask] = None
        self._sandbox_controller = LocalController()
        self._agent_scores: Dict[str, float] = {}
        self._episode_done: bool = False
 
    def reset(self, seed=None, episode_id=None, **kwargs) -> OrchidObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_queue = list(TASK_BANK)
        self._agent_scores = {}
        self._episode_done = False
        self._current_task = self._task_queue.pop(0) if self._task_queue else None
 
        if self._current_task:
            self._sandbox_controller.set_dataset(self._current_task.dataset_path)
 
        return self._next_task_observation()
 
    def step(self, action: OrchidAction) -> OrchidObservation:
        self._state.step_count += 1
 
        if self._episode_done or self._current_task is None:
            return OrchidObservation(task_id="", task_description="Done", done=True, reward=0.0)
 
        task = self._current_task
        agent_id = action.agent_id or "default"
 
        execution_output, correctness, sub_agent_errors, execution_time = self._evaluate_in_sandbox(
            action, task
        )
 
        reward = correctness - 0.05 * self._state.step_count
        if sub_agent_errors > 0:
            reward -= 0.1 * sub_agent_errors
        if "Error" in execution_output or "Traceback" in execution_output:
            reward -= 0.3
 
        reward = max(0.0, min(1.0, reward))
 
        self._agent_scores[agent_id] = self._agent_scores.get(agent_id, 0.0) + reward
 
        if self._task_queue:
            self._current_task = self._task_queue.pop(0)
            self._sandbox_controller.set_dataset(self._current_task.dataset_path)
            done = False
        else:
            self._current_task = None
            self._episode_done = True
            done = True
 
        return OrchidObservation(
            task_id=self._current_task.id if self._current_task else "",
            task_description=self._current_task.description if self._current_task else "",
            execution_output=execution_output,
            correctness_score=correctness,
            score=reward,
            done=done,
            reward=reward,
            metadata={"agent_id": agent_id, "step": self._state.step_count},
        )
 
    @property
    def state(self) -> State:
        return self._state
 
    def _evaluate_in_sandbox(self, action: OrchidAction, task: BigDataTask):
        sub_outputs = [None] * len(action.sub_agents)
        filename = os.path.basename(task.dataset_path)
 
        def run_sub(idx, sa):
            code = (
                f"with open('/data/{filename}','r') as f:\n"
                f" lines=f.readlines()[{sa.start_line}:{sa.end_line}]\n"
                "chunk_data=''.join(lines)\n"
                f"{sa.python_code}\n"
            )
            return idx, self._sandbox_controller.run_code(code)
 
        start = time.time()
 
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(run_sub, i, sa) for i, sa in enumerate(action.sub_agents)]
            for f in futures:
                try:
                    i, out = f.result()
                    sub_outputs[i] = out
                except:
                    pass
 
        errors = sum(1 for o in sub_outputs if isinstance(o, str) and ("Error" in o or "Traceback" in o))
 
        encoded = base64.b64encode(repr(sub_outputs).encode()).decode()
 
        synth = (
            "import base64,ast\n"
            f"sub_outputs=ast.literal_eval(base64.b64decode('{encoded}').decode())\n"
            f"{action.synthesis_code}\n"
        )
 
        final_output = self._sandbox_controller.run_code(synth)
        end = time.time()
 
        correctness = self._grade(task, final_output)
 
        return final_output, correctness, errors, (end - start)
 
    def _grade(self, task: BigDataTask, output: str) -> float:
        try:
            truth = ast.literal_eval(task.ground_truth)
        except:
            truth = task.ground_truth
 
        try:
            pred = ast.literal_eval(output)
        except:
            pred = output
 
        return task.grader(truth, pred)
 
    def _next_task_observation(self):
        if not self._current_task:
            return OrchidObservation(task_id="", task_description="No tasks", done=True, reward=0.0)
 
        return OrchidObservation(
            task_id=self._current_task.id,
            task_description=self._current_task.description,
            dataset_path=self._current_task.dataset_path,
            dataset_lines=self._current_task.dataset_lines,
            done=False,
            reward=0.0,
        )
 
