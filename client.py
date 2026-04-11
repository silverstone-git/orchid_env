# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Orchid Env RL Evaluation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import OrchidAction, OrchidObservation
except ImportError:
    from models import OrchidAction, OrchidObservation


class OrchidEnv(
    EnvClient[OrchidAction, OrchidObservation, State]
):
    """
    Client for the Orchid Env RL Evaluation Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance gets its own isolated environment session (and Daytona sandbox).

    Example::

        with OrchidEnv(base_url="http://localhost:7860") as env:
            obs = env.reset()
            print(obs.observation.task_description)
            print(obs.observation.broken_code)

            result = env.step(OrchidAction(
                task_id=obs.observation.task_id,
                code_submission="def sum_list(nums): return sum(nums)",
                agent_id="agent-1",
            ))
            print(result.observation.feedback)
            print(result.reward)
    """

    def _step_payload(self, action: OrchidAction) -> Dict:
        """
        Convert OrchidAction to JSON payload for step message.
        """
        return {
            "chunking_strategy": action.chunking_strategy,
            "sub_agents": [sa.model_dump() for sa in action.sub_agents],
            "synthesis_code": action.synthesis_code,
            "agent_id": action.agent_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OrchidObservation]:
        """
        Parse server response into StepResult[OrchidObservation].
        """
        obs_data = payload.get("observation", {})
        observation = OrchidObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            dataset_path=obs_data.get("dataset_path", ""),
            dataset_lines=obs_data.get("dataset_lines", 0),
            execution_output=obs_data.get("execution_output", ""),
            correctness_score=obs_data.get("correctness_score", 0.0),
            decomposition_score=obs_data.get("decomposition_score", 0.0),
            prompt_score=obs_data.get("prompt_score", 0.0),
            score=obs_data.get("score", 0.0),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
