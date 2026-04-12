# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Forge Orchestrator Game Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import OrchestratorAction, OrchestratorObservation, OrchestratorState
except ImportError:
    from models import OrchestratorAction, OrchestratorObservation, OrchestratorState


class OrchidEnv(
    EnvClient[OrchestratorAction, OrchestratorObservation, OrchestratorState]
):
    """
    Client for the Data Forge Orchestrator Game.
    """

    def _step_payload(self, action: OrchestratorAction) -> Dict:
        """
        Convert OrchestratorAction to JSON payload for step message.
        """
        return {
            "chunking_strategy": action.chunking_strategy,
            "sub_agents": [sa.model_dump() for sa in action.sub_agents],
            "synthesis_code": action.synthesis_code,
            "agent_id": action.agent_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OrchestratorObservation]:
        """
        Parse server response into StepResult[OrchestratorObservation].
        """
        obs_data = payload.get("observation", {})
        observation = OrchestratorObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            dataset_path=obs_data.get("dataset_path", ""),
            dataset_lines=obs_data.get("dataset_lines", 0),
            dataset_sample=obs_data.get("dataset_sample", ""),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            message=obs_data.get("message", ""),
            execution_output=obs_data.get("execution_output", ""),
            sub_agent_errors=obs_data.get("sub_agent_errors", 0),
            correctness_score=obs_data.get("correctness_score", 0.0),
            decomposition_score=obs_data.get("decomposition_score", 0.0),
            prompt_score=obs_data.get("prompt_score", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> OrchestratorState:
        """
        Parse server response into OrchestratorState object.
        """
        meta = payload.get("metadata", {})
        return OrchestratorState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=meta.get("task_id", ""),
            max_attempts=meta.get("max_attempts", 5),
            current_task_index=meta.get("current_task_index", 0),
        )
