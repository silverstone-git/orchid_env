# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data contracts for the Data Forge Orchestrator Game.
"""

from typing import List, Dict, Optional, Any
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

class SubAgentDeploy(BaseModel):
    """A worker agent deployed to process a specific chunk of data."""
    start_line: int = Field(..., description="The starting line of the dataset")
    end_line: int = Field(..., description="The ending line of the dataset")
    role_prompt: str = Field(..., description="Instructions for this agent")
    python_code: str = Field(..., description="Python script to execute")

class OrchestratorAction(Action):
    """The player submits a full Map-Reduce architecture."""
    chunking_strategy: str = Field(default="Single chunk", description="Logic explanation")
    sub_agents: List[SubAgentDeploy] = Field(default_factory=list, description="Worker pool")
    synthesis_code: str = Field(default="print(sub_outputs)", description="Final reduce script")
    agent_id: str = Field(default="default", description="ID of the model acting")

class OrchestratorObservation(Observation):
    """What the player sees after deploying their architecture."""
    # Note: 'done' and 'reward' are inherited from Observation
    
    # Task Context (remains static during the episode)
    task_id: str = Field(default="", description="ID of the current task")
    task_description: str = Field(default="", description="Task explanation")
    dataset_path: str = Field(default="", description="Path to the file")
    dataset_lines: int = Field(default=0, description="Total lines in file")
    dataset_sample: str = Field(default="", description="First 5 lines of the file")
    
    # Game State (evolves during the episode)
    attempts_remaining: int = Field(default=5, description="Number of deploys left")
    
    # Feedback from the last deployment
    message: str = Field(default="", description="High-level feedback")
    execution_output: str = Field(default="", description="Output from synthesis_code")
    sub_agent_errors: int = Field(default=0, description="Number of crashed workers")
    sub_agent_outputs: List[str] = Field(default_factory=list, description="Outputs from individual workers")
    
    # Partial Credit Breakdown (from the Grader)
    correctness_score: float = Field(default=0.0, description="Accuracy of final answer")
    decomposition_score: float = Field(default=0.0, description="Chunk overlap/efficiency score")
    prompt_score: float = Field(default=0.0, description="Quality of role_prompts")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Misc metadata")

class OrchestratorState(State):
    """Internal episode metadata."""
    # Note: 'episode_id' and 'step_count' are inherited from State
    task_id: str = Field(default="")
    max_attempts: int = Field(default=5)
    current_task_index: int = Field(default=0)
