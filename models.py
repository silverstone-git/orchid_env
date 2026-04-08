# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Orchid Env RL Evaluation Environment.

OrchidAction  — code-fix submission from a child agent.
OrchidObservation — result of evaluating that submission.
"""

from typing import List, Union
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator
import json

class SubAgentConfig(BaseModel):
    """Configuration for a single spawned sub-agent sandbox."""
    role_prompt: str = Field(..., description="The persona or instructions defining this agent's purpose.")
    start_line: int = Field(..., description="The starting line of the dataset this agent will process.")
    end_line: int = Field(..., description="The ending line of the dataset this agent will process.")
    python_code: str = Field(..., description="The Python script this agent will execute inside its sandbox to extract data.")

class OrchidAction(Action):
    """Orchestrator submission for task breakdown and mapping."""
    agent_id: str = Field(default="", description="Identifier for the orchestrator.")
    chunking_strategy: str = Field(default="Single chunk", description="Explanation of why the data was chunked this way.")
    sub_agents: List[SubAgentConfig] = Field(default_factory=list, description="The list of sub-agents to spawn.")
    synthesis_code: str = Field(default="print(sub_outputs)", description="The Python script to run on the synthesized JSON outputs of the sub-agents.")

    @model_validator(mode='before')
    @classmethod
    def parse_sub_agents_string(cls, values):
        # Gradio sometimes passes lists as JSON strings
        if isinstance(values, dict) and 'sub_agents' in values:
            if isinstance(values['sub_agents'], str):
                try:
                    values['sub_agents'] = json.loads(values['sub_agents'])
                except json.JSONDecodeError:
                    pass  # Let Pydantic fail normally if it's invalid JSON
        return values

class OrchidObservation(Observation):
    """Result of the multi-agent orchestration execution."""
    task_id: str = Field(default="", description="ID of the next task to be attempted")
    task_description: str = Field(default="", description="Human-readable description of the next task")
    dataset_path: str = Field(default="", description="Path to the large context file for the next task")
    dataset_lines: int = Field(default=0, description="Total number of lines in the dataset")
    
    execution_output: str = Field(default="", description="Synthesized output from the PREVIOUS task's map-reduce execution")
    correctness_score: float = Field(default=0.0, description="Score based on the accuracy of the final synthesis (0.0 to 1.0)")
    decomposition_score: float = Field(default=0.0, description="Score based on efficiency of chunking (punishes overlap, huge chunks, or too many agents)")
    prompt_score: float = Field(default=0.0, description="Score evaluating the quality of the sub-agent role prompts")
    score: float = Field(default=0.0, description="Overall weighted score of the orchestration")
    reward: float = Field(default=0.0, description="The RL reward signal")
    feedback: str = Field(default="", description="Detailed feedback on the orchestration strategy")
    done: bool = Field(default=False, description="Whether the episode is complete")
