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

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OrchidAction(Action):
    """Code-fix submission from a child agent."""

    task_id: str = Field(default="", description="ID of the task being attempted")
    code_submission: str = Field(default="", description="The fixed code submitted by the agent")
    agent_id: str = Field(default="", description="Identifier for the submitting agent")


class OrchidObservation(Observation):
    """Result of evaluating an agent's code-fix submission and info for the next task."""

    task_id: str = Field(default="", description="ID of the next task to be attempted")
    task_description: str = Field(default="", description="Human-readable description of the next task")
    broken_code: str = Field(default="", description="Original broken code for the next task")
    execution_output: str = Field(default="", description="Full pytest output from the PREVIOUS task's sandbox run")
    tests_passed: int = Field(default=0, description="Number of tests that passed in the PREVIOUS task")
    tests_total: int = Field(default=0, description="Total number of tests run in the PREVIOUS task")
    score: float = Field(default=0.0, description="Normalized score of the PREVIOUS task (tests_passed / tests_total)")
    feedback: str = Field(default="", description="Human-readable feedback from the PREVIOUS task's evaluation")
