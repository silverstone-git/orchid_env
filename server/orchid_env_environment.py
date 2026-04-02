# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Orchid Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import os
from uuid import uuid4
from daytona import Daytona, DaytonaConfig
from dotenv import load_dotenv

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OrchidAction, OrchidObservation
except ImportError:
    from models import OrchidAction, OrchidObservation

# Load environment variables
load_dotenv()

class OrchidEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = OrchidEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Orchid Env environment ready!"
        >>>
        >>> obs = env.step(OrchidAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the orchid_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> OrchidObservation:
        """
        Reset the environment.

        Returns:
            OrchidObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Initialize sandbox_output in state
        self._state.sandbox_output = ""
        self._reset_count += 1

        return OrchidObservation(
            echoed_message="Orchid Env environment ready!",
            message_length=0,
            sandbox_output="",
            done=False,
            reward=0.0,
        )

    def step(self, action: OrchidAction) -> OrchidObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.
        Also runs a 'Hello World' on Daytona and captures the output.

        Args:
            action: OrchidAction containing the message to echo

        Returns:
            OrchidObservation with the echoed message, its length, and Daytona result
        """
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        # --- Daytona Integration ---
        sandbox_output = ""
        try:
            # Replicate test_daytona.py logic
            config = DaytonaConfig(
                api_key=os.getenv("DAYTONA_API_KEY", ""), 
            )
            daytona = Daytona(config)
            sandbox = daytona.create()
            response = sandbox.process.code_run('print("Hello World!")')
            sandbox_output = response.result
        except Exception as e:
            sandbox_output = f"Daytona Error: {str(e)}"
        
        # Update state with the Daytona result
        self._state.sandbox_output = sandbox_output

        return OrchidObservation(
            echoed_message=message,
            message_length=length,
            sandbox_output=sandbox_output,
            done=False,
            reward=reward,
            metadata={
                "original_message": message, 
                "step": self._state.step_count,
                "sandbox_output": sandbox_output
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
