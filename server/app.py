# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Data Forge Orchestrator Game.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import OrchestratorAction, OrchestratorObservation
    from .orchid_env_environment import OrchidEnvironment
except ImportError:
    from models import OrchestratorAction, OrchestratorObservation
    from server.orchid_env_environment import OrchidEnvironment


# Create the app with web interface and README integration
app = create_app(
    OrchidEnvironment,
    OrchestratorAction,
    OrchestratorObservation,
    env_name="orchid_env",
    max_concurrent_envs=5,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    import sys
    
    if len(sys.argv) > 1 and "--port" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=port)
        args, _ = parser.parse_known_args()
        port = args.port

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
