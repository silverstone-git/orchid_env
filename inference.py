"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import OrchidEnv
from models import OrchidAction, SubAgentConfig

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("ORCHID_ENV_TASK", "map_reduce_orchestration")
BENCHMARK = os.getenv("ORCHID_ENV_BENCHMARK", "orchid_env")
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 2048
SUCCESS_SCORE_THRESHOLD = 0.5  # average score

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Big Data Orchestrator.
    You will be given a massive task and the size of the dataset.
    Your goal is to divide the problem, delegate it to sub-agents by providing them with specific python extraction code, and then provide a synthesis script to combine their outputs.
    
    You must output a VALID JSON object matching this schema:
    {
      "chunking_strategy": "Explain your logic",
      "sub_agents": [
        {
          "role_prompt": "Specific instructions for this agent",
          "start_line": int,
          "end_line": int,
          "python_code": "print('extracted_data')" # The code the agent will run on `chunk_data` string
        }
      ],
      "synthesis_code": "print(sub_outputs)" # The code to combine the array of outputs
    }
    
    Note: Each sub-agent's python code will be executed in a sandbox where `chunk_data` is a string containing their assigned lines.
    The `synthesis_code` will be executed in a sandbox where `sub_outputs` is a list of strings returned by the sub-agents.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(task_desc: str, dataset_lines: int, feedback: str) -> str:
    prompt = f"Task Description: {task_desc}\nDataset Size: {dataset_lines} lines\n"
    if feedback:
        prompt += f"\nPrevious Feedback:\n{feedback}\nPlease refine your orchestration strategy."
    return prompt


def get_model_message(client: OpenAI, task_desc: str, dataset_lines: int, feedback: str) -> Optional[OrchidAction]:
    user_prompt = build_user_prompt(task_desc, dataset_lines, feedback)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        
        sub_agents = [
            SubAgentConfig(**sa) for sa in data.get("sub_agents", [])
        ]
        
        return OrchidAction(
            agent_id=MODEL_NAME,
            chunking_strategy=data.get("chunking_strategy", ""),
            sub_agents=sub_agents,
            synthesis_code=data.get("synthesis_code", "")
        )
    except Exception as exc:
        print(f"[DEBUG] Model request or parsing failed: {exc}", flush=True)
        return None


async def main() -> None:
    # Initialize OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    total_correctness = 0.0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with OrchidEnv(base_url="http://localhost:8000", connect_timeout_s=300.0, message_timeout_s=300.0) as env:
            result = await env.reset()
            obs = result.observation
            
            feedback = ""

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                    
                task_id = obs.task_id
                
                # Ask LLM for Orchestration Plan
                action = get_model_message(client, obs.task_description, obs.dataset_lines, feedback)
                
                if not action:
                    print("[DEBUG] Failed to generate valid action. Skipping step.")
                    break

                action_str = f"orchestrate(agents={len(action.sub_agents)})"

                # Execute action
                result = await env.step(action)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done
                error = None
                feedback = obs.feedback
                
                total_correctness += obs.score

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

            # Calculate normalized score in [0, 1] range based on average task score
            score = total_correctness / steps_taken if steps_taken > 0 else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Environment execution error: {e}", flush=True)
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())