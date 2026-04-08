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
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import OrchidEnv
from models import OrchidAction

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("ORCHID_ENV_TASK", "code_fix")
BENCHMARK = os.getenv("ORCHID_ENV_BENCHMARK", "orchid_env")
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5  # average score

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Python programmer.
    You will be provided with a task description and some broken code.
    Your goal is to provide the fixed code that satisfies the description and passes all tests.
    You must return ONLY the python code inside a ```python ``` block, nothing else.
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


def extract_code(text: str) -> str:
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_user_prompt(task_desc: str, broken_code: str, feedback: str) -> str:
    prompt = f"Task Description: {task_desc}\n\nBroken Code:\n```python\n{broken_code}\n```"
    if feedback:
        prompt += f"\n\nPrevious Feedback:\n{feedback}\nPlease try again and fix the errors."
    return prompt


def get_model_message(client: OpenAI, task_desc: str, broken_code: str, feedback: str) -> str:
    user_prompt = build_user_prompt(task_desc, broken_code, feedback)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return extract_code(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return broken_code


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
        # Connecting to the locally running instance since it is running on port 8000
        # Included extended timeouts for Daytona sandbox creation
        async with OrchidEnv(base_url="http://localhost:8000", connect_timeout_s=300.0, message_timeout_s=300.0) as env:
            result = await env.reset()
            obs = result.observation
            
            feedback = ""

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                    
                task_id = obs.task_id
                
                # Ask LLM for fixed code
                fixed_code = get_model_message(client, obs.task_description, obs.broken_code, feedback)
                
                # Define short action string for the mandatory [STEP] log
                action_str = f"submit_fix(task_id={task_id})"

                # Execute action
                result = await env.step(OrchidAction(task_id=task_id, code_submission=fixed_code, agent_id=MODEL_NAME))
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done
                error = None
                feedback = obs.feedback
                
                # We use the native environment "score" (correctness = tests_passed / tests_total)
                # to calculate the normalized [0, 1] end-of-episode score requested by HF inference rules
                total_correctness += obs.score

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

            # Calculate normalized score in [0, 1] range based on average task correctness
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