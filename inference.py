"""
Inference Script: Data Forge Orchestrator Game
"""

import asyncio
import os
import json
import textwrap
import sys
import re
from typing import List, Optional

from openai import AsyncOpenAI

from client import OrchidEnv
from models import OrchestratorAction, SubAgentDeploy

# Check for debug flag
DEBUG_MODE = "--debug" in sys.argv

def log_debug(msg: str) -> None:
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}", flush=True)

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("ORCHID_ENV_TASK", "data_forge_orchestration")
BENCHMARK = os.getenv("ORCHID_ENV_BENCHMARK", "orchid_env")
MAX_STEPS = 5 # Match max_attempts in environment
TEMPERATURE = 0.2
MAX_TOKENS = 2048
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Big Data Orchestrator playing 'Data Forge'.
    Divide the extraction task into logical chunks, delegate them to sub-agents via python code, and provide a synthesis script.
    
    You must output a VALID JSON object matching this schema:
    {
      "chunking_strategy": "Explain your logic",
      "sub_agents": [
        {
          "role_prompt": "Instructions for this agent",
          "start_line": int,
          "end_line": int,
          "python_code": "print(extracted_data)" 
        }
      ],
      "synthesis_code": "print(final_result)" 
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    action = action.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs) -> str:
    prompt = f"Task: {obs.task_description}\nDataset: {obs.dataset_lines} lines\nSample:\n{obs.dataset_sample}\n"
    prompt += f"Attempts Remaining: {obs.attempts_remaining}\n"
    if obs.message or obs.execution_output:
        prompt += f"\nFeedback from last attempt: {obs.message}\n"
        if obs.execution_output:
            prompt += f"Execution Output:\n{obs.execution_output}\n"
        prompt += "Please fix the errors and try again."
    return prompt


async def get_model_message(client: AsyncOpenAI, obs, use_json_format: bool) -> OrchestratorAction:
    user_prompt = build_user_prompt(obs)
    try:
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
            
        completion = await client.chat.completions.create(**kwargs)
        raw_text = (completion.choices[0].message.content or "").strip()
        
        cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
        json_text = json_match.group(1) if json_match else cleaned_text
            
        data = json.loads(json_text)
        sub_agents = [SubAgentDeploy(**sa) for sa in data.get("sub_agents", [])]
        
        return OrchestratorAction(
            agent_id=MODEL_NAME,
            chunking_strategy=data.get("chunking_strategy", ""),
            sub_agents=sub_agents,
            synthesis_code=data.get("synthesis_code", "")
        )
    except Exception as exc:
        return OrchestratorAction(agent_id=MODEL_NAME, chunking_strategy=f"ERROR: {exc}")


async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Simple JSON mode probe
    use_json_format = False
    try:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Output JSON: {'r':4}"}],
            response_format={"type": "json_object"},
            max_tokens=10
        )
        use_json_format = True
    except: pass

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with OrchidEnv(base_url="http://localhost:7860", connect_timeout_s=600.0) as env:
            result = await env.reset()
            obs = result.observation
            
            for step in range(1, MAX_STEPS + 1):
                if result.done: break
                
                action = await get_model_message(client, obs, use_json_format)
                error = action.chunking_strategy if "ERROR" in action.chunking_strategy else None
                action_str = f"deploy(workers={len(action.sub_agents)})"

                result = await env.step(action)
                obs = result.observation

                reward = result.reward if result.reward is not None else 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=result.done, error=error)
                if result.done: break

            final_score = sum(rewards) / steps_taken if steps_taken > 0 else 0.0
            score = min(max(final_score, 0.0), 1.0)
            success = obs.correctness_score >= 1.0

    except Exception as e:
        log_debug(f"Error: {e}")
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
