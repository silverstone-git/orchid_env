"""
Inference Script: Data Forge Orchestrator Game (Multi-Task)
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

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("ORCHID_ENV_BENCHMARK", "orchid_env")
MAX_ATTEMPTS = 5 
NUM_TASKS_TO_RUN = 3
TEMPERATURE = 0.1
MAX_TOKENS = 2048

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
          "python_code": "import re, json; ...; print(extracted_data)" 
        }
      ],
      "synthesis_code": "import json; ...; print(final_result)" 
    }

    Execution Context:
    - Map Phase: Each sub-agent's 'python_code' runs in an isolated sandbox. 'chunk_data' (string) contains the line chunk.
    - Reduce Phase: 'synthesis_code' runs in a fresh sandbox. 'sub_outputs' (list of strings) contains the output from each sub-agent.
    - Sub-agents MUST print a JSON-formatted string so the synthesis code can parse it.
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
        prompt += f"\n--- FEEDBACK FROM PREVIOUS ATTEMPT ---\n"
        prompt += f"Status: {obs.message}\n"
        if obs.sub_agent_outputs:
            prompt += f"Worker Outputs (first few): {obs.sub_agent_outputs[:3]}\n"
        if obs.execution_output:
            prompt += f"Synthesis Output: {obs.execution_output}\n"
        prompt += "Please fix the logic and try again."
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
    
    use_json_format = False
    try:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Output JSON: {'r':4}"}],
            response_format={"type": "json_object"},
            max_tokens=10,
            timeout=10.0
        )
        use_json_format = True
    except: pass

    # Use local server or remote endpoint
    base_url = "http://localhost:7860"
    if "inference_remote.py" in sys.argv[0]:
        base_url = "https://eridians-orchid-env.hf.space"

    try:
        async with OrchidEnv(base_url=base_url, connect_timeout_s=600.0, message_timeout_s=600.0) as env:
            
            for task_num in range(1, NUM_TASKS_TO_RUN + 1):
                rewards: List[float] = []
                steps_taken = 0
                success = False

                result = await env.reset()
                obs = result.observation
                
                log_start(task=obs.task_id, env=BENCHMARK, model=MODEL_NAME)
                
                for step in range(1, MAX_ATTEMPTS + 1):
                    if result.done: break
                    
                    action = await get_model_message(client, obs, use_json_format)
                    llm_error = action.chunking_strategy if action.chunking_strategy.startswith("ERROR") else None
                    action_str = f"deploy(workers={len(action.sub_agents)})"

                    result = await env.step(action)
                    obs = result.observation

                    reward = result.reward if result.reward is not None else 0.0
                    rewards.append(reward)
                    steps_taken = step

                    env_error = None
                    if reward < 0.3 and "HINT" in obs.message:
                        env_error = obs.message.split("HINT:")[1].strip()
                    
                    error_msg = llm_error or env_error
                    log_step(step=step, action=action_str, reward=reward, done=result.done, error=error_msg)
                    
                    if result.done: break

                final_score = sum(rewards) / steps_taken if steps_taken > 0 else 0.0
                score = min(max(final_score, 0.0), 1.0)
                success = obs.correctness_score >= 1.0
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as e:
        log_debug(f"Benchmark run error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
