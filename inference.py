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
import sys
from typing import List, Optional

from openai import AsyncOpenAI

from client import OrchidEnv
from models import OrchidAction, SubAgentConfig

# Check for debug flag
DEBUG_MODE = "--debug" in sys.argv

def log_debug(msg: str) -> None:
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}", flush=True)

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
    Your goal is to divide the problem into logical chunks, delegate them to sub-agents by providing them with specific python extraction code, and then provide a synthesis script to combine their outputs.
    
    You must output a VALID JSON object matching this schema:
    {
      "chunking_strategy": "Explain your logic",
      "sub_agents": [
        {
          "role_prompt": "Specific instructions for this agent (e.g., 'Extract JSON from root processes')",
          "start_line": int,
          "end_line": int,
          "python_code": "import re, json; ...; print(extracted_data)" 
        }
      ],
      "synthesis_code": "import json; ...; print(final_result)" 
    }
    
    Execution Context:
    - Map Phase: Each sub-agent's 'python_code' runs in an isolated MSB sandbox. 'chunk_data' (string) contains the line chunk.
    - Reduce Phase: 'synthesis_code' runs in a fresh sandbox. 'sub_outputs' (list of strings) contains the output from each sub-agent.
    - Efficiency: Aim for roughly 2000 lines per agent. Avoid overlapping chunks or missing lines.
    - Tips: Use 're' for regex, 'json' for parsing, and 'ast' if needed. Sub-agents should print a structured string (like JSON) so the synthesis code can easily parse it.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    # Ensure action has no newlines to keep single-line requirement
    action = action.replace("\n", " ")
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(f"[END]   success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(task_desc: str, dataset_lines: int, feedback: str) -> str:
    prompt = f"Task Description: {task_desc}\nDataset Size: {dataset_lines} lines\n"
    if feedback:
        prompt += f"\nPrevious Feedback:\n{feedback}\nPlease refine your orchestration strategy."
    return prompt


import re

async def check_json_support(client: AsyncOpenAI) -> bool:
    log_debug(f"Probing {MODEL_NAME} for <think> tags and JSON mode support...")
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2? Explain your reasoning, then output a JSON object with key 'result'."}],
            max_tokens=150,
            temperature=0.1
        )
        content = response.choices[0].message.content or ""
        if "<think>" in content:
            log_debug("🧠 Detected thinking model. Disabling strict JSON mode.")
            return False
            
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Output JSON with key 'result' and value 4."}],
            response_format={"type": "json_object"},
            max_tokens=50,
            temperature=0.1
        )
        log_debug("✅ Model supports strict JSON mode.")
        return True
    except Exception as e:
        log_debug(f"⚠️ Strict JSON mode not supported or probe failed ({e}). Falling back to robust parsing.")
        return False

async def get_model_message(client: AsyncOpenAI, task_desc: str, dataset_lines: int, feedback: str, use_json_format: bool) -> Optional[OrchidAction]:
    user_prompt = build_user_prompt(task_desc, dataset_lines, feedback)
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
        
        # 1. Strip Thinking Tags
        cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        
        # 2. Extract JSON payload
        json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_text = cleaned_text
            
        # 3. Robust JSON Fix: Sometimes LLMs output invalid escapes in regex/code
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            # Try to fix unescaped backslashes (common in regex)
            # This regex looks for backslashes that are NOT followed by valid JSON escape chars
            fixed_json = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_text)
            data = json.loads(fixed_json)
        
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
        log_debug(f"Model request or parsing failed: {exc}")
        return OrchidAction(
            agent_id=MODEL_NAME,
            chunking_strategy=f"FAILED_TO_PARSE_JSON: {exc}",
            sub_agents=[],
            synthesis_code="print('Error')"
        )


async def main() -> None:
    # Initialize OpenAI Client
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    use_json_format = await check_json_support(client)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    total_correctness = 0.0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with OrchidEnv(base_url="http://localhost:8000", connect_timeout_s=600.0, message_timeout_s=600.0) as env:
            result = await env.reset()
            obs = result.observation
            
            feedback = ""

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                    
                task_id = obs.task_id
                
                # Ask LLM for Orchestration Plan
                action = await get_model_message(client, obs.task_description, obs.dataset_lines, feedback, use_json_format)
                
                error = None
                if action.chunking_strategy.startswith("FAILED_TO_PARSE_JSON"):
                    error = action.chunking_strategy

                action_str = f"orchestrate(agents={len(action.sub_agents)})"

                # Execute action
                result = await env.step(action)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done
                feedback = obs.feedback
                
                total_correctness += obs.correctness_score if hasattr(obs, 'correctness_score') else obs.score

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
        log_debug(f"Environment execution error: {e}")
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
