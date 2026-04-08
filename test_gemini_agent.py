import asyncio
import httpx
import re
import os
import json
import time
from typing import Optional, List
from pydantic import ValidationError

from client import OrchidEnv
from models import OrchidAction, SubAgentConfig, OrchidObservation

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
# Using the OpenAI-compatible endpoint for convenience with response_format
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

SYSTEM_PROMPT = """You are an expert Big Data Orchestrator.
You will be given a task and a dataset size (number of lines).
Your goal is to divide the problem, delegate it to sub-agents by providing them with specific python extraction code, and then provide a synthesis script to combine their outputs.

You must output a VALID JSON object matching this schema:
{
  "chunking_strategy": "Explain your logic",
  "sub_agents": [
    {
      "role_prompt": "Specific instructions for this agent",
      "start_line": int,
      "end_line": int,
      "python_code": "print('extracted_data')" 
    }
  ],
  "synthesis_code": "print(sub_outputs)" 
}

Note: 
- Each sub-agent's python code will be executed in a sandbox where 'chunk_data' (string) is available.
- The 'synthesis_code' will be executed in a sandbox where 'sub_outputs' (list of strings) is available.
- Aim for roughly 2000 lines per agent for efficiency.
"""

async def generate_orchestration(task_desc: str, dataset_lines: int, feedback: str) -> Optional[OrchidAction]:
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        return None

    prompt = f"Task: {task_desc}\nDataset Size: {dataset_lines} lines"
    if feedback:
        prompt += f"\n\nPrevious Feedback:\n{feedback}"

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use OpenAI API Key format for Gemini's OpenAI endpoint
    # Gemini requires the key in the Authorization header as a Bearer token
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    start_time = time.time()
    print(f"📡 Sending request to Gemini ({GEMINI_MODEL})...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GEMINI_URL, headers=headers, json=payload, timeout=90.0)
            elapsed = time.time() - start_time
            print(f"⏱️ Gemini responded in {elapsed:.2f}s")
            
            response.raise_for_status()
            result = response.json()
            
            text = result['choices'][0]['message']['content']
            data = json.loads(text)
            
            print(f"📝 Strategy: {data.get('chunking_strategy')}")
            print(f"🤖 Spawning {len(data.get('sub_agents', []))} sub-agents")
            
            sub_agents = [SubAgentConfig(**sa) for sa in data.get("sub_agents", [])]
            return OrchidAction(
                agent_id="gemini-orchestrator",
                chunking_strategy=data.get("chunking_strategy", ""),
                sub_agents=sub_agents,
                synthesis_code=data.get("synthesis_code", "")
            )
        except Exception as e:
            print(f"❌ Error during Gemini API call/parsing: {e}")
            if 'response' in locals():
                print(f"Response content: {response.text}")
            return None

async def main():
    print(f"🚀 Starting Verbose Gemini Orchestrator Test...")
    print(f"Model: {GEMINI_MODEL}")
    
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY is not set.")
        return

    async with OrchidEnv(base_url="http://localhost:8000", connect_timeout_s=300.0, message_timeout_s=300.0) as env:
        print("🔄 Resetting environment (Provisioning Daytona & Logs)...")
        start_reset = time.time()
        obs_result = await env.reset()
        print(f"⏱️ Reset took {time.time() - start_reset:.2f}s")
        
        obs = obs_result.observation
        done = obs_result.done
        
        step_num = 1
        total_score = 0.0
        feedback = ""
        
        while not done:
            print(f"\n{'#'*70}")
            print(f"### STEP {step_num}")
            print(f"### Task: {obs.task_id}")
            print(f"### Description: {obs.task_description}")
            print(f"### Dataset: {obs.dataset_path} ({obs.dataset_lines} lines)")
            print(f"{'#'*70}")
            
            # 1. Ask Gemini for the plan
            action = await generate_orchestration(obs.task_description, obs.dataset_lines, feedback)
            
            if not action:
                print("⚠️ Failed to get action from Gemini. Retrying or exiting...")
                break
                
            # 2. Submit to Env
            print(f"\n⚙️ Submitting orchestration to environment...")
            start_step = time.time()
            result = await env.step(action)
            elapsed_step = time.time() - start_step
            print(f"⏱️ env.step() completed in {elapsed_step:.2f}s")
            
            obs = result.observation
            done = result.done
            
            print(f"\n📊 RESULTS FOR TASK: {result.observation.metadata.get('completed_task_id')}")
            print(f"  - Correctness Score:  {obs.correctness_score:.2f}")
            print(f"  - Decomposition Score: {obs.decomposition_score:.2f}")
            print(f"  - Prompt Score:        {obs.prompt_score:.2f}")
            print(f"  - TOTAL STEP REWARD:   {result.reward:.2f}")
            
            print(f"\n💬 FEEDBACK FROM ENV:\n{obs.feedback}")
            
            if obs.execution_output:
                print(f"\n📥 FINAL SYNTHESIZED OUTPUT:\n{obs.execution_output}")
            
            feedback = obs.feedback
            total_score += obs.score
            step_num += 1
            
            if done:
                print("\n🏁 All tasks in bank completed.")
                break
        
        print(f"\n{'='*70}")
        print(f"🏆 EPISODE COMPLETE")
        print(f"Steps taken: {step_num - 1}")
        print(f"Final Aggregated Score: {total_score:.2f}")
        print(f"{'='*70}")

if __name__ == "__main__":
    asyncio.run(main())
