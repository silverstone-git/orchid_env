import asyncio
import os
import json
import time
import textwrap
import re
from typing import Optional

from openai import AsyncOpenAI

from client import OrchidEnv
from models import OrchestratorAction, SubAgentDeploy

# Configuration mapping
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

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
    
    Execution Context:
    - Map Phase: Each sub-agent's 'python_code' runs in an isolated MSB sandbox. 'chunk_data' (string) contains the line chunk.
    - Reduce Phase: 'synthesis_code' runs in a fresh sandbox. 'sub_outputs' (list of strings) contains the output from each sub-agent.
    - Efficiency: Aim for roughly 2000 lines per agent. Avoid overlapping chunks or missing lines.
    - Tips: Use 're' for regex, 'json' for parsing, and 'ast' if needed. Sub-agents should print a structured string (like JSON) so the synthesis code can easily parse it.
    """
).strip()


async def check_json_support(client: AsyncOpenAI) -> bool:
    print(f"🔍 Probing {MODEL_NAME} for <think> tags and JSON mode support...")
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2? Explain your reasoning, then output a JSON object with key 'result'."}],
            max_tokens=150,
            temperature=0.1
        )
        content = response.choices[0].message.content or ""
        if "<think>" in content:
            print("🧠 Detected thinking model. Disabling strict JSON mode.")
            return False
            
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Output JSON with key 'result' and value 4."}],
            response_format={"type": "json_object"},
            max_tokens=50,
            temperature=0.1
        )
        print("✅ Model supports strict JSON mode.")
        return True
    except Exception as e:
        print(f"⚠️ Strict JSON mode not supported or probe failed ({e}). Falling back to robust parsing.")
        return False

async def generate_orchestration(client: AsyncOpenAI, obs, use_json_format: bool) -> OrchestratorAction:
    user_prompt = f"Task: {obs.task_description}\nDataset: {obs.dataset_lines} lines\nSample:\n{obs.dataset_sample}\n"
    user_prompt += f"Attempts Remaining: {obs.attempts_remaining}\n"
    
    if obs.message or obs.execution_output:
        user_prompt += f"\n--- FEEDBACK FROM PREVIOUS ATTEMPT ---\n"
        user_prompt += f"System Message: {obs.message}\n"
        if obs.execution_output:
            user_prompt += f"Execution Output:\n{obs.execution_output}\n"
        user_prompt += "Please fix the errors and try again."

    print(f"📡 Sending request to {MODEL_NAME}...")
    start_time = time.time()
    
    try:
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = await client.chat.completions.create(**kwargs)
        elapsed = time.time() - start_time
        print(f"⏱️ LLM responded in {elapsed:.2f}s")
        
        raw_text = (response.choices[0].message.content or "").strip()
        
        cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
        json_text = json_match.group(1) if json_match else cleaned_text

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            fixed_json = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_text)
            data = json.loads(fixed_json)

        print(f"📝 Strategy: {data.get('chunking_strategy')}")
        sub_agents_data = data.get("sub_agents", [])
        print(f"🤖 Spawning {len(sub_agents_data)} sub-agents")
        print("🧑‍💻 First agent code snippet:")
        if sub_agents_data:
            print(textwrap.indent(sub_agents_data[0].get('python_code', '')[:150] + '...', '    '))
        print("🧑‍💻 Synthesis code snippet:")
        print(textwrap.indent(data.get('synthesis_code', '')[:150] + '...', '    '))
        
        sub_agents = [SubAgentDeploy(**sa) for sa in sub_agents_data]
        
        return OrchestratorAction(
            agent_id=MODEL_NAME,
            chunking_strategy=data.get("chunking_strategy", ""),
            sub_agents=sub_agents,
            synthesis_code=data.get("synthesis_code", "")
        )
    except Exception as e:
        print(f"❌ Error during LLM API call/parsing: {e}")
        return OrchestratorAction(
            agent_id=MODEL_NAME,
            chunking_strategy=f"FAILED_VALIDATION_OR_API: {str(e)}",
            sub_agents=[],
            synthesis_code="print('Error')"
        )


async def main():
    print(f"🚀 Starting Verbose Data Forge Orchestrator Test...")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    
    if not API_KEY:
        print("❌ Error: HF_TOKEN or OPENAI_API_KEY environment variable is not set.")
        return

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    use_json_format = await check_json_support(client)

    async with OrchidEnv(base_url="http://localhost:7860", connect_timeout_s=600.0, message_timeout_s=600.0) as env:
        print("\n🔄 Initializing first task...")
        obs_result = await env.reset()
        obs = obs_result.observation
        done = obs_result.done
        
        step_num = 1
        
        while not done:
            print(f"\n{'#'*70}")
            print(f"### DEPLOYMENT ATTEMPT {step_num}")
            print(f"### Task: {obs.task_id}")
            print(f"### Attempts Remaining: {obs.attempts_remaining}")
            print(f"{'#'*70}")
            
            action = await generate_orchestration(client, obs, use_json_format)
            
            print(f"\n⚙️ Submitting orchestration to environment...")
            start_step = time.time()
            try:
                result = await env.step(action)
            except Exception as step_e:
                print(f"❌ Environment execution error during step(): {step_e}")
                break
                
            elapsed_step = time.time() - start_step
            print(f"⏱️ env.step() completed in {elapsed_step:.2f}s")
            
            obs = result.observation
            done = result.done
            
            print(f"\n📊 GRADER RESULTS:")
            print(f"  - Message:             {obs.message}")
            print(f"  - Correctness Score:   {obs.correctness_score:.2f}")
            print(f"  - Decomposition Score: {obs.decomposition_score:.2f}")
            print(f"  - Prompt Score:        {obs.prompt_score:.2f}")
            print(f"  - Sub-Agent Errors:    {obs.sub_agent_errors}")
            print(f"  - STEP REWARD:         {result.reward:.2f}")
            
            if obs.execution_output:
                print(f"\n📥 SYNTHESIZED OUTPUT:\n{obs.execution_output}")
            
            step_num += 1
            
            if done:
                print("\n🏁 Episode complete.")
                break

if __name__ == "__main__":
    asyncio.run(main())
