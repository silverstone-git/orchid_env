import asyncio
import os
import json
import time
import textwrap
import re
from typing import Optional

from openai import AsyncOpenAI
from pydantic import ValidationError

from client import OrchidEnv
from models import OrchidAction, SubAgentConfig

# Configuration mapping (uses HF_TOKEN, fallbacks to OPENAI_API_KEY)
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

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


async def check_json_support(client: AsyncOpenAI) -> bool:
    print(f"🔍 Probing {MODEL_NAME} for <think> tags and JSON mode support...")
    try:
        # 1. Check for <think> tags natively
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
            
        # 2. Check if strict JSON mode is supported by the endpoint
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

async def generate_orchestration(client: AsyncOpenAI, task_desc: str, dataset_lines: int, feedback: str, use_json_format: bool) -> Optional[OrchidAction]:
    user_prompt = f"Task Description: {task_desc}\nDataset Size: {dataset_lines} lines\n"
    if feedback:
        user_prompt += f"\nPrevious Feedback:\n{feedback}\nPlease refine your orchestration strategy."

    print(f"📡 Sending request to {MODEL_NAME} via OpenAI API...")
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
        
        # 1. Handle Thinking Tags: Strip anything between <think> and </think>
        cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        
        # 2. Extract JSON: Look for the first { and last } to avoid conversational chatter
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
            fixed_json = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_text)
            data = json.loads(fixed_json)

        print(f"🔍 Cleaned Output Snippet: {json_text[:200]}...")
        
        print(f"📝 Strategy: {data.get('chunking_strategy')}")
        sub_agents_data = data.get("sub_agents", [])
        print(f"🤖 Spawning {len(sub_agents_data)} sub-agents")
        
        sub_agents = [SubAgentConfig(**sa) for sa in sub_agents_data]
        
        return OrchidAction(
            agent_id=MODEL_NAME,
            chunking_strategy=data.get("chunking_strategy", ""),
            sub_agents=sub_agents,
            synthesis_code=data.get("synthesis_code", "")
        )
    except json.JSONDecodeError as je:
        print(f"❌ JSON Decode Error: {je}")
        print(f"Cleaned text was:\n{cleaned_text}")
        return OrchidAction(
            agent_id=MODEL_NAME,
            chunking_strategy=f"FAILED_JSON_DECODE",
            sub_agents=[],
            synthesis_code="print('Error')"
        )
    except Exception as e:
        print(f"❌ Error during LLM API call/parsing: {e}")
        return OrchidAction(
            agent_id=MODEL_NAME,
            chunking_strategy=f"FAILED_VALIDATION_OR_API: {str(e)}",
            sub_agents=[],
            synthesis_code="print('Error')"
        )


async def main():
    print(f"🚀 Starting Verbose HuggingFace/OpenAI SDK Orchestrator Test...")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    
    if not API_KEY:
        print("❌ Error: HF_TOKEN or OPENAI_API_KEY environment variable is not set.")
        return

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    use_json_format = await check_json_support(client)

    # Note: Increased message_timeout_s to 600.0 (10 minutes)
    # The "1011 keepalive ping timeout" usually happens because the LLM takes too long to respond,
    # or the evaluation code blocks the event loop for too long. A 10-minute timeout helps prevent this.
    async with OrchidEnv(base_url="http://localhost:7860", connect_timeout_s=600.0, message_timeout_s=600.0) as env:
        print("🔄 Resetting environment (Provisioning MSB & Logs)...")
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
            
            # 1. Ask LLM for the plan
            action = await generate_orchestration(client, obs.task_description, obs.dataset_lines, feedback, use_json_format)
            
            # 2. Submit to Env
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
