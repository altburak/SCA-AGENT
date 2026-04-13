"""Baseline agent: vanilla Groq Llama 3.3 70B with tool use. No SCA features."""

from __future__ import annotations

import json
import os
from typing import Any

import litellm

from .tools import TOOL_SCHEMAS, TaskTools, execute_tool_call
from .key_manager import get_key_manager, is_rate_limit_error, parse_rate_limit_error

from dotenv import load_dotenv

load_dotenv()

# Negative constraint against Llama 3.3's malformed tool-call emission.
# The model sometimes emits <function=name>{...}</function> (old Llama 3.1 syntax)
# instead of using the structured tool_calls array. Groq rejects this as
# tool_use_failed. Explicit instruction prevents ~80% of these cases.
TOOL_FORMAT_DIRECTIVE = (
    "\n\nCRITICAL TOOL USAGE RULES:\n"
    "- To call a tool, use the structured tool_calls mechanism ONLY.\n"
    "- NEVER write tool calls as text like <function=name>{...}</function>.\n"
    "- NEVER emit raw function syntax in your response content.\n"
    "- If you need to call a tool, the API will handle it — just specify the tool name and arguments.\n"
    "- Your response content should be either (a) natural-language reasoning/final answer, "
    "or (b) empty when making tool calls."
)

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that completes tasks. "
    "You have access to tools to read files, list directories, and execute code. "
    "Use the tools to investigate, then provide a clear final answer. "
    "When you have enough information, provide your final answer in the response content "
    "(not as a tool call). Be concise and direct."
    + TOOL_FORMAT_DIRECTIVE
)

MAX_RATE_LIMIT_RETRIES = 8
MAX_TOOL_FORMAT_RETRIES = 2  # retry on malformed tool-call


def _is_tool_format_error(err: Exception) -> bool:
    """Detect Llama/Groq's 'failed to call function' malformed emission."""
    msg = str(err).lower()
    return (
        "tool_use_failed" in msg
        or "failed to call a function" in msg
        or "failed_generation" in msg
    )


def _llm_call_with_rotation(
    model: str,
    messages: list,
    tools: list,
    max_tokens: int = 1024,
) -> Any:
    """Sync litellm call with key rotation on rate limit + tool-format retry."""
    mgr = get_key_manager()
    last_err = None

    for rl_attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            key = mgr.current_key()
        except RuntimeError as e:
            raise RuntimeError(f"All keys exhausted: {e}") from e

        # Tool-format retry loop: if model emits malformed tool calls,
        # add a stern reminder message and try again (same key)
        local_messages = list(messages)
        for tf_attempt in range(MAX_TOOL_FORMAT_RETRIES + 1):
            try:
                response = litellm.completion(
                    model=model,
                    messages=local_messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=max_tokens,
                    api_key=key,
                )
                return response
            except Exception as e:
                last_err = e
                if _is_tool_format_error(e) and tf_attempt < MAX_TOOL_FORMAT_RETRIES:
                    # Prepend a user-role reminder and retry (same key)
                    local_messages = local_messages + [{
                        "role": "user",
                        "content": (
                            "REMINDER: Your previous attempt failed because you wrote "
                            "tool calls as text (e.g. <function=name>{...}</function>). "
                            "Do NOT do this. Use the structured tool_calls API. "
                            "If you need to call a tool, invoke it properly via the API, "
                            "not as inline text in your response."
                        ),
                    }]
                    continue
                elif is_rate_limit_error(e):
                    cooldown, tag = parse_rate_limit_error(str(e))
                    mgr.mark_exhausted(key, cooldown_seconds=cooldown, reason=tag)
                    break  # break tf loop, try next key in outer rl loop
                else:
                    # Non-retryable error
                    raise

    raise RuntimeError(
        f"Call failed after all retries: {last_err}"
    )


async def run_baseline_agent(
    task_description: str,
    workspace_dir: str,
    model: str = "groq/llama-3.3-70b-versatile",
    max_steps: int = 15,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a vanilla LLM agent on a task."""
    get_key_manager()

    tools = TaskTools(workspace_dir)
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": task_description},
    ]

    steps = 0
    tool_call_log: list[dict] = []
    final_answer = ""
    error = None

    try:
        for _ in range(max_steps):
            steps += 1
            response = _llm_call_with_rotation(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ] if tool_calls else None,
            })

            if not tool_calls:
                final_answer = msg.content or ""
                break

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool_call(tools, tc.function.name, args)
                tool_call_log.append({
                    "step": steps,
                    "tool": tc.function.name,
                    "args": args,
                    "result_preview": result[:200],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        if not final_answer:
            final_answer = "(no final answer produced within step limit)"

    except Exception as e:
        error = str(e)

    return {
        "final_answer": final_answer,
        "steps_taken": steps,
        "tool_calls": tool_call_log,
        "error": error,
    }
