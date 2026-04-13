"""SCA agent: uses PSM context management with provenance tagging."""

from __future__ import annotations

import json
import os
from typing import Any

import litellm

from sca.context import ContextBlock, ContextManager, Provenance
from sca.formatter import PromptFormatter

from .tools import TOOL_SCHEMAS, TaskTools, execute_tool_call
from .key_manager import get_key_manager, is_rate_limit_error, parse_rate_limit_error

from dotenv import load_dotenv

load_dotenv()

TOOL_FORMAT_DIRECTIVE = (
    "\n\nCRITICAL TOOL USAGE RULES:\n"
    "- To call a tool, use the structured tool_calls mechanism ONLY.\n"
    "- NEVER write tool calls as text like <function=name>{...}</function>.\n"
    "- NEVER emit raw function syntax in your response content.\n"
    "- If you need to call a tool, the API will handle it — just specify the tool name and arguments.\n"
    "- Your response content should be either (a) natural-language reasoning/final answer, "
    "or (b) empty when making tool calls."
)

SCA_BASE_SYSTEM = (
    "You are a helpful AI assistant that completes tasks. "
    "You have access to tools to read files, list directories, and execute code. "
    "Use the tools to investigate, then provide a clear final answer. "
    "Be concise and direct."
    + TOOL_FORMAT_DIRECTIVE
)

MAX_RATE_LIMIT_RETRIES = 8
MAX_TOOL_FORMAT_RETRIES = 2


def _is_tool_format_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "tool_use_failed" in msg
        or "failed to call a function" in msg
        or "failed_generation" in msg
    )


def _llm_call_with_rotation_sync(
    model: str,
    messages: list,
    tools: list,
    max_tokens: int = 1024,
) -> Any:
    """Sync litellm call with key rotation + tool-format retry."""
    mgr = get_key_manager()
    last_err = None

    for rl_attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            key = mgr.current_key()
        except RuntimeError as e:
            raise RuntimeError(f"All keys exhausted: {e}") from e

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
                    local_messages = local_messages + [{
                        "role": "user",
                        "content": (
                            "REMINDER: Your previous attempt failed because you wrote "
                            "tool calls as text (e.g. <function=name>{...}</function>). "
                            "Do NOT do this. Use the structured tool_calls API."
                        ),
                    }]
                    continue
                elif is_rate_limit_error(e):
                    cooldown, tag = parse_rate_limit_error(str(e))
                    mgr.mark_exhausted(key, cooldown_seconds=cooldown, reason=tag)
                    break
                else:
                    raise

    raise RuntimeError(f"Call failed after all retries: {last_err}")


async def run_sca_agent(
    task_description: str,
    workspace_dir: str,
    model: str = "groq/llama-3.3-70b-versatile",
    max_steps: int = 15,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run SCA agent (with PSM provenance tagging) on a task."""
    get_key_manager()

    tools = TaskTools(workspace_dir)
    context = ContextManager()
    try:
        formatter = PromptFormatter(mode="default")
    except TypeError:
        formatter = PromptFormatter()

    context.add(ContextBlock(
        content=task_description,
        provenance=Provenance.USER,
        confidence=1.0,
    ))

    steps = 0
    tool_call_log: list[dict] = []
    final_answer = ""
    error = None

    try:
        for _ in range(max_steps):
            steps += 1

            provenance_prompt = formatter.get_system_prompt()
            system_content = SCA_BASE_SYSTEM + "\n\n" + provenance_prompt
            context_text = formatter.format_all(context)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": context_text},
            ]

            response = _llm_call_with_rotation_sync(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                final_answer = msg.content or ""
                if final_answer.strip():
                    context.add(ContextBlock(
                        content=final_answer,
                        provenance=Provenance.SELF_GENERATED,
                        confidence=0.7,
                    ))
                break

            if msg.content and msg.content.strip():
                context.add(ContextBlock(
                    content=f"(reasoning) {msg.content}",
                    provenance=Provenance.SELF_GENERATED,
                    confidence=0.6,
                ))

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool_call(tools, tc.function.name, args)

                tool_desc = f"(tool call: {tc.function.name}({json.dumps(args)}))"
                context.add(ContextBlock(
                    content=tool_desc,
                    provenance=Provenance.SELF_GENERATED,
                    confidence=0.9,
                    metadata={"tool_name": tc.function.name},
                ))
                context.add(ContextBlock(
                    content=f"Tool result:\n{result}",
                    provenance=Provenance.EXTERNAL_TOOL,
                    confidence=0.95,
                    metadata={"tool_name": tc.function.name},
                ))

                tool_call_log.append({
                    "step": steps,
                    "tool": tc.function.name,
                    "args": args,
                    "result_preview": result[:200],
                })

        if not final_answer:
            final_answer = "(no final answer produced within step limit)"

    except Exception as e:
        error = str(e)

    distribution: dict[str, int] = {}
    for block in context.blocks:
        key = block.provenance.value
        distribution[key] = distribution.get(key, 0) + 1

    return {
        "final_answer": final_answer,
        "steps_taken": steps,
        "tool_calls": tool_call_log,
        "provenance_distribution": distribution,
        "total_context_blocks": len(context.blocks),
        "error": error,
    }
