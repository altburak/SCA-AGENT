"""
SCA v3.1 — Lean Adversarial Consensus

Fixes from v3:
- 5 roles → 3 roles (OPTIMIST / EVIDENCE_LITERAL / MINIMAL)
- Jaccard token overlap → LLM-based semantic consensus (does core fact match?)
- Shared evidence stage (not per-role)
- 4 LLM calls total per task (was 7): evidence + 3 roles (parallel) + consensus
- Parallel role calls via asyncio + 8-key rotation
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import litellm

from .tools import TOOL_SCHEMAS, TaskTools, execute_tool_call
from .key_manager import get_key_manager, is_rate_limit_error, parse_rate_limit_error

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Role prompts — 3 carefully chosen orthogonal perspectives
# ---------------------------------------------------------------------------

ROLE_PROMPTS = {
    "OPTIMIST": (
        "You are a confident expert. Give a direct, committed answer based on evidence. "
        "Do not hedge. If the evidence suggests an answer, state it plainly."
    ),
    "EVIDENCE_LITERAL": (
        "You answer ONLY using facts that appear LITERALLY in the evidence. "
        "Do NOT infer, generalize, or use prior knowledge. "
        "Quote specific file/line/value references from the evidence. "
        "If evidence does not contain the answer, respond exactly: NOT_IN_EVIDENCE"
    ),
    "MINIMAL": (
        "You are a cautious expert. You only state what you are CERTAIN about. "
        "If ANY part of the answer has reasonable doubt, say: UNCERTAIN. "
        "If you are confident, give a minimal direct answer."
    ),
}

ROLE_ORDER = ["OPTIMIST", "EVIDENCE_LITERAL", "MINIMAL"]

TOOL_FORMAT_DIRECTIVE = (
    "\n\nCRITICAL: Use structured tool_calls only. "
    "NEVER emit <function=name>{...}</function> text."
)

EVIDENCE_SYSTEM = (
    "Gather evidence to answer the question. Use tools to read files, list directories, "
    "or execute code. Stop when you have enough evidence and summarize findings concisely."
    + TOOL_FORMAT_DIRECTIVE
)

# Consensus judge — checks whether the CORE FACT matches across roles
CONSENSUS_SYSTEM = """You judge whether 3 role-answers agree on the CORE FACT.

Ignore wording differences. Focus on whether the SUBSTANTIVE claim is the same
(same number, same function name, same file, same boolean, same list, etc.).

Respond ONLY in JSON:
{
  "agreement": "FULL" | "PARTIAL" | "NONE",
  "core_fact": "<the agreed-upon answer in one line, or 'none' if NONE>",
  "uncertainty_flags": <int: count of roles saying NOT_IN_EVIDENCE / UNCERTAIN>,
  "final_answer": "<the answer to give the user>",
  "confidence": <float 0.0-1.0>
}

Rules for final_answer:
- FULL agreement + uncertainty_flags == 0 → give confident answer
- PARTIAL agreement OR uncertainty_flags == 1 → give answer with a brief qualifier
- NONE agreement OR uncertainty_flags >= 2 → final_answer MUST be "I do not know with confidence. " followed by a one-sentence reason
- NEVER invent a compromise answer. Stick to what roles agreed on.
"""


MAX_RATE_LIMIT_RETRIES = 8
MAX_TOOL_FORMAT_RETRIES = 2
MAX_EVIDENCE_STEPS = 5


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _is_tool_format_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "tool_use_failed" in msg
        or "failed to call a function" in msg
        or "failed_generation" in msg
    )


def _llm_call(
    model: str,
    messages: list,
    tools: list | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
    preferred_key: str | None = None,
) -> Any:
    """Sync LLM call. Returns raw response object."""
    mgr = get_key_manager()
    last_err = None

    for _ in range(MAX_RATE_LIMIT_RETRIES):
        key = preferred_key
        if key is None:
            try:
                key = mgr.current_key()
            except RuntimeError as e:
                raise RuntimeError(f"All keys exhausted: {e}") from e

        local_messages = list(messages)
        for tf_attempt in range(MAX_TOOL_FORMAT_RETRIES + 1):
            try:
                kwargs = {
                    "model": model,
                    "messages": local_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "api_key": key,
                }
                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                return litellm.completion(**kwargs)
            except Exception as e:
                last_err = e
                if _is_tool_format_error(e) and tf_attempt < MAX_TOOL_FORMAT_RETRIES:
                    local_messages = local_messages + [{
                        "role": "user",
                        "content": "REMINDER: Use structured tool_calls, not <function=...> text.",
                    }]
                    continue
                elif is_rate_limit_error(e):
                    cooldown, tag = parse_rate_limit_error(str(e))
                    mgr.mark_exhausted(key, cooldown_seconds=cooldown, reason=tag)
                    preferred_key = None
                    break
                else:
                    raise

    raise RuntimeError(f"Call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Stage 1: Evidence gathering (tools)
# ---------------------------------------------------------------------------

def _gather_evidence(
    model: str, task: str, tools_handle: TaskTools
) -> tuple[str, list[dict]]:
    messages = [
        {"role": "system", "content": EVIDENCE_SYSTEM},
        {"role": "user", "content": task},
    ]
    tool_log: list[dict] = []
    evidence_text = ""

    for step in range(MAX_EVIDENCE_STEPS):
        resp = _llm_call(model, messages, tools=TOOL_SCHEMAS, max_tokens=1024)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in tool_calls
            ] if tool_calls else None,
        })

        if not tool_calls:
            evidence_text = msg.content or ""
            break

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = execute_tool_call(tools_handle, tc.function.name, args)
            tool_log.append({
                "step": step + 1,
                "tool": tc.function.name,
                "args": args,
                "result_preview": result[:300],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    if not evidence_text and tool_log:
        parts = [
            f"- {t['tool']}({json.dumps(t['args'])}):\n{t['result_preview'][:500]}"
            for t in tool_log
        ]
        evidence_text = "Evidence:\n" + "\n".join(parts)

    return evidence_text, tool_log


# ---------------------------------------------------------------------------
# Stage 2: 3 parallel role calls
# ---------------------------------------------------------------------------

async def _call_role(
    model: str, role: str, task: str, evidence: str, key_override: str
) -> str:
    system = ROLE_PROMPTS[role]
    user = (
        f"QUESTION:\n{task}\n\n"
        f"EVIDENCE:\n{evidence[:3500]}\n\n"
        f"Your answer (follow your role's rules strictly):"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: _llm_call(
                model, messages, tools=None,
                max_tokens=400, temperature=0.3,
                preferred_key=key_override,
            ),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ROLE_ERROR: {str(e)[:120]}]"


async def _parallel_role_panel(
    model: str, task: str, evidence: str
) -> dict[str, str]:
    mgr = get_key_manager()
    keys: list[str] = []
    for _ in range(3):
        try:
            keys.append(mgr.current_key())
        except RuntimeError:
            if keys:
                keys.append(keys[-1])
            else:
                raise

    coros = [
        _call_role(model, role, task, evidence, keys[i])
        for i, role in enumerate(ROLE_ORDER)
    ]
    answers = await asyncio.gather(*coros)
    return dict(zip(ROLE_ORDER, answers))


# ---------------------------------------------------------------------------
# Stage 3: Semantic consensus via LLM
# ---------------------------------------------------------------------------

_UNCERTAIN_TOKENS_RE = re.compile(
    r"\b(NOT_IN_EVIDENCE|UNCERTAIN|I DO NOT KNOW|I DON'T KNOW|INSUFFICIENT)\b",
    re.IGNORECASE,
)


def _count_uncertainty_flags(answers: dict[str, str]) -> int:
    count = 0
    for a in answers.values():
        if _UNCERTAIN_TOKENS_RE.search(a or ""):
            count += 1
    return count


def _semantic_consensus(
    model: str, task: str, answers: dict[str, str]
) -> dict:
    """LLM judges whether the 3 role-answers agree on the CORE FACT."""
    panel_text = "\n\n".join(
        f"[{role}]\n{answers[role]}" for role in ROLE_ORDER
    )
    user = (
        f"QUESTION:\n{task}\n\n"
        f"THREE ROLE ANSWERS:\n{panel_text}\n\n"
        f"Judge agreement. Return ONLY JSON (no prose)."
    )
    messages = [
        {"role": "system", "content": CONSENSUS_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        resp = _llm_call(model, messages, tools=None,
                         max_tokens=512, temperature=0.0)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {
            "agreement": "NONE",
            "core_fact": "none",
            "uncertainty_flags": _count_uncertainty_flags(answers),
            "final_answer": f"(consensus_failed: {e}) Best effort: {answers.get('OPTIMIST', '')[:400]}",
            "confidence": 0.2,
            "raw": "",
        }

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {
            "agreement": "NONE",
            "core_fact": "none",
            "uncertainty_flags": _count_uncertainty_flags(answers),
            "final_answer": f"(parse_failed) {answers.get('OPTIMIST', '')[:400]}",
            "confidence": 0.2,
            "raw": raw[:300],
        }

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {
            "agreement": "NONE",
            "core_fact": "none",
            "uncertainty_flags": _count_uncertainty_flags(answers),
            "final_answer": f"(json_decode_failed) {answers.get('OPTIMIST', '')[:400]}",
            "confidence": 0.2,
            "raw": raw[:300],
        }

    agreement = str(data.get("agreement", "NONE")).upper()
    if agreement not in ("FULL", "PARTIAL", "NONE"):
        agreement = "NONE"
    flags = int(data.get("uncertainty_flags", _count_uncertainty_flags(answers)))
    conf = float(data.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))
    final = str(data.get("final_answer", "")).strip()
    if not final:
        if agreement == "NONE" or flags >= 2:
            final = "I do not know with confidence. The role panel did not converge."
        else:
            final = str(data.get("core_fact", answers.get("OPTIMIST", "")))[:600]

    return {
        "agreement": agreement,
        "core_fact": str(data.get("core_fact", "none")),
        "uncertainty_flags": flags,
        "final_answer": final,
        "confidence": conf,
        "raw": raw[:300],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_sca_v31_agent(
    task_description: str,
    workspace_dir: str,
    model: str = "groq/llama-3.3-70b-versatile",
    max_steps: int = 15,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run SCA v3.1 lean adversarial consensus agent."""
    get_key_manager()
    tools_handle = TaskTools(workspace_dir)

    error = None
    evidence = ""
    tool_log: list[dict] = []
    answers: dict[str, str] = {}
    consensus: dict = {}
    final_answer = ""

    try:
        # Stage 1: Evidence
        evidence, tool_log = _gather_evidence(model, task_description, tools_handle)

        # Stage 2: 3 roles in parallel
        answers = await _parallel_role_panel(model, task_description, evidence)

        # Stage 3: Semantic consensus
        consensus = _semantic_consensus(model, task_description, answers)
        final_answer = consensus["final_answer"]

    except Exception as e:
        error = str(e)
        if not final_answer:
            if answers:
                final_answer = answers.get("OPTIMIST") or answers.get("EVIDENCE_LITERAL") or f"(error: {e})"
            elif evidence:
                final_answer = f"(error) Evidence: {evidence[:600]}"
            else:
                final_answer = f"(error: {e})"

    provenance_distribution = {
        "EVIDENCE": len(tool_log),
        "ROLE_PANEL": len(answers),
        "CONSENSUS": 1 if consensus else 0,
    }

    return {
        "final_answer": final_answer,
        "steps_taken": 1 + len(tool_log) + 1,
        "tool_calls": tool_log,
        "provenance_distribution": provenance_distribution,
        "total_context_blocks": len(tool_log) + len(answers) + 1,
        "error": error,
        "role_answers": answers,
        "consensus": consensus,
        "evidence_summary": evidence[:500],
    }
