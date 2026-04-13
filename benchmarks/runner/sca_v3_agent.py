"""
SCA v3 — Adversarial Multi-Path Consensus

Architecture: Same question asked by 5 DIFFERENT adversarial roles in PARALLEL.
Agreement across roles = true confidence. Disagreement = model doesn't know.

Roles:
1. OPTIMIST    — answer confidently from your knowledge
2. SKEPTIC     — find reasons the obvious answer might be wrong
3. EVIDENCE    — only report what tools literally showed, no inference
4. ADVERSARY   — what would someone trying to trick you say?
5. MINIMAL     — state only things you are 100% certain about

Post-hoc detectors (Patronus, Vectara) check AFTER generation.
This checks DURING generation by forcing disagreement if model is uncertain.

Key insight: a model CANNOT consistently hallucinate the same wrong thing
across 5 adversarial framings. Consistency = truth signal.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from typing import Any

import litellm

from .tools import TOOL_SCHEMAS, TaskTools, execute_tool_call
from .key_manager import get_key_manager, is_rate_limit_error, parse_rate_limit_error

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

ROLE_PROMPTS = {
    "OPTIMIST": (
        "You are a confident expert. Answer the question directly and concisely "
        "based on the evidence provided. Give your best answer. "
        "Do not hedge — commit to an answer."
    ),
    "SKEPTIC": (
        "You are a careful skeptic. Look at the evidence and ask: what is the MOST LIKELY "
        "WRONG interpretation? Then ignore that and give the correct answer. "
        "If the evidence is ambiguous or insufficient, say 'INSUFFICIENT EVIDENCE'."
    ),
    "EVIDENCE": (
        "You are a strict literalist. Answer ONLY using facts directly stated in the evidence. "
        "Do NOT infer, generalize, or use outside knowledge. "
        "If the evidence does not contain the answer, say 'NOT IN EVIDENCE'."
    ),
    "ADVERSARY": (
        "You are testing whether someone would be fooled by the question. "
        "The question may contain misleading claims or false premises. "
        "Check the evidence carefully. "
        "If the question's premise is wrong, say so clearly and correct it. "
        "Otherwise answer normally."
    ),
    "MINIMAL": (
        "You answer only what you are 100% certain about. "
        "If ANY part of the answer has doubt, mark it with [UNCERTAIN]. "
        "If the whole answer is uncertain, say 'I DO NOT KNOW'."
    ),
}

ROLE_ORDER = ["OPTIMIST", "SKEPTIC", "EVIDENCE", "ADVERSARY", "MINIMAL"]

TOOL_FORMAT_DIRECTIVE = (
    "\n\nCRITICAL: Use structured tool_calls only. "
    "NEVER emit <function=name>{...}</function> text."
)

EVIDENCE_SYSTEM = (
    "You are gathering evidence to answer a question. "
    "Use tools to read files, list directories, execute code. "
    "When you have enough evidence, stop and summarize findings."
    + TOOL_FORMAT_DIRECTIVE
)

CONSENSUS_SYSTEM = (
    "You are a judge synthesizing 5 expert opinions into a final answer.\n"
    "You will be given a question and 5 answers from different perspectives.\n"
    "Your job:\n"
    "1. Determine if the 5 answers AGREE (same core claim).\n"
    "2. If 4-5 agree → output that answer confidently.\n"
    "3. If 3 agree → output that answer but note uncertainty.\n"
    "4. If <3 agree OR any role says 'I DO NOT KNOW' / 'NOT IN EVIDENCE' / "
    "'INSUFFICIENT' → output 'I do not know with confidence' and explain why.\n"
    "NEVER invent a compromise answer. Stick to what roles actually agreed on.\n"
    "Be concise."
)


MAX_RATE_LIMIT_RETRIES = 8
MAX_TOOL_FORMAT_RETRIES = 2
MAX_EVIDENCE_STEPS = 5


# ---------------------------------------------------------------------------
# LLM call with rotation
# ---------------------------------------------------------------------------

def _is_tool_format_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "tool_use_failed" in msg
        or "failed to call a function" in msg
        or "failed_generation" in msg
    )


def _llm_call_sync(
    model: str,
    messages: list,
    tools: list | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
    preferred_key: str | None = None,
) -> str:
    """Sync LLM call. Returns content string. Uses key rotation on rate-limit."""
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
                resp = litellm.completion(**kwargs)
                msg = resp.choices[0].message
                # For tool-using calls, caller may want the raw message
                return resp  # caller handles
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
                    preferred_key = None  # next iter gets new key
                    break
                else:
                    raise

    raise RuntimeError(f"Call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Stage 1: Evidence gathering (sequential, with tools)
# ---------------------------------------------------------------------------

def _gather_evidence(model: str, task: str, tools_handle: TaskTools) -> tuple[str, list[dict]]:
    """Gather evidence using tools. Returns (evidence_summary, tool_log)."""
    messages = [
        {"role": "system", "content": EVIDENCE_SYSTEM},
        {"role": "user", "content": task},
    ]
    tool_log: list[dict] = []
    evidence_text = ""

    for step in range(MAX_EVIDENCE_STEPS):
        resp = _llm_call_sync(model, messages, tools=TOOL_SCHEMAS, max_tokens=1024)
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
            f"- {t['tool']}({json.dumps(t['args'])}):\n{t['result_preview'][:400]}"
            for t in tool_log
        ]
        evidence_text = "Evidence:\n" + "\n".join(parts)

    return evidence_text, tool_log


# ---------------------------------------------------------------------------
# Stage 2: 5 parallel role calls (distributed across keys)
# ---------------------------------------------------------------------------

async def _call_role(
    model: str,
    role: str,
    task: str,
    evidence: str,
    key_override: str,
) -> str:
    """Call one role with a dedicated key. Async via run_in_executor."""
    system = ROLE_PROMPTS[role]
    user = (
        f"QUESTION:\n{task}\n\n"
        f"EVIDENCE (from tool investigation):\n{evidence[:3000]}\n\n"
        f"Your answer (following your role's rules):"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: _llm_call_sync(
                model, messages, tools=None,
                max_tokens=400, temperature=0.3,
                preferred_key=key_override,
            ),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ROLE_ERROR: {str(e)[:100]}]"


async def _parallel_role_panel(
    model: str, task: str, evidence: str
) -> dict[str, str]:
    """Launch 5 roles in parallel, each on a different key."""
    mgr = get_key_manager()
    # Get 5 available keys (cycling through)
    keys: list[str] = []
    for _ in range(5):
        try:
            keys.append(mgr.current_key())
        except RuntimeError:
            # Fallback: reuse keys we already got
            if keys:
                keys.append(keys[0])
            else:
                raise

    tasks = [
        _call_role(model, role, task, evidence, keys[i])
        for i, role in enumerate(ROLE_ORDER)
    ]
    answers = await asyncio.gather(*tasks)
    return dict(zip(ROLE_ORDER, answers))


# ---------------------------------------------------------------------------
# Stage 3: Consensus analysis
# ---------------------------------------------------------------------------

_UNCERTAIN_MARKERS = [
    "i do not know", "i don't know", "not in evidence",
    "insufficient evidence", "insufficient", "cannot determine",
    "uncertain", "unknown", "no information",
]


def _has_uncertainty_marker(answer: str) -> bool:
    lower = answer.lower()
    return any(m in lower for m in _UNCERTAIN_MARKERS)


def _simple_consensus_score(answers: dict[str, str]) -> dict:
    """Quick heuristic: how many roles express uncertainty?
    How much do non-uncertain answers overlap (shared normalized tokens)?
    """
    uncertain_count = sum(1 for a in answers.values() if _has_uncertainty_marker(a))
    confident_answers = [a for a in answers.values() if not _has_uncertainty_marker(a)]

    # Very rough token overlap across confident answers
    if len(confident_answers) >= 2:
        token_sets = []
        for a in confident_answers:
            # Keep tokens of length >= 4 (skip stopwords-ish)
            toks = {
                w.lower().strip(".,!?;:()[]\"'")
                for w in re.findall(r"[A-Za-z0-9_\-\.]+", a)
                if len(w) >= 4
            }
            token_sets.append(toks)
        # Pairwise Jaccard avg
        jaccards = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                a, b = token_sets[i], token_sets[j]
                u = len(a | b)
                if u == 0:
                    continue
                jaccards.append(len(a & b) / u)
        agreement = sum(jaccards) / len(jaccards) if jaccards else 0.0
    else:
        agreement = 0.0

    return {
        "uncertain_roles": uncertain_count,
        "confident_roles": len(confident_answers),
        "agreement_score": round(agreement, 3),
    }


def _synthesize_consensus(model: str, task: str, answers: dict[str, str]) -> str:
    """Use one LLM call to synthesize the 5 answers into a final answer."""
    panel_text = "\n\n".join(
        f"[{role}]\n{answers[role]}" for role in ROLE_ORDER
    )
    user_content = (
        f"QUESTION:\n{task}\n\n"
        f"FIVE ROLE ANSWERS:\n{panel_text}\n\n"
        f"Synthesize a final answer following your rules."
    )
    messages = [
        {"role": "system", "content": CONSENSUS_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = _llm_call_sync(model, messages, tools=None, max_tokens=512, temperature=0.0)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # Fallback: if synthesis fails, count-based output
        scores = _simple_consensus_score(answers)
        if scores["uncertain_roles"] >= 2:
            return "I do not know with confidence. The roles disagreed or flagged insufficient evidence."
        # Return OPTIMIST as best guess with warning
        return f"(synthesis_failed: {e}) Best guess: {answers.get('OPTIMIST', '')[:500]}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_sca_v3_agent(
    task_description: str,
    workspace_dir: str,
    model: str = "groq/llama-3.3-70b-versatile",
    max_steps: int = 15,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run SCA v3 Adversarial Multi-Path Consensus agent."""
    get_key_manager()
    tools_handle = TaskTools(workspace_dir)

    error = None
    evidence = ""
    tool_log: list[dict] = []
    answers: dict[str, str] = {}
    consensus_stats: dict = {}
    final_answer = ""

    try:
        # Stage 1: Evidence (sequential, tool use)
        evidence, tool_log = _gather_evidence(model, task_description, tools_handle)

        # Stage 2: 5 adversarial roles in PARALLEL (8 keys → at most 5 used simultaneously)
        answers = await _parallel_role_panel(model, task_description, evidence)

        # Stage 3a: Heuristic consensus stats
        consensus_stats = _simple_consensus_score(answers)

        # Stage 3b: LLM synthesis → final answer
        final_answer = _synthesize_consensus(model, task_description, answers)

    except Exception as e:
        error = str(e)
        if not final_answer:
            # Best-effort fallback
            if answers:
                final_answer = answers.get("OPTIMIST") or answers.get("EVIDENCE") or f"(error: {e})"
            elif evidence:
                final_answer = f"(error in synthesis) Evidence: {evidence[:800]}"
            else:
                final_answer = f"(error: {e})"

    provenance_distribution = {
        "EVIDENCE": len(tool_log),
        "ROLE_PANEL": len(answers),
        "CONSENSUS": 1 if final_answer else 0,
    }

    return {
        "final_answer": final_answer,
        "steps_taken": 1 + len(tool_log) + 1,  # evidence steps + role panel + consensus
        "tool_calls": tool_log,
        "provenance_distribution": provenance_distribution,
        "total_context_blocks": len(tool_log) + len(answers) + 1,
        "error": error,
        # v3-specific analytics
        "role_answers": answers,
        "consensus_stats": consensus_stats,
        "evidence_summary": evidence[:500],
    }
