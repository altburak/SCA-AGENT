"""
SCA v2 — Pre-Answer Verification Agent

Architecture: Prior Belief → Evidence → Contradiction Check → Calibrated Answer

Flow:
1. Model produces a PRIOR answer without tools (its internal belief).
2. Tools run, gather EVIDENCE from reality.
3. A lightweight contradiction check asks: "Does evidence support prior, contradict it, or is it insufficient?"
4. Based on contradiction outcome, final answer is produced with appropriate confidence:
   - AGREEMENT → confident answer
   - CONTRADICTION → trust evidence, flag the disagreement
   - INSUFFICIENT → say "I don't know" or "cannot verify"

This differs from post-hoc detectors (Patronus/Vectara) which check AFTER generation.
We check DURING generation, using evidence to calibrate confidence before answering.
"""

from __future__ import annotations

import json
import re
from typing import Any

import litellm

from .tools import TOOL_SCHEMAS, TaskTools, execute_tool_call
from .key_manager import get_key_manager, is_rate_limit_error, parse_rate_limit_error

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TOOL_FORMAT_DIRECTIVE = (
    "\n\nCRITICAL: Use the structured tool_calls mechanism to call tools. "
    "NEVER write tool calls as text like <function=name>{...}</function>. "
    "If calling a tool, leave response content empty; the API handles it."
)

# Stage 1: Prior belief (no tools)
PRIOR_SYSTEM = (
    "You are an AI assistant answering a question based ONLY on your internal knowledge. "
    "You have NO access to tools, files, or external information. "
    "Give your best answer based on what you know. "
    "Be concise. If you genuinely don't know, say 'I don't know'. "
    "Do NOT make up file contents or specific values you cannot know."
)

# Stage 2: Evidence gathering (tools)
EVIDENCE_SYSTEM = (
    "You are an AI assistant gathering evidence to answer a question. "
    "Use the tools to read files, list directories, and execute code as needed. "
    "Your job is NOT to answer yet — just gather relevant facts. "
    "When you have enough evidence, respond with a concise summary of what you found."
    + TOOL_FORMAT_DIRECTIVE
)

# Stage 3: Contradiction check
CONTRADICTION_SYSTEM = (
    "You are a verification judge. You will be given a question, a PRIOR answer "
    "(based on internal knowledge without tools), and EVIDENCE (from tool-based "
    "investigation of the actual environment). "
    "Your job: determine if the evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT "
    "for the prior answer. Respond ONLY in JSON:\n"
    '{"verdict": "SUPPORT|CONTRADICT|INSUFFICIENT", '
    '"reasoning": "<one sentence>", '
    '"confidence_adjustment": <float: suggested confidence 0.0-1.0>}'
)

# Stage 4: Final answer (calibrated)
FINAL_SYSTEM_TEMPLATE = """You are producing a final answer for the user.

You have THREE sources of information, tagged with provenance:

1. [PRIOR — conf: {prior_conf:.2f}] — what your internal knowledge said BEFORE seeing evidence.
2. [EVIDENCE — conf: 0.95] — what tool-based investigation actually found in the environment.
3. [VERIFICATION — verdict: {verdict}] — whether the evidence supports or contradicts prior.

RULES:
- EVIDENCE is ground truth. When it contradicts PRIOR, trust EVIDENCE and say so.
- If verdict is INSUFFICIENT and you genuinely don't know, say "I don't know" or express low confidence.
- Never invent specifics (line numbers, values) that are not in EVIDENCE.
- Be concise and direct in your final answer."""


MAX_RATE_LIMIT_RETRIES = 8
MAX_TOOL_FORMAT_RETRIES = 2
MAX_EVIDENCE_STEPS = 6


# ---------------------------------------------------------------------------
# LLM call with retries
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
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Any:
    """Sync LLM call with key rotation + tool-format retry."""
    mgr = get_key_manager()
    last_err = None

    for _ in range(MAX_RATE_LIMIT_RETRIES):
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
                        "content": (
                            "REMINDER: Use structured tool_calls, not <function=...> text."
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


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _stage_1_prior(model: str, task: str) -> tuple[str, float]:
    """Get model's prior belief WITHOUT tools.

    Returns (prior_text, estimated_confidence).
    Confidence heuristic: if prior says 'I don't know' or similar, low confidence.
    """
    messages = [
        {"role": "system", "content": PRIOR_SYSTEM},
        {"role": "user", "content": task},
    ]
    resp = _llm_call(model, messages, tools=None, max_tokens=512)
    prior = (resp.choices[0].message.content or "").strip()

    # Heuristic confidence from prior
    lower = prior.lower()
    if any(p in lower for p in ["i don't know", "i cannot", "i'm not sure",
                                 "unable to", "no access", "cannot determine"]):
        conf = 0.2
    elif any(p in lower for p in ["without access", "without seeing",
                                   "need to see", "would need"]):
        conf = 0.3
    else:
        conf = 0.5  # default: claim without verification
    return prior, conf


def _stage_2_evidence(
    model: str,
    task: str,
    tools_handle: TaskTools,
) -> tuple[str, list[dict]]:
    """Gather evidence using tools. Returns (evidence_summary, tool_log)."""
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
            # Model decided it has enough evidence
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

    if not evidence_text:
        # Evidence not summarized; synthesize from tool results
        parts = []
        for t in tool_log:
            parts.append(
                f"- Called {t['tool']}({json.dumps(t['args'])}):\n"
                f"  {t['result_preview'][:200]}"
            )
        evidence_text = "Evidence collected:\n" + "\n".join(parts) if parts else "(no evidence)"

    return evidence_text, tool_log


def _stage_3_verify(
    model: str, task: str, prior: str, evidence: str
) -> dict:
    """Check if evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT for prior."""
    payload = (
        f"QUESTION:\n{task}\n\n"
        f"PRIOR ANSWER (from internal knowledge, no tools):\n{prior}\n\n"
        f"EVIDENCE (from tool-based investigation):\n{evidence[:3000]}\n\n"
        f"Verdict?"
    )
    messages = [
        {"role": "system", "content": CONTRADICTION_SYSTEM},
        {"role": "user", "content": payload},
    ]
    try:
        resp = _llm_call(model, messages, tools=None, max_tokens=256, temperature=0.0)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"verdict": "INSUFFICIENT",
                "reasoning": f"verification_failed: {e}",
                "confidence_adjustment": 0.3}

    # Extract JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"verdict": "INSUFFICIENT",
                "reasoning": "parse_failed",
                "confidence_adjustment": 0.3}
    try:
        data = json.loads(m.group(0))
        verdict = str(data.get("verdict", "INSUFFICIENT")).upper()
        if verdict not in ("SUPPORT", "CONTRADICT", "INSUFFICIENT"):
            verdict = "INSUFFICIENT"
        conf = float(data.get("confidence_adjustment", 0.5))
        conf = max(0.0, min(1.0, conf))
        return {
            "verdict": verdict,
            "reasoning": str(data.get("reasoning", "")),
            "confidence_adjustment": conf,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"verdict": "INSUFFICIENT",
                "reasoning": "parse_failed",
                "confidence_adjustment": 0.3}


def _stage_4_final(
    model: str,
    task: str,
    prior: str,
    prior_conf: float,
    evidence: str,
    verification: dict,
) -> str:
    """Produce final answer using all stratified context."""
    system_content = FINAL_SYSTEM_TEMPLATE.format(
        prior_conf=prior_conf,
        verdict=verification["verdict"],
    )

    user_content = (
        f"QUESTION:\n{task}\n\n"
        f"[PRIOR — conf: {prior_conf:.2f}]\n{prior}\n\n"
        f"[EVIDENCE — conf: 0.95]\n{evidence[:3000]}\n\n"
        f"[VERIFICATION — verdict: {verification['verdict']}]\n"
        f"Reasoning: {verification['reasoning']}\n"
        f"Suggested confidence: {verification['confidence_adjustment']:.2f}\n\n"
        f"Now give the final answer to the user, following the rules."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    resp = _llm_call(model, messages, tools=None, max_tokens=1024)
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_sca_v2_agent(
    task_description: str,
    workspace_dir: str,
    model: str = "groq/llama-3.3-70b-versatile",
    max_steps: int = 15,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run SCA v2 Pre-Answer Verification agent on a task."""
    get_key_manager()
    tools_handle = TaskTools(workspace_dir)

    final_answer = ""
    error = None
    stages_completed = 0
    prior = ""
    prior_conf = 0.5
    evidence = ""
    verification = {"verdict": "INSUFFICIENT", "reasoning": "", "confidence_adjustment": 0.3}
    tool_log: list[dict] = []

    try:
        # Stage 1: Prior belief
        prior, prior_conf = _stage_1_prior(model, task_description)
        stages_completed = 1

        # Stage 2: Evidence
        evidence, tool_log = _stage_2_evidence(model, task_description, tools_handle)
        stages_completed = 2

        # Stage 3: Verify
        verification = _stage_3_verify(model, task_description, prior, evidence)
        stages_completed = 3

        # Stage 4: Final calibrated answer
        final_answer = _stage_4_final(
            model, task_description, prior, prior_conf, evidence, verification
        )
        stages_completed = 4

    except Exception as e:
        error = str(e)
        if not final_answer:
            # Fallback: use evidence if we have it
            if evidence:
                final_answer = f"(partial, error in final stage) Evidence: {evidence[:800]}"
            else:
                final_answer = f"(error: {e})"

    provenance_distribution = {
        "PRIOR": 1,
        "EVIDENCE": len(tool_log),
        "VERIFICATION": 1 if stages_completed >= 3 else 0,
    }

    return {
        "final_answer": final_answer,
        "steps_taken": stages_completed,
        "tool_calls": tool_log,
        "provenance_distribution": provenance_distribution,
        "total_context_blocks": 3 + len(tool_log),
        "error": error,
        # Extras for analysis
        "prior": prior,
        "prior_confidence": prior_conf,
        "evidence_summary": evidence[:500],
        "verification_verdict": verification["verdict"],
        "verification_reasoning": verification["reasoning"],
    }
