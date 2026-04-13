"""LLM-based judge for task evaluation."""

from __future__ import annotations

import json
import os
import re

import litellm

from .key_manager import get_key_manager, is_rate_limit_error, parse_rate_limit_error

from dotenv import load_dotenv

load_dotenv()

JUDGE_SYSTEM = (
    "You are an impartial judge evaluating an AI agent's answer to a task. "
    "Given the task description, expected criteria, and the agent's answer, "
    "you must score it objectively."
)


JUDGE_PROMPT_TEMPLATE = """TASK DESCRIPTION:
{task_description}

EVALUATION CRITERIA:
{criteria}

EXPECTED ANSWER / KEY POINTS:
{expected}

AGENT'S FINAL ANSWER:
{answer}

Evaluate the agent's answer. Provide a JSON response with these keys:
{{
  "score": <float 0.0 to 1.0>,
  "success": <true if score >= 0.6, else false>,
  "reasoning": "<brief explanation>",
  "key_points_matched": <int>,
  "key_points_missed": <int>
}}

Be strict but fair. Partial credit is allowed. Return ONLY the JSON, no other text."""


MAX_RETRIES = 8


async def judge_answer(
    task_description: str,
    criteria: str,
    expected: str,
    answer: str,
    model: str = "groq/llama-3.3-70b-versatile",
    api_key: str | None = None,  # ignored
) -> dict:
    """Use LLM to judge an agent's answer. Returns dict with score, success, reasoning."""
    mgr = get_key_manager()

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_description=task_description[:1000],
        criteria=criteria[:1000],
        expected=expected[:1500],
        answer=answer[:2000],
    )

    last_err = None
    raw = ""
    for attempt in range(MAX_RETRIES):
        try:
            key = mgr.current_key()
        except RuntimeError as e:
            return {"score": 0.0, "success": False,
                    "reasoning": f"Judge failed: all keys exhausted ({e})",
                    "key_points_matched": 0, "key_points_missed": 0}

        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                api_key=key,
            )
            raw = response.choices[0].message.content or ""
            break
        except Exception as e:
            last_err = e
            if is_rate_limit_error(e):
                cooldown, tag = parse_rate_limit_error(str(e))
                mgr.mark_exhausted(key, cooldown_seconds=cooldown, reason=f"judge:{tag}")
                continue
            else:
                return {"score": 0.0, "success": False,
                        "reasoning": f"Judge error: {e}",
                        "key_points_matched": 0, "key_points_missed": 0}
    else:
        return {"score": 0.0, "success": False,
                "reasoning": f"Judge rate-limit retries exhausted: {last_err}",
                "key_points_matched": 0, "key_points_missed": 0}

    # Try to extract JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": 0.0, "success": False,
                "reasoning": f"Judge parse failed: {raw[:200]}",
                "key_points_matched": 0, "key_points_missed": 0}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"score": 0.0, "success": False,
                "reasoning": f"Judge JSON decode failed: {e}",
                "key_points_matched": 0, "key_points_missed": 0}

    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "success": bool(data.get("success", score >= 0.6)),
        "reasoning": str(data.get("reasoning", "")),
        "key_points_matched": int(data.get("key_points_matched", 0)),
        "key_points_missed": int(data.get("key_points_missed", 0)),
    }
