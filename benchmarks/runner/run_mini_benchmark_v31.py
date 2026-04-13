"""Run the v3.1 benchmark: 10 tasks, baseline vs SCA v3.1.

Usage:
    python benchmarks/runner/run_mini_benchmark_v31.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

from benchmarks.runner.baseline_agent import run_baseline_agent
from benchmarks.runner.sca_v31_agent import run_sca_v31_agent
from benchmarks.runner.judge import judge_answer
from benchmarks.runner.key_manager import get_key_manager


TASKS_DIR = Path(__file__).parent.parent / "mini_tasks"
RESULTS_DIR = Path(__file__).parent.parent / "mini_results_v31"
RESULTS_DIR.mkdir(exist_ok=True)

PAUSE_BETWEEN_AGENTS_SEC = 4
PAUSE_BETWEEN_TASKS_SEC = 5

AGENT_NAMES = ["baseline", "sca_v31"]


def load_task(task_dir: Path) -> dict:
    with open(task_dir / "task.json", "r", encoding="utf-8") as f:
        return json.load(f)


def format_expected(task: dict) -> str:
    parts = []
    for key in ("expected_answer", "expected_findings", "expected_inconsistencies",
                "truth", "bug_location", "actual_output"):
        if key in task:
            parts.append(f"{key}: {json.dumps(task[key], indent=2)}")
    return "\n\n".join(parts) if parts else "(no explicit expected answer)"


async def run_single_task(task_dir: Path, agent_type: str) -> dict:
    task = load_task(task_dir)
    workspace = task_dir / "workspace"
    description = task["description"]
    max_steps = task.get("max_steps", 10)

    t0 = time.time()
    if agent_type == "baseline":
        result = await run_baseline_agent(
            task_description=description,
            workspace_dir=str(workspace),
            max_steps=max_steps,
        )
    elif agent_type == "sca_v31":
        result = await run_sca_v31_agent(
            task_description=description,
            workspace_dir=str(workspace),
            max_steps=max_steps,
        )
    else:
        raise ValueError(agent_type)
    elapsed = time.time() - t0

    judgment = await judge_answer(
        task_description=description,
        criteria=task.get("evaluation_criteria", ""),
        expected=format_expected(task),
        answer=result["final_answer"] or "(empty)",
    )

    return {
        "task_id": task["id"],
        "agent": agent_type,
        "final_answer": result["final_answer"],
        "steps_taken": result["steps_taken"],
        "tool_calls_count": len(result.get("tool_calls", [])),
        "elapsed_seconds": round(elapsed, 2),
        "error": result.get("error"),
        "provenance_distribution": result.get("provenance_distribution"),
        "role_answers": result.get("role_answers"),
        "consensus": result.get("consensus"),
        "judge_score": judgment["score"],
        "judge_success": judgment["success"],
        "judge_reasoning": judgment["reasoning"],
    }


async def main() -> None:
    print("=" * 70)
    print("  SCA v3.1 Benchmark — Lean Adversarial Consensus")
    print("  3 roles, semantic consensus, 4 LLM calls/task")
    print("=" * 70)

    mgr = get_key_manager()

    task_dirs = sorted(p for p in TASKS_DIR.iterdir()
                       if p.is_dir() and p.name.startswith("task_"))
    if not task_dirs:
        print(f"No tasks found in {TASKS_DIR}")
        return

    print(f"\nFound {len(task_dirs)} tasks.\n")

    all_results: list[dict] = []

    for i, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        print(f"[{i}/{len(task_dirs)}] {task_name}")

        for j, agent_type in enumerate(AGENT_NAMES):
            print(f"    {agent_type:>9}...", end=" ", flush=True)
            try:
                result = await run_single_task(task_dir, agent_type)
                all_results.append(result)
                status = "✓" if result["judge_success"] else "✗"
                err_note = ""
                if result.get("error"):
                    err_short = result["error"][:60].replace("\n", " ")
                    err_note = f" [err: {err_short}...]"
                extra = ""
                if agent_type == "sca_v31":
                    c = result.get("consensus") or {}
                    extra = (
                        f" agree={c.get('agreement', '?')} "
                        f"flags={c.get('uncertainty_flags', '?')} "
                        f"conf={c.get('confidence', 0.0):.2f}"
                    )
                print(f"{status} score={result['judge_score']:.2f} "
                      f"steps={result['steps_taken']} "
                      f"({result['elapsed_seconds']}s){extra}{err_note}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "task_id": task_name,
                    "agent": agent_type,
                    "error": str(e),
                    "judge_score": 0.0,
                    "judge_success": False,
                })

            if j < len(AGENT_NAMES) - 1:
                await asyncio.sleep(PAUSE_BETWEEN_AGENTS_SEC)

        if i < len(task_dirs):
            await asyncio.sleep(PAUSE_BETWEEN_TASKS_SEC)

    # Save
    json_path = RESULTS_DIR / "raw_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved: {json_path}")

    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "agent", "score", "success", "steps",
                         "tool_calls", "elapsed_s", "agreement",
                         "uncertainty_flags", "confidence", "error"])
        for r in all_results:
            c = r.get("consensus") or {}
            writer.writerow([
                r.get("task_id", ""),
                r.get("agent", ""),
                round(r.get("judge_score", 0.0), 3),
                r.get("judge_success", False),
                r.get("steps_taken", 0),
                r.get("tool_calls_count", 0),
                r.get("elapsed_seconds", 0),
                c.get("agreement", ""),
                c.get("uncertainty_flags", ""),
                c.get("confidence", ""),
                (r.get("error") or "")[:100],
            ])
    print(f"Summary CSV: {csv_path}")

    print(f"\n  Key usage:")
    print(mgr.status())

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    by_agent = {a: [r for r in all_results if r.get("agent") == a] for a in AGENT_NAMES}

    def is_clean(r: dict) -> bool:
        err = (r.get("error") or "").lower()
        return "rate_limit" not in err and "exhausted" not in err

    def avg_score(rs: list) -> float:
        s = [r.get("judge_score", 0.0) for r in rs]
        return sum(s) / len(s) if s else 0.0

    def success_rate(rs: list) -> float:
        total = len(rs)
        wins = sum(1 for r in rs if r.get("judge_success"))
        return wins / total if total else 0.0

    clean_b = [r for r in by_agent["baseline"] if is_clean(r)]
    clean_v = [r for r in by_agent["sca_v31"] if is_clean(r)]

    print(f"\n  Clean: baseline={len(clean_b)}/10  v3.1={len(clean_v)}/10")

    b_avg = avg_score(by_agent["baseline"])
    v_avg = avg_score(by_agent["sca_v31"])
    b_succ = success_rate(by_agent["baseline"])
    v_succ = success_rate(by_agent["sca_v31"])

    print(f"\n  Metric                   Baseline    v3.1     Delta")
    print(f"  {'-' * 55}")
    print(f"  Avg score (all)          {b_avg:.3f}      {v_avg:.3f}    {v_avg - b_avg:+.3f}")
    print(f"  Success rate (all)       {b_succ:.1%}      {v_succ:.1%}    "
          f"{(v_succ - b_succ) * 100:+.1f}pp")

    # Matched clean pairs
    by_tid_b = {r.get("task_id"): r for r in by_agent["baseline"]}
    by_tid_v = {r.get("task_id"): r for r in by_agent["sca_v31"]}
    matched = [t for t in by_tid_b if t in by_tid_v
               and is_clean(by_tid_b[t]) and is_clean(by_tid_v[t])]
    if matched:
        mb = [by_tid_b[t] for t in matched]
        mv = [by_tid_v[t] for t in matched]
        print(f"\n  Matched clean pairs: {len(matched)}")
        print(f"  Avg score (matched)      {avg_score(mb):.3f}      "
              f"{avg_score(mv):.3f}    {avg_score(mv) - avg_score(mb):+.3f}")

    # Per-task
    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<35} {'Base':>6} {'v3.1':>6}  Agree  Flg  Winner")
    print(f"  {'-' * 74}")

    task_ids = sorted(set(r.get("task_id", "") for r in all_results))
    v_wins = b_wins = ties = 0
    for tid in task_ids:
        b = by_tid_b.get(tid)
        v = by_tid_v.get(tid)
        if not b or not v:
            continue
        b_s = b.get("judge_score", 0.0)
        v_s = v.get("judge_score", 0.0)
        c = v.get("consensus") or {}
        agr = c.get("agreement", "?")[:4]
        flg = c.get("uncertainty_flags", "?")

        if v_s > b_s + 0.05:
            w = "v3.1"
            v_wins += 1
        elif b_s > v_s + 0.05:
            w = "baseline"
            b_wins += 1
        else:
            w = "tie"
            ties += 1
        print(f"  {tid:<35} {b_s:>6.2f} {v_s:>6.2f}  {agr:>5}  {flg:>3}  {w}")

    print(f"\n  Wins: baseline={b_wins}  v3.1={v_wins}  ties={ties}")

    # CALIBRATION: does consensus signal predict correctness?
    print(f"\n  CALIBRATION ANALYSIS (v3.1):")
    v_rs = by_agent["sca_v31"]

    # Group by consensus verdict
    full_agree = [r for r in v_rs if (r.get("consensus") or {}).get("agreement") == "FULL"
                  and (r.get("consensus") or {}).get("uncertainty_flags", 99) == 0]
    partial_or_flagged = [r for r in v_rs
                          if (r.get("consensus") or {}).get("agreement") == "PARTIAL"
                          or (r.get("consensus") or {}).get("uncertainty_flags", 0) >= 1]
    no_agree = [r for r in v_rs if (r.get("consensus") or {}).get("agreement") == "NONE"
                or (r.get("consensus") or {}).get("uncertainty_flags", 0) >= 2]

    print(f"    FULL agree + 0 flags   : n={len(full_agree):2}  "
          f"avg_score={avg_score(full_agree):.2f}  success_rate={success_rate(full_agree):.0%}")
    print(f"    PARTIAL or 1 flag      : n={len(partial_or_flagged):2}  "
          f"avg_score={avg_score(partial_or_flagged):.2f}  success_rate={success_rate(partial_or_flagged):.0%}")
    print(f"    NONE or 2+ flags       : n={len(no_agree):2}  "
          f"avg_score={avg_score(no_agree):.2f}  success_rate={success_rate(no_agree):.0%}")
    print(f"    → Good calibration: FULL > PARTIAL > NONE in score.")

    print("\n" + "=" * 70)
    print(f"  Done. Results in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
