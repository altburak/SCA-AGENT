"""Run the mini benchmark: 10 tasks, baseline vs SCA.

Usage:
    python benchmarks/runner/run_mini_benchmark.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

from benchmarks.runner.baseline_agent import run_baseline_agent
from benchmarks.runner.sca_agent import run_sca_agent
from benchmarks.runner.judge import judge_answer
from benchmarks.runner.key_manager import get_key_manager


TASKS_DIR = Path(__file__).parent.parent / "mini_tasks"
RESULTS_DIR = Path(__file__).parent.parent / "mini_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Small pause between runs to spread load across keys
PAUSE_BETWEEN_AGENTS_SEC = 3
PAUSE_BETWEEN_TASKS_SEC = 3


def load_task(task_dir: Path) -> dict:
    """Load a task's task.json."""
    with open(task_dir / "task.json", "r", encoding="utf-8") as f:
        return json.load(f)


def format_expected(task: dict) -> str:
    """Extract expected answer / criteria from task dict into a string."""
    parts = []
    for key in ("expected_answer", "expected_findings", "expected_inconsistencies",
                "truth", "bug_location", "actual_output"):
        if key in task:
            parts.append(f"{key}: {json.dumps(task[key], indent=2)}")
    return "\n\n".join(parts) if parts else "(no explicit expected answer)"


async def run_single_task(task_dir: Path, agent_type: str) -> dict:
    """Run one task with one agent type."""
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
    elif agent_type == "sca":
        result = await run_sca_agent(
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
        "judge_score": judgment["score"],
        "judge_success": judgment["success"],
        "judge_reasoning": judgment["reasoning"],
    }


async def main() -> None:
    print("=" * 70)
    print("  SCA Mini Benchmark — Baseline vs SCA")
    print("  10 real tasks, real tool calls, LLM judge")
    print("=" * 70)

    # Initialize key manager (prints how many keys loaded)
    mgr = get_key_manager()

    task_dirs = sorted(p for p in TASKS_DIR.iterdir() if p.is_dir() and p.name.startswith("task_"))
    if not task_dirs:
        print(f"No tasks found in {TASKS_DIR}")
        return

    print(f"\nFound {len(task_dirs)} tasks. Running both agents on each.\n")

    all_results: list[dict] = []

    for i, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        print(f"[{i}/{len(task_dirs)}] {task_name}")

        for j, agent_type in enumerate(("baseline", "sca")):
            print(f"    Running {agent_type}...", end=" ", flush=True)
            try:
                result = await run_single_task(task_dir, agent_type)
                all_results.append(result)
                status = "✓" if result["judge_success"] else "✗"
                err_note = ""
                if result.get("error"):
                    err_short = result["error"][:80].replace("\n", " ")
                    err_note = f" [err: {err_short}...]"
                print(f"{status} score={result['judge_score']:.2f} "
                      f"steps={result['steps_taken']} ({result['elapsed_seconds']}s){err_note}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "task_id": task_name,
                    "agent": agent_type,
                    "error": str(e),
                    "judge_score": 0.0,
                    "judge_success": False,
                })

            if j == 0:
                await asyncio.sleep(PAUSE_BETWEEN_AGENTS_SEC)

        if i < len(task_dirs):
            await asyncio.sleep(PAUSE_BETWEEN_TASKS_SEC)

    # Save raw results
    json_path = RESULTS_DIR / "raw_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved: {json_path}")

    # Save CSV summary
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "agent", "score", "success", "steps",
                         "tool_calls", "elapsed_s", "error"])
        for r in all_results:
            writer.writerow([
                r.get("task_id", ""),
                r.get("agent", ""),
                round(r.get("judge_score", 0.0), 3),
                r.get("judge_success", False),
                r.get("steps_taken", 0),
                r.get("tool_calls_count", 0),
                r.get("elapsed_seconds", 0),
                (r.get("error") or "")[:100],
            ])
    print(f"Summary CSV saved: {csv_path}")

    # Key usage report
    print(f"\n  Key usage:")
    print(mgr.status())

    # Aggregate comparison
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    baseline_results = [r for r in all_results if r.get("agent") == "baseline"]
    sca_results = [r for r in all_results if r.get("agent") == "sca"]

    def is_clean(r: dict) -> bool:
        err = (r.get("error") or "").lower()
        return "rate_limit" not in err and "ratelimit" not in err and "exhausted" not in err

    clean_baseline = [r for r in baseline_results if is_clean(r)]
    clean_sca = [r for r in sca_results if is_clean(r)]

    def avg_score(results: list) -> float:
        scores = [r.get("judge_score", 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0

    def success_rate(results: list) -> float:
        total = len(results)
        wins = sum(1 for r in results if r.get("judge_success"))
        return wins / total if total else 0.0

    print(f"\n  Clean runs (no rate-limit errors):")
    print(f"    Baseline: {len(clean_baseline)}/{len(baseline_results)}  "
          f"SCA: {len(clean_sca)}/{len(sca_results)}")

    print(f"\n  Metric                  Baseline    SCA       Delta")
    print(f"  {'-' * 55}")
    b_avg = avg_score(baseline_results)
    s_avg = avg_score(sca_results)
    print(f"  Avg score (all)         {b_avg:.3f}       {s_avg:.3f}     "
          f"{s_avg - b_avg:+.3f}")

    b_succ = success_rate(baseline_results)
    s_succ = success_rate(sca_results)
    print(f"  Success rate (all)      {b_succ:.1%}       {s_succ:.1%}     "
          f"{(s_succ - b_succ) * 100:+.1f}pp")

    # Matched clean pairs
    by_tid_baseline = {r.get("task_id"): r for r in baseline_results}
    by_tid_sca = {r.get("task_id"): r for r in sca_results}
    matched_tids = {
        tid for tid in by_tid_baseline
        if tid in by_tid_sca
        and is_clean(by_tid_baseline[tid])
        and is_clean(by_tid_sca[tid])
    }
    matched_b = [by_tid_baseline[t] for t in matched_tids]
    matched_s = [by_tid_sca[t] for t in matched_tids]
    if matched_tids:
        print(f"\n  Matched clean pairs: {len(matched_tids)}")
        print(f"  Avg score (matched)     {avg_score(matched_b):.3f}       "
              f"{avg_score(matched_s):.3f}     "
              f"{avg_score(matched_s) - avg_score(matched_b):+.3f}")
        b_m_succ = success_rate(matched_b)
        s_m_succ = success_rate(matched_s)
        print(f"  Success rate (matched)  {b_m_succ:.1%}       {s_m_succ:.1%}     "
              f"{(s_m_succ - b_m_succ) * 100:+.1f}pp")

    # Per-task breakdown
    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<35} {'Baseline':>10} {'SCA':>10}  Winner")
    print(f"  {'-' * 68}")

    task_ids = sorted(set(r.get("task_id", "") for r in all_results))
    for tid in task_ids:
        b = next((r for r in baseline_results if r.get("task_id") == tid), None)
        s = next((r for r in sca_results if r.get("task_id") == tid), None)
        if not b or not s:
            continue
        b_score = b.get("judge_score", 0.0)
        s_score = s.get("judge_score", 0.0)
        b_err = "!" if not is_clean(b) else " "
        s_err = "!" if not is_clean(s) else " "
        winner = "SCA" if s_score > b_score + 0.05 else (
            "baseline" if b_score > s_score + 0.05 else "tie"
        )
        print(f"  {tid:<35} {b_score:>9.2f}{b_err} {s_score:>9.2f}{s_err}  {winner}")

    print(f"\n  (! = agent hit rate-limit or other error)")
    print("\n" + "=" * 70)
    print(f"  Done. Results in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
