"""Run the v3 benchmark: 10 tasks, baseline vs SCA v3.

Usage:
    python benchmarks/runner/run_mini_benchmark_v3.py
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
from benchmarks.runner.sca_v3_agent import run_sca_v3_agent
from benchmarks.runner.judge import judge_answer
from benchmarks.runner.key_manager import get_key_manager


TASKS_DIR = Path(__file__).parent.parent / "mini_tasks"
RESULTS_DIR = Path(__file__).parent.parent / "mini_results_v3"
RESULTS_DIR.mkdir(exist_ok=True)

PAUSE_BETWEEN_AGENTS_SEC = 3
PAUSE_BETWEEN_TASKS_SEC = 4

AGENT_NAMES = ["baseline", "sca_v3"]


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
    elif agent_type == "sca_v3":
        result = await run_sca_v3_agent(
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
        "consensus_stats": result.get("consensus_stats"),
        "judge_score": judgment["score"],
        "judge_success": judgment["success"],
        "judge_reasoning": judgment["reasoning"],
    }


async def main() -> None:
    print("=" * 70)
    print("  SCA v3 Benchmark — Baseline vs Adversarial Multi-Path Consensus")
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
            print(f"    {agent_type:>8}...", end=" ", flush=True)
            try:
                result = await run_single_task(task_dir, agent_type)
                all_results.append(result)
                status = "✓" if result["judge_success"] else "✗"
                err_note = ""
                if result.get("error"):
                    err_short = result["error"][:60].replace("\n", " ")
                    err_note = f" [err: {err_short}...]"
                extra = ""
                if agent_type == "sca_v3":
                    cs = result.get("consensus_stats") or {}
                    extra = (
                        f" uncertain={cs.get('uncertain_roles', '?')}/5 "
                        f"agree={cs.get('agreement_score', '?')}"
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
                         "tool_calls", "elapsed_s", "uncertain_roles",
                         "agreement", "error"])
        for r in all_results:
            cs = r.get("consensus_stats") or {}
            writer.writerow([
                r.get("task_id", ""),
                r.get("agent", ""),
                round(r.get("judge_score", 0.0), 3),
                r.get("judge_success", False),
                r.get("steps_taken", 0),
                r.get("tool_calls_count", 0),
                r.get("elapsed_seconds", 0),
                cs.get("uncertain_roles", ""),
                cs.get("agreement_score", ""),
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

    def avg_score(rs: list) -> float:
        s = [r.get("judge_score", 0.0) for r in rs]
        return sum(s) / len(s) if s else 0.0

    def success_rate(rs: list) -> float:
        total = len(rs)
        wins = sum(1 for r in rs if r.get("judge_success"))
        return wins / total if total else 0.0

    b_avg = avg_score(by_agent["baseline"])
    v3_avg = avg_score(by_agent["sca_v3"])
    b_succ = success_rate(by_agent["baseline"])
    v3_succ = success_rate(by_agent["sca_v3"])

    print(f"\n  Metric                  Baseline    SCA v3    Delta")
    print(f"  {'-' * 55}")
    print(f"  Avg score               {b_avg:.3f}       {v3_avg:.3f}     {v3_avg - b_avg:+.3f}")
    print(f"  Success rate            {b_succ:.1%}       {v3_succ:.1%}     "
          f"{(v3_succ - b_succ) * 100:+.1f}pp")

    # Per-task
    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<35} {'Base':>7} {'v3':>7}  Uncert  Agree  Winner")
    print(f"  {'-' * 75}")

    task_ids = sorted(set(r.get("task_id", "") for r in all_results))
    v3_wins = base_wins = ties = 0
    for tid in task_ids:
        b = next((r for r in by_agent["baseline"] if r.get("task_id") == tid), None)
        v = next((r for r in by_agent["sca_v3"] if r.get("task_id") == tid), None)
        if not b or not v:
            continue
        b_s = b.get("judge_score", 0.0)
        v_s = v.get("judge_score", 0.0)
        cs = v.get("consensus_stats") or {}
        unc = cs.get("uncertain_roles", "?")
        agr = cs.get("agreement_score", "?")

        if v_s > b_s + 0.05:
            w = "SCA v3"
            v3_wins += 1
        elif b_s > v_s + 0.05:
            w = "baseline"
            base_wins += 1
        else:
            w = "tie"
            ties += 1
        print(f"  {tid:<35} {b_s:>7.2f} {v_s:>7.2f}  {unc:>5}/5  {agr:>5}  {w}")

    print(f"\n  Wins: baseline={base_wins}  v3={v3_wins}  ties={ties}")

    # Calibration signal: did uncertainty correlate with failure?
    print(f"\n  CALIBRATION ANALYSIS (v3):")
    v3_rs = by_agent["sca_v3"]
    high_uncertain = [r for r in v3_rs
                      if (r.get("consensus_stats") or {}).get("uncertain_roles", 0) >= 2]
    low_uncertain = [r for r in v3_rs
                     if (r.get("consensus_stats") or {}).get("uncertain_roles", 0) < 2]
    if high_uncertain:
        print(f"    Tasks where ≥2 roles uncertain: {len(high_uncertain)}  "
              f"avg_score={avg_score(high_uncertain):.2f}")
    if low_uncertain:
        print(f"    Tasks where <2 roles uncertain: {len(low_uncertain)}  "
              f"avg_score={avg_score(low_uncertain):.2f}")
    print(f"    → Expected: uncertain tasks score LOWER (model correctly knew it didn't know)")

    print("\n" + "=" * 70)
    print(f"  Done. Results in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
