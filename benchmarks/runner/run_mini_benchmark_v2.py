"""Run the mini benchmark: 10 tasks, baseline vs SCA v1 vs SCA v2.

Usage:
    python benchmarks/runner/run_mini_benchmark_v2.py
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
from benchmarks.runner.sca_agent import run_sca_agent
from benchmarks.runner.sca_v2_agent import run_sca_v2_agent
from benchmarks.runner.judge import judge_answer
from benchmarks.runner.key_manager import get_key_manager


TASKS_DIR = Path(__file__).parent.parent / "mini_tasks"
RESULTS_DIR = Path(__file__).parent.parent / "mini_results_v2"
RESULTS_DIR.mkdir(exist_ok=True)

PAUSE_BETWEEN_AGENTS_SEC = 3
PAUSE_BETWEEN_TASKS_SEC = 3

AGENT_NAMES = ["baseline", "sca_v1", "sca_v2"]


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
    elif agent_type == "sca_v1":
        result = await run_sca_agent(
            task_description=description,
            workspace_dir=str(workspace),
            max_steps=max_steps,
        )
    elif agent_type == "sca_v2":
        result = await run_sca_v2_agent(
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
        # v2 extras
        "prior": result.get("prior", ""),
        "prior_confidence": result.get("prior_confidence"),
        "verification_verdict": result.get("verification_verdict"),
        "verification_reasoning": result.get("verification_reasoning"),
        "judge_score": judgment["score"],
        "judge_success": judgment["success"],
        "judge_reasoning": judgment["reasoning"],
    }


async def main() -> None:
    print("=" * 70)
    print("  SCA Mini Benchmark v2 — Baseline vs SCA v1 vs SCA v2")
    print("  v2 = Pre-Answer Verification Loop")
    print("=" * 70)

    mgr = get_key_manager()

    task_dirs = sorted(p for p in TASKS_DIR.iterdir()
                       if p.is_dir() and p.name.startswith("task_"))
    if not task_dirs:
        print(f"No tasks found in {TASKS_DIR}")
        return

    print(f"\nFound {len(task_dirs)} tasks. Running 3 agents on each.\n")

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
                if agent_type == "sca_v2":
                    v = result.get("verification_verdict", "?")
                    extra = f" verdict={v}"
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

    # Save raw
    json_path = RESULTS_DIR / "raw_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved: {json_path}")

    # CSV summary
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "agent", "score", "success", "steps",
                         "tool_calls", "elapsed_s", "verdict", "error"])
        for r in all_results:
            writer.writerow([
                r.get("task_id", ""),
                r.get("agent", ""),
                round(r.get("judge_score", 0.0), 3),
                r.get("judge_success", False),
                r.get("steps_taken", 0),
                r.get("tool_calls_count", 0),
                r.get("elapsed_seconds", 0),
                r.get("verification_verdict", ""),
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

    print(f"\n  {'Agent':<12} {'Avg Score':>10} {'Success':>10} {'Clean':>10}")
    print(f"  {'-' * 50}")
    for a in AGENT_NAMES:
        rs = by_agent[a]
        clean_count = sum(1 for r in rs if is_clean(r))
        print(f"  {a:<12} {avg_score(rs):>10.3f} {success_rate(rs):>9.1%} "
              f"{clean_count:>6}/{len(rs)}")

    # Per-task breakdown
    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<35} {'Base':>7} {'v1':>7} {'v2':>7}  Winner")
    print(f"  {'-' * 70}")

    task_ids = sorted(set(r.get("task_id", "") for r in all_results))
    v2_wins = 0
    v1_wins = 0
    base_wins = 0
    ties = 0

    for tid in task_ids:
        scores = {}
        for a in AGENT_NAMES:
            r = next((r for r in by_agent[a] if r.get("task_id") == tid), None)
            scores[a] = r.get("judge_score", 0.0) if r else 0.0

        best = max(scores.values())
        winners = [a for a, s in scores.items() if abs(s - best) < 0.05]
        if len(winners) > 1:
            win_str = "tie"
            ties += 1
        elif "sca_v2" in winners:
            win_str = "SCA v2"
            v2_wins += 1
        elif "sca_v1" in winners:
            win_str = "SCA v1"
            v1_wins += 1
        else:
            win_str = "baseline"
            base_wins += 1

        print(f"  {tid:<35} {scores['baseline']:>7.2f} "
              f"{scores['sca_v1']:>7.2f} {scores['sca_v2']:>7.2f}  {win_str}")

    print(f"\n  Wins: baseline={base_wins}  v1={v1_wins}  v2={v2_wins}  ties={ties}")

    # v2-specific: verdict distribution
    v2_rs = by_agent["sca_v2"]
    verdicts: dict[str, int] = {}
    for r in v2_rs:
        v = r.get("verification_verdict") or "?"
        verdicts[v] = verdicts.get(v, 0) + 1
    print(f"\n  v2 verdict distribution: {verdicts}")

    print("\n" + "=" * 70)
    print(f"  Done. Results in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
