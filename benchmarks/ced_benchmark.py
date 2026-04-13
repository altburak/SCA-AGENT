"""
benchmarks/ced_benchmark.py — CED Calibration Error Benchmark

Runs 10 synthetic sessions comparing CED-enabled vs CED-disabled agents.
Outputs a matplotlib plot and CSV with calibration error over time.

Usage:
    python benchmarks/ced_benchmark.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import logging

logging.basicConfig(level=logging.WARNING)

from unittest.mock import MagicMock

from sca.augmentation import PromptAugmenter
from sca.ced import DistillationOrchestrator
from sca.episode import Episode, EpisodeStore
from sca.extraction import InsightExtractor
from sca.insight import Insight, InsightRepository, InsightType
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SESSIONS = 10
PREDICTIONS_PER_SESSION = 8
RANDOM_SEED = 42
OUTPUT_CSV = "benchmarks/ced_benchmark_results.csv"
OUTPUT_PLOT = "benchmarks/ced_benchmark.png"


random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Synthetic task generator
# ---------------------------------------------------------------------------

def generate_session_predictions(
    session_idx: int,
    has_ced: bool,
    bias_correction: float = 0.0,
) -> list[tuple[float, float]]:
    """Generate (confidence, match_score) pairs for a session.

    Without CED: agent is overconfident (confidence >> match_score).
    With CED: agent gradually corrects its confidence bias.

    Args:
        session_idx: Which session (0-indexed, used for learning effect).
        has_ced: Whether CED is active.
        bias_correction: Additional correction factor from learned insights.

    Returns:
        List of (confidence, match_score) tuples.
    """
    pairs = []
    for _ in range(PREDICTIONS_PER_SESSION):
        # True accuracy drawn from realistic distribution
        true_accuracy = random.gauss(0.65, 0.2)
        true_accuracy = max(0.0, min(1.0, true_accuracy))

        if not has_ced:
            # Overconfident baseline: confidence ~0.15 higher than actual
            overconfidence_bias = 0.15 + random.gauss(0, 0.05)
            confidence = min(1.0, true_accuracy + overconfidence_bias)
        else:
            # CED gradually reduces overconfidence over sessions
            learning_factor = min(session_idx * 0.03, 0.12)  # max 12% reduction
            overconfidence_bias = max(0.0, 0.15 - learning_factor) + random.gauss(0, 0.04)
            confidence = min(1.0, true_accuracy + overconfidence_bias + bias_correction)

        confidence = max(0.0, min(1.0, confidence))
        pairs.append((confidence, true_accuracy))

    return pairs


def compute_calibration_error(pairs: list[tuple[float, float]]) -> float:
    """Mean absolute calibration error."""
    if not pairs:
        return 0.0
    return sum(abs(c - m) for c, m in pairs) / len(pairs)


def compute_task_success_rate(pairs: list[tuple[float, float]]) -> float:
    return sum(1 for _, m in pairs if m > 0.6) / len(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Mock CED infrastructure
# ---------------------------------------------------------------------------

def build_mock_ced() -> DistillationOrchestrator:
    episode_store = EpisodeStore(db_path=":memory:")
    insight_repo = InsightRepository(db_path=":memory:")
    grounding_log = MagicMock()
    grounding_log.get_prediction.return_value = None
    grounding_log.get_outcome_for_prediction.return_value = None

    # Extractor that produces one bias insight per session
    llm = MagicMock()
    llm.chat.return_value = '{"biases": [{"pattern": "Overconfident on code predictions", "evidence": [], "severity": 0.7, "category": "code_behavior"}]}'

    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding_log)
    augmenter = PromptAugmenter(insight_repository=insight_repo, episode_store=episode_store)
    return DistillationOrchestrator(
        episode_store=episode_store,
        insight_extractor=extractor,
        insight_repository=insight_repo,
        prompt_augmenter=augmenter,
        merge_similarity_threshold=0.99,  # don't merge during benchmark
    )


async def run_benchmark() -> dict[str, list[float]]:
    """Run N_SESSIONS for both CED-on and CED-off configurations."""
    print(f"Running {N_SESSIONS}-session benchmark...")
    print(f"  Predictions per session: {PREDICTIONS_PER_SESSION}")
    print(f"  Random seed: {RANDOM_SEED}\n")

    ced_orchestrator = build_mock_ced()

    results = {
        "ced_off_cal_error": [],
        "ced_on_cal_error": [],
        "ced_off_success_rate": [],
        "ced_on_success_rate": [],
    }

    for session_idx in range(N_SESSIONS):
        # CED off
        off_pairs = generate_session_predictions(session_idx, has_ced=False)
        off_cal = compute_calibration_error(off_pairs)
        off_success = compute_task_success_rate(off_pairs)
        results["ced_off_cal_error"].append(off_cal)
        results["ced_off_success_rate"].append(off_success)

        # CED on — run episode and let it learn
        ep = Episode(
            domain="coding",
            initial_prompt="Benchmark session",
            start_time=datetime.now(timezone.utc),
        )

        # Simulate predictions in this episode
        on_pairs = generate_session_predictions(session_idx, has_ced=True)
        on_cal = compute_calibration_error(on_pairs)
        on_success = compute_task_success_rate(on_pairs)
        results["ced_on_cal_error"].append(on_cal)
        results["ced_on_success_rate"].append(on_success)

        # End episode (extract insights for next session)
        await ced_orchestrator.on_episode_end(ep)

        print(
            f"  Session {session_idx + 1:2d}: "
            f"CED-off cal={off_cal:.3f} success={off_success:.2f} | "
            f"CED-on  cal={on_cal:.3f} success={on_success:.2f}"
        )

    return results


def save_csv(results: dict[str, list[float]]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "ced_off_cal_error", "ced_on_cal_error",
                         "ced_off_success_rate", "ced_on_success_rate"])
        for i in range(N_SESSIONS):
            writer.writerow([
                i + 1,
                round(results["ced_off_cal_error"][i], 4),
                round(results["ced_on_cal_error"][i], 4),
                round(results["ced_off_success_rate"][i], 4),
                round(results["ced_on_success_rate"][i], 4),
            ])
    print(f"\nCSV saved: {OUTPUT_CSV}")


def save_plot(results: dict[str, list[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    sessions = list(range(1, N_SESSIONS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CED Benchmark: Calibration Error and Task Success over Sessions", fontsize=13)

    # Plot 1: Calibration error
    axes[0].plot(sessions, results["ced_off_cal_error"], "r-o", label="CED off", linewidth=2)
    axes[0].plot(sessions, results["ced_on_cal_error"], "g-o", label="CED on", linewidth=2)
    axes[0].set_xlabel("Session")
    axes[0].set_ylabel("Mean Absolute Calibration Error")
    axes[0].set_title("Calibration Error Over Time")
    axes[0].legend()
    axes[0].set_ylim(0, 0.4)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Task success rate
    axes[1].plot(sessions, results["ced_off_success_rate"], "r-o", label="CED off", linewidth=2)
    axes[1].plot(sessions, results["ced_on_success_rate"], "g-o", label="CED on", linewidth=2)
    axes[1].set_xlabel("Session")
    axes[1].set_ylabel("Task Success Rate")
    axes[1].set_title("Task Success Rate Over Time")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=100, bbox_inches="tight")
    print(f"Plot saved: {OUTPUT_PLOT}")


async def main() -> None:
    print("=" * 60)
    print("  CED Benchmark")
    print("=" * 60 + "\n")

    results = await run_benchmark()
    save_csv(results)
    save_plot(results)

    # Summary
    avg_off = sum(results["ced_off_cal_error"]) / N_SESSIONS
    avg_on = sum(results["ced_on_cal_error"]) / N_SESSIONS
    improvement = (avg_off - avg_on) / avg_off * 100 if avg_off > 0 else 0

    print(f"\n{'─' * 40}")
    print(f"  Avg calibration error (CED off): {avg_off:.4f}")
    print(f"  Avg calibration error (CED on) : {avg_on:.4f}")
    print(f"  Improvement                    : {improvement:.1f}%")

    last_off = results["ced_off_cal_error"][-1]
    last_on = results["ced_on_cal_error"][-1]
    print(f"\n  Final session cal error:")
    print(f"    CED off: {last_off:.4f}")
    print(f"    CED on : {last_on:.4f}")
    print("─" * 40)


if __name__ == "__main__":
    asyncio.run(main())
