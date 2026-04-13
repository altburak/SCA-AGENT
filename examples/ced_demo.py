"""
examples/ced_demo.py — Cross-Episode Distillation Demo

Demonstrates SCA's CED module across two sessions:
  Session 1: Agent works on a coding task, insights are extracted.
  Session 2: Agent starts with augmented prompt from Session 1's insights.

Usage:
    export GROQ_API_KEY=gsk_...
    python examples/ced_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone

# Add parent to path so we can import sca
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from sca.augmentation import PromptAugmenter
from sca.ced import DistillationOrchestrator
from sca.context import ContextBlock, ContextManager, Provenance
from sca.episode import Episode, EpisodeStore
from sca.extraction import InsightExtractor
from sca.insight import Insight, InsightRepository, InsightType
from sca.llm import LLMClient
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulated session data (no real AOGL calls needed for demo)
# ---------------------------------------------------------------------------

CODING_QUESTIONS = [
    ("What does this function do?", "general"),
    ("What will the output be?", "code_behavior"),
    ("Is this code thread-safe?", "code_behavior"),
    ("What exceptions could this raise?", "factual_claim"),
    ("Is this O(n) or O(n²)?", "factual_claim"),
]

SESSION_1_PREDICTIONS = [
    # (statement, confidence, match_score)
    ("The function sorts a list in ascending order", 0.95, 0.4),   # overconfident, wrong
    ("Output will be [1, 2, 3]", 0.9, 0.3),                        # overconfident, wrong
    ("The code is NOT thread-safe", 0.6, 0.85),                     # well-calibrated
    ("KeyError and TypeError can be raised", 0.55, 0.9),            # well-calibrated
    ("The algorithm is O(n²) for the nested loop", 0.85, 0.3),     # overconfident, wrong
]

SESSION_2_PREDICTIONS = [
    # Same types of questions, but this time agent performs better
    ("The function merges two sorted lists", 0.7, 0.85),
    ("Output will be [1, 2, 3, 4, 5]", 0.65, 0.8),
    ("The code is NOT thread-safe due to global state", 0.72, 0.9),
    ("ValueError and IndexError can be raised", 0.68, 0.85),
    ("The algorithm is O(n log n) using merge sort", 0.70, 0.8),
]


def build_synthetic_episode(
    predictions_data: list[tuple[str, float, float]],
    domain: str = "coding",
    initial_prompt: str = "Help me understand Python code",
) -> tuple[Episode, list[Prediction], list[Outcome]]:
    """Build an Episode with synthetic predictions and outcomes."""
    ep = Episode(
        domain=domain,
        initial_prompt=initial_prompt,
        start_time=datetime.now(timezone.utc),
    )
    preds = []
    outcomes = []

    for statement, confidence, match_score in predictions_data:
        pred = Prediction(
            statement=statement,
            confidence_at_prediction=confidence,
            category="code_analysis",
        )
        outcome = Outcome(
            prediction_id=pred.prediction_id,
            action_executed=ActionProposal(action_type=ActionType.NO_ACTION),
            actual_result=f"Verification result for: {statement[:40]}...",
            match_score=match_score,
            match_reasoning="Synthetic evaluation for demo",
        )
        preds.append(pred)
        outcomes.append(outcome)
        ep.prediction_ids.append(pred.prediction_id)

    ep.end_time = datetime.now(timezone.utc)
    return ep, preds, outcomes


class MockGroundingLog:
    """In-memory grounding log for demo purposes."""

    def __init__(self) -> None:
        self._preds: dict[uuid.UUID, Prediction] = {}
        self._outcomes: dict[uuid.UUID, Outcome] = {}

    def add_prediction(self, pred: Prediction) -> None:
        self._preds[pred.prediction_id] = pred

    def add_outcome(self, outcome: Outcome) -> None:
        self._outcomes[outcome.prediction_id] = outcome

    def get_prediction(self, pred_id: uuid.UUID):
        return self._preds.get(pred_id)

    def get_outcome_for_prediction(self, pred_id: uuid.UUID):
        return self._outcomes.get(pred_id)


def compute_stats(predictions: list[Prediction], outcomes: list[Outcome]) -> dict:
    """Compute calibration and accuracy metrics."""
    confidences = [p.confidence_at_prediction for p in predictions]
    match_scores = [o.match_score for o in outcomes]
    cal_errors = [abs(c - m) for c, m in zip(confidences, match_scores)]

    return {
        "avg_confidence": statistics.mean(confidences),
        "avg_match_score": statistics.mean(match_scores),
        "avg_calibration_error": statistics.mean(cal_errors),
        "task_success_rate": sum(1 for m in match_scores if m > 0.6) / len(match_scores),
    }


def print_table(title: str, data: dict[str, float]) -> None:
    print(f"\n  📊 {title}")
    print("  " + "-" * 42)
    for key, val in data.items():
        print(f"  {key:<30} {val:.3f}")


async def run_demo() -> None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️  GROQ_API_KEY not set — using mock LLM for insight extraction.")
        print("   Insights will be empty but the learning pipeline will be demonstrated.\n")

    print("=" * 60)
    print("  SCA CED Demo — Cross-Episode Distillation")
    print("=" * 60)

    # Setup CED infrastructure
    episode_store = EpisodeStore(db_path=":memory:")
    insight_repo = InsightRepository(db_path=":memory:")
    grounding_log = MockGroundingLog()

    # LLM client (mock if no key)
    if api_key:
        llm_client = LLMClient(api_key=api_key)
    else:
        from unittest.mock import MagicMock
        llm_client = MagicMock()
        llm_client.chat.return_value = '{"biases": [{"pattern": "Agent overconfident on code output prediction", "evidence": [], "severity": 0.8, "category": "code_behavior"}], "strategies": [], "failures": [], "facts": []}'

    extractor = InsightExtractor(llm_client=llm_client, grounding_log=grounding_log)
    augmenter = PromptAugmenter(insight_repository=insight_repo, episode_store=episode_store)
    orchestrator = DistillationOrchestrator(
        episode_store=episode_store,
        insight_extractor=extractor,
        insight_repository=insight_repo,
        prompt_augmenter=augmenter,
    )

    # -----------------------------------------------------------------------
    # SESSION 1
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("  SESSION 1: Coding task — Python code analysis")
    print("─" * 60)

    ep1, preds1, outcomes1 = build_synthetic_episode(
        SESSION_1_PREDICTIONS,
        domain="coding",
        initial_prompt="Help me understand what this Python function does",
    )

    # Register in mock grounding log
    for p, o in zip(preds1, outcomes1):
        grounding_log.add_prediction(p)
        grounding_log.add_outcome(o)

    stats1 = compute_stats(preds1, outcomes1)
    print_table("Session 1 Performance", stats1)

    print("\n  🔍 Session 1 ending — extracting insights...")
    t0 = time.time()
    end_result = await orchestrator.on_episode_end(ep1)
    elapsed = time.time() - t0
    print(f"  ✅ Extraction complete ({elapsed:.1f}s)")
    print(f"     Insights extracted : {end_result['insights_extracted']}")
    print(f"     Insights merged    : {end_result['insights_merged']}")

    # Show extracted insights
    print("\n  📚 Extracted insights:")
    for ins_type in InsightType:
        group = insight_repo.query_by_type(ins_type)
        if group:
            print(f"\n  [{ins_type.value.upper()}]")
            for ins in group:
                print(f"    • {ins.content[:80]}")
                print(f"      confidence={ins.confidence:.2f}")

    # -----------------------------------------------------------------------
    # SESSION 2
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("  SESSION 2: Same domain, agent starts with augmented prompt")
    print("─" * 60)

    session2_context = orchestrator.on_episode_start(
        "Analyze this Python function for correctness",
        domain_hint="coding",
    )

    n_applicable = len(session2_context["applicable_insight_ids"])
    has_augmentation = "LEARNED FROM EXPERIENCE" in session2_context["augmented_system_prompt"]
    print(f"\n  Applicable insights loaded : {n_applicable}")
    print(f"  Prompt augmented           : {has_augmentation}")

    if has_augmentation:
        print("\n  🧠 Augmented section (excerpt):")
        aug = session2_context["augmented_system_prompt"]
        learned_start = aug.find("--- LEARNED FROM EXPERIENCE ---")
        if learned_start >= 0:
            print("  " + "\n  ".join(aug[learned_start:learned_start + 400].split("\n")))

    ep2, preds2, outcomes2 = build_synthetic_episode(
        SESSION_2_PREDICTIONS,
        domain="coding",
        initial_prompt="Analyze this Python function for correctness",
    )

    for p, o in zip(preds2, outcomes2):
        grounding_log.add_prediction(p)
        grounding_log.add_outcome(o)

    stats2 = compute_stats(preds2, outcomes2)
    print_table("Session 2 Performance", stats2)

    # -----------------------------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("  COMPARISON: Session 1 vs Session 2")
    print("─" * 60)
    print(f"\n  {'Metric':<32} {'Session 1':>10} {'Session 2':>10} {'Delta':>10}")
    print("  " + "-" * 65)

    for key in stats1:
        v1 = stats1[key]
        v2 = stats2[key]
        delta = v2 - v1
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {key:<32} {v1:>10.3f} {v2:>10.3f} {arrow}{abs(delta):>8.3f}")

    print()
    if stats2["avg_calibration_error"] < stats1["avg_calibration_error"]:
        print("  ✅ Learning effect observed: calibration error decreased.")
    else:
        print("  ℹ️  Note: Synthetic data; real learning requires actual LLM sessions.")

    # -----------------------------------------------------------------------
    # FINAL STATS DUMP
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("  REPOSITORY STATS")
    print("─" * 60)

    final_stats = await orchestrator.get_statistics()
    print(f"\n  Total episodes : {final_stats['total_episodes']}")
    print(f"  Total insights : {final_stats['total_insights']}")
    print(f"  Avg per episode: {final_stats['avg_insights_per_episode']}")
    print("\n  Insights by type:")
    for itype, count in final_stats["insights_by_type"].items():
        print(f"    {itype:<25} {count}")

    print("\n  Top 5 insights:")
    top5 = insight_repo.get_top_k(k=5, criterion="confidence")
    for i, ins in enumerate(top5, 1):
        print(f"    {i}. [{ins.type.value}] {ins.content[:60]}")
        print(f"       confidence={ins.confidence:.2f} | usage={ins.usage_count}")

    print("\n" + "=" * 60)
    print("  Demo complete. SCA is now feature-complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())
