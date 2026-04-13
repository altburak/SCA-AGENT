"""
tests/test_ced_integration.py — CED integration tests
"""

import asyncio
import json
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from sca.augmentation import PromptAugmenter
from sca.ced import DistillationOrchestrator, LoRADistillationHook
from sca.episode import Episode, EpisodeStore
from sca.extraction import InsightExtractor
from sca.insight import Insight, InsightRepository, InsightType
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(statement: str = "The sky is blue") -> Prediction:
    return Prediction(
        statement=statement,
        confidence_at_prediction=0.8,
        category="general",
    )


def make_outcome(pred_id: uuid.UUID, score: float = 0.75) -> Outcome:
    return Outcome(
        prediction_id=pred_id,
        action_executed=ActionProposal(action_type=ActionType.NO_ACTION),
        actual_result="Confirmed",
        match_score=score,
    )


def make_mock_extractor(insights: list[Insight] | None = None) -> InsightExtractor:
    extractor = MagicMock(spec=InsightExtractor)
    extractor.extract_all = AsyncMock(return_value=insights or [])
    return extractor


def make_orchestrator(
    insights_to_extract: list[Insight] | None = None,
) -> tuple[DistillationOrchestrator, EpisodeStore, InsightRepository]:
    episode_store = EpisodeStore(db_path=":memory:")
    insight_repo = InsightRepository(db_path=":memory:")
    extractor = make_mock_extractor(insights_to_extract or [])
    augmenter = PromptAugmenter(insight_repository=insight_repo, episode_store=episode_store)
    orchestrator = DistillationOrchestrator(
        episode_store=episode_store,
        insight_extractor=extractor,
        insight_repository=insight_repo,
        prompt_augmenter=augmenter,
    )
    return orchestrator, episode_store, insight_repo


# ---------------------------------------------------------------------------
# on_episode_end tests
# ---------------------------------------------------------------------------

def test_on_episode_end_saves_episode():
    orchestrator, episode_store, _ = make_orchestrator()
    ep = Episode(domain="coding", initial_prompt="Help me debug")
    asyncio.run(orchestrator.on_episode_end(ep))
    loaded = episode_store.load_episode(ep.episode_id)
    assert loaded is not None
    assert loaded.episode_id == ep.episode_id


def test_on_episode_end_returns_stats():
    insights = [
        Insight(type=InsightType.BIAS_PATTERN, content="Bias one", confidence=0.7),
        Insight(type=InsightType.SUCCESSFUL_STRATEGY, content="Strategy one", confidence=0.6),
    ]
    orchestrator, _, _ = make_orchestrator(insights)
    ep = Episode(domain="coding")
    result = asyncio.run(orchestrator.on_episode_end(ep))
    assert "insights_extracted" in result
    assert result["insights_extracted"] == 2


def test_on_episode_end_stores_insights():
    insights = [
        Insight(type=InsightType.BIAS_PATTERN, content="Overconfident on SQL queries", confidence=0.75),
    ]
    orchestrator, _, insight_repo = make_orchestrator(insights)
    ep = Episode(domain="research")
    asyncio.run(orchestrator.on_episode_end(ep))
    stored = insight_repo.query_by_type(InsightType.BIAS_PATTERN)
    assert len(stored) == 1
    assert "SQL" in stored[0].content


def test_on_episode_end_extraction_failure_graceful():
    episode_store = EpisodeStore(db_path=":memory:")
    insight_repo = InsightRepository(db_path=":memory:")
    extractor = MagicMock(spec=InsightExtractor)
    extractor.extract_all = AsyncMock(side_effect=RuntimeError("LLM down"))
    augmenter = PromptAugmenter(insight_repository=insight_repo)
    orchestrator = DistillationOrchestrator(
        episode_store=episode_store,
        insight_extractor=extractor,
        insight_repository=insight_repo,
        prompt_augmenter=augmenter,
    )
    ep = Episode(domain="coding")
    result = asyncio.run(orchestrator.on_episode_end(ep))
    assert result["insights_extracted"] == 0


# ---------------------------------------------------------------------------
# on_episode_start tests
# ---------------------------------------------------------------------------

def test_on_episode_start_returns_required_keys():
    orchestrator, _, _ = make_orchestrator()
    result = orchestrator.on_episode_start("Help me with Python", "coding")
    assert "augmented_system_prompt" in result
    assert "few_shot_examples" in result
    assert "applicable_insight_ids" in result


def test_on_episode_start_augmented_prompt_non_empty():
    orchestrator, _, _ = make_orchestrator()
    result = orchestrator.on_episode_start("Debug my Python script")
    assert len(result["augmented_system_prompt"]) > 0


def test_on_episode_start_with_insights_augments():
    episode_store = EpisodeStore(db_path=":memory:")
    insight_repo = InsightRepository(db_path=":memory:")
    insight_repo.add_insight(Insight(
        type=InsightType.BIAS_PATTERN,
        content="Overconfident on Python typing",
        confidence=0.8,
        applicability={"domain": None, "category": None, "keywords": []},
    ))
    extractor = make_mock_extractor([])
    augmenter = PromptAugmenter(insight_repository=insight_repo, episode_store=episode_store)
    orchestrator = DistillationOrchestrator(
        episode_store=episode_store,
        insight_extractor=extractor,
        insight_repository=insight_repo,
        prompt_augmenter=augmenter,
    )
    result = orchestrator.on_episode_start("Python typing question")
    assert "LEARNED FROM EXPERIENCE" in result["augmented_system_prompt"]


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

def test_get_statistics():
    orchestrator, episode_store, insight_repo = make_orchestrator()
    episode_store.save_episode(Episode(domain="coding"))
    insight_repo.add_insight(Insight(type=InsightType.BIAS_PATTERN, content="Bias", confidence=0.6))
    stats = asyncio.run(orchestrator.get_statistics())
    assert stats["total_episodes"] >= 1
    assert stats["total_insights"] >= 1
    assert "insights_by_type" in stats


# ---------------------------------------------------------------------------
# StratifiedAgent backward compat
# ---------------------------------------------------------------------------

def test_agent_without_ced_chat_works():
    """Agent without CED should work exactly as before."""
    from unittest.mock import MagicMock, patch
    from sca.context import ContextManager
    from sca.agent import StratifiedAgent

    psm = ContextManager()
    csm = MagicMock()
    csm.score = AsyncMock(return_value=MagicMock(final_score=0.8))
    aogl = MagicMock()
    aogl.run_full_cycle = AsyncMock(return_value=(make_prediction(), None))
    llm = MagicMock()
    llm.chat.return_value = "Hello from agent"

    agent = StratifiedAgent(
        psm_manager=psm,
        csm_scorer=csm,
        aogl_controller=aogl,
        main_llm=llm,
        ced_orchestrator=None,
    )
    response = asyncio.run(agent.chat("Hello"))
    assert "Hello from agent" in response


def test_agent_start_end_session_without_ced():
    from sca.context import ContextManager
    from sca.agent import StratifiedAgent

    psm = ContextManager()
    csm = MagicMock()
    aogl = MagicMock()
    llm = MagicMock()

    agent = StratifiedAgent(psm_manager=psm, csm_scorer=csm, aogl_controller=aogl, main_llm=llm)
    result_start = asyncio.run(agent.start_session("Hello"))
    result_end = asyncio.run(agent.end_session())
    assert result_start == {}
    assert result_end == {}


def test_agent_async_context_manager_without_ced():
    from sca.context import ContextManager
    from sca.agent import StratifiedAgent

    psm = ContextManager()
    csm = MagicMock()
    aogl = MagicMock()
    llm = MagicMock()
    llm.chat.return_value = "Test response"
    csm.score = AsyncMock(return_value=MagicMock(final_score=0.9))
    aogl.run_full_cycle = AsyncMock(return_value=(make_prediction(), None))

    async def _run():
        agent = StratifiedAgent(
            psm_manager=psm, csm_scorer=csm, aogl_controller=aogl, main_llm=llm
        )
        async with agent as a:
            resp = await a.chat("Hello world")
        return resp

    result = asyncio.run(_run())
    assert isinstance(result, str)


def test_lora_hook_raises_not_implemented():
    class ConcreteHook(LoRADistillationHook):
        def train_lora_from_insights(self, insights):
            return super().train_lora_from_insights(insights)

    hook = ConcreteHook()
    with pytest.raises(NotImplementedError):
        hook.train_lora_from_insights([])


# ---------------------------------------------------------------------------
# End-to-end: episode -> extract -> store -> augment
# ---------------------------------------------------------------------------

def test_end_to_end_learning_pipeline():
    """Full pipeline: episode ends, insights stored, next session augmented."""
    insight_to_inject = Insight(
        type=InsightType.BIAS_PATTERN,
        content="Agent underestimates edge cases in Python code",
        confidence=0.8,
        applicability={"domain": "coding", "category": None, "keywords": []},
    )

    episode_store = EpisodeStore(db_path=":memory:")
    insight_repo = InsightRepository(db_path=":memory:")
    extractor = make_mock_extractor([insight_to_inject])
    augmenter = PromptAugmenter(insight_repository=insight_repo, episode_store=episode_store)
    orchestrator = DistillationOrchestrator(
        episode_store=episode_store,
        insight_extractor=extractor,
        insight_repository=insight_repo,
        prompt_augmenter=augmenter,
    )

    # Session 1 ends
    ep1 = Episode(domain="coding", initial_prompt="Debug Python edge cases")
    stats = asyncio.run(orchestrator.on_episode_end(ep1))
    assert stats["insights_extracted"] == 1

    # Session 2 starts — should see the learned insight
    context = orchestrator.on_episode_start("Python code review", domain_hint="coding")
    assert "LEARNED FROM EXPERIENCE" in context["augmented_system_prompt"]
    assert "edge cases" in context["augmented_system_prompt"].lower()
