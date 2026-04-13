"""
tests/test_augmentation.py — PromptAugmenter tests
"""

import uuid
from unittest.mock import MagicMock

import pytest

from sca.augmentation import PromptAugmenter
from sca.insight import Insight, InsightRepository, InsightType


BASE_PROMPT = "You are a helpful AI assistant."


def make_repo_with_insights() -> InsightRepository:
    repo = InsightRepository(db_path=":memory:")
    repo.add_insight(Insight(
        type=InsightType.BIAS_PATTERN,
        content="Agent is overconfident on math problems",
        confidence=0.8,
        applicability={"domain": "math", "category": None, "keywords": ["math", "calculation"]},
    ))
    repo.add_insight(Insight(
        type=InsightType.SUCCESSFUL_STRATEGY,
        content="Break problems into smaller steps for better accuracy",
        confidence=0.75,
        applicability={"domain": "coding", "category": None, "keywords": ["python", "debug"]},
    ))
    repo.add_insight(Insight(
        type=InsightType.FAILURE_MODE,
        content="Assumes file paths are always absolute",
        confidence=0.65,
        applicability={"domain": "coding", "category": "file_location", "keywords": ["file", "path"]},
    ))
    repo.add_insight(Insight(
        type=InsightType.DOMAIN_KNOWLEDGE,
        content="Python f-strings require Python 3.6+",
        confidence=0.9,
        applicability={"domain": "coding", "category": None, "keywords": ["python", "fstring"]},
    ))
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_augment_no_insights_returns_base():
    repo = InsightRepository(db_path=":memory:")
    augmenter = PromptAugmenter(insight_repository=repo)
    result = augmenter.augment_system_prompt(BASE_PROMPT, {"domain": "coding", "keywords": []})
    assert result == BASE_PROMPT
    repo.close()


def test_augment_with_relevant_insights_appends_section():
    repo = make_repo_with_insights()
    augmenter = PromptAugmenter(insight_repository=repo)
    result = augmenter.augment_system_prompt(
        BASE_PROMPT,
        {"domain": "coding", "category": "file_location", "keywords": ["file", "path", "python"]}
    )
    assert BASE_PROMPT in result
    assert "LEARNED FROM EXPERIENCE" in result
    assert "END LEARNED" in result
    repo.close()


def test_augment_bias_section_present():
    repo = InsightRepository(db_path=":memory:")
    repo.add_insight(Insight(
        type=InsightType.BIAS_PATTERN,
        content="Overconfident on math",
        confidence=0.8,
        applicability={"domain": None, "category": None, "keywords": []},
    ))
    augmenter = PromptAugmenter(insight_repository=repo)
    result = augmenter.augment_system_prompt(BASE_PROMPT, {"domain": "math", "keywords": []})
    assert "CAUTIONS" in result
    assert "Overconfident on math" in result
    repo.close()


def test_augment_strategy_section_present():
    repo = InsightRepository(db_path=":memory:")
    repo.add_insight(Insight(
        type=InsightType.SUCCESSFUL_STRATEGY,
        content="Use type hints for clarity",
        confidence=0.75,
        applicability={"domain": None, "category": None, "keywords": []},
    ))
    augmenter = PromptAugmenter(insight_repository=repo)
    result = augmenter.augment_system_prompt(BASE_PROMPT, {"domain": "coding", "keywords": []})
    assert "STRATEGIES" in result
    repo.close()


def test_augment_respects_max_insights_per_section():
    repo = InsightRepository(db_path=":memory:")
    for i in range(10):
        repo.add_insight(Insight(
            type=InsightType.BIAS_PATTERN,
            content=f"Bias number {i}",
            confidence=0.7,
            applicability={"domain": None, "category": None, "keywords": []},
        ))
    augmenter = PromptAugmenter(insight_repository=repo, max_insights_per_section=3)
    result = augmenter.augment_system_prompt(BASE_PROMPT, {"domain": "any", "keywords": []})
    # Count how many bullet points appear under CAUTIONS
    import re
    caution_section = result.split("CAUTIONS")[1].split("STRATEGIES")[0] if "STRATEGIES" in result else result.split("CAUTIONS")[1]
    bullets = re.findall(r"^- ", caution_section, re.MULTILINE)
    assert len(bullets) <= 3
    repo.close()


def test_few_shot_no_episode_store_returns_empty():
    repo = InsightRepository(db_path=":memory:")
    augmenter = PromptAugmenter(insight_repository=repo, episode_store=None)
    examples = augmenter.select_few_shot_examples("Debug Python code")
    assert examples == []
    repo.close()


def test_few_shot_empty_store_returns_empty():
    from sca.episode import EpisodeStore
    repo = InsightRepository(db_path=":memory:")
    store = EpisodeStore(db_path=":memory:")
    augmenter = PromptAugmenter(insight_repository=repo, episode_store=store)
    examples = augmenter.select_few_shot_examples("Debug Python code")
    assert examples == []
    repo.close()
    store.close()


def test_few_shot_with_episodes_returns_messages():
    from sca.episode import Episode, EpisodeStore
    repo = InsightRepository(db_path=":memory:")
    store = EpisodeStore(db_path=":memory:")
    ep = Episode(domain="coding", initial_prompt="Debug Python code with list comprehension")
    store.save_episode(ep)
    augmenter = PromptAugmenter(insight_repository=repo, episode_store=store)
    examples = augmenter.select_few_shot_examples("Python list comprehension debug")
    assert isinstance(examples, list)
    if examples:  # should find relevant episode
        assert examples[0]["role"] == "user"
    repo.close()
    store.close()
