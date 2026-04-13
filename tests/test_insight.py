"""
tests/test_insight.py — Insight model and InsightRepository tests
"""

import uuid
from datetime import datetime, timezone

import pytest

from sca.insight import Insight, InsightRepository, InsightType


def make_insight(**kwargs) -> Insight:
    defaults = dict(
        type=InsightType.BIAS_PATTERN,
        content="Agent is overconfident on math problems",
        confidence=0.7,
    )
    defaults.update(kwargs)
    return Insight(**defaults)


def make_repo() -> InsightRepository:
    return InsightRepository(db_path=":memory:")


# ---------------------------------------------------------------------------
# Insight model validators
# ---------------------------------------------------------------------------

def test_insight_default_fields():
    ins = make_insight()
    assert isinstance(ins.insight_id, uuid.UUID)
    assert ins.usage_count == 0
    assert ins.success_count == 0
    assert ins.evidence == []


def test_insight_content_empty_raises():
    with pytest.raises(ValueError):
        Insight(type=InsightType.BIAS_PATTERN, content="", confidence=0.5)


def test_insight_content_whitespace_raises():
    with pytest.raises(ValueError):
        Insight(type=InsightType.BIAS_PATTERN, content="   ", confidence=0.5)


def test_insight_confidence_below_zero_raises():
    with pytest.raises(ValueError):
        Insight(type=InsightType.BIAS_PATTERN, content="Valid", confidence=-0.1)


def test_insight_confidence_above_one_raises():
    with pytest.raises(ValueError):
        Insight(type=InsightType.BIAS_PATTERN, content="Valid", confidence=1.1)


def test_insight_confidence_boundary_values():
    ins_min = Insight(type=InsightType.DOMAIN_KNOWLEDGE, content="Fact", confidence=0.0)
    ins_max = Insight(type=InsightType.SUCCESSFUL_STRATEGY, content="Strategy", confidence=1.0)
    assert ins_min.confidence == 0.0
    assert ins_max.confidence == 1.0


def test_insight_success_rate_no_usage():
    ins = make_insight()
    assert ins.success_rate == 0.0


def test_insight_success_rate_computed():
    ins = make_insight()
    ins.usage_count = 10
    ins.success_count = 7
    assert abs(ins.success_rate - 0.7) < 1e-9


def test_all_insight_types_valid():
    for itype in InsightType:
        ins = Insight(type=itype, content=f"Content for {itype.value}", confidence=0.5)
        assert ins.type == itype


# ---------------------------------------------------------------------------
# InsightRepository CRUD
# ---------------------------------------------------------------------------

def test_add_and_get():
    repo = make_repo()
    ins = make_insight()
    repo.add_insight(ins)
    loaded = repo.get_insight(ins.insight_id)
    assert loaded is not None
    assert loaded.insight_id == ins.insight_id
    assert loaded.content == ins.content
    repo.close()


def test_get_nonexistent_returns_none():
    repo = make_repo()
    assert repo.get_insight(uuid.uuid4()) is None
    repo.close()


def test_query_by_type():
    repo = make_repo()
    repo.add_insight(make_insight(type=InsightType.BIAS_PATTERN, content="Bias one"))
    repo.add_insight(make_insight(type=InsightType.BIAS_PATTERN, content="Bias two"))
    repo.add_insight(make_insight(type=InsightType.SUCCESSFUL_STRATEGY, content="Strategy one"))
    biases = repo.query_by_type(InsightType.BIAS_PATTERN)
    strategies = repo.query_by_type(InsightType.SUCCESSFUL_STRATEGY)
    assert len(biases) == 2
    assert len(strategies) == 1
    repo.close()


def test_query_by_applicability_domain():
    repo = make_repo()
    ins_coding = make_insight(
        content="Coding insight",
        applicability={"domain": "coding", "category": None, "keywords": []},
    )
    ins_general = make_insight(
        content="General insight",
        applicability={"domain": None, "category": None, "keywords": []},
    )
    repo.add_insight(ins_coding)
    repo.add_insight(ins_general)
    results = repo.query_by_applicability(domain="coding", category=None, keywords=[])
    ids = {r.insight_id for r in results}
    assert ins_coding.insight_id in ids
    repo.close()


def test_get_top_k_confidence():
    repo = make_repo()
    for i, conf in enumerate([0.3, 0.9, 0.6, 0.1]):
        repo.add_insight(make_insight(content=f"Insight {i}", confidence=conf))
    top2 = repo.get_top_k(k=2, criterion="confidence")
    assert len(top2) == 2
    assert top2[0].confidence >= top2[1].confidence
    repo.close()


def test_get_top_k_usage_count():
    repo = make_repo()
    for i in range(4):
        ins = make_insight(content=f"Insight {i}")
        ins.usage_count = i * 10
        repo.add_insight(ins)
    top2 = repo.get_top_k(k=2, criterion="usage_count")
    assert top2[0].usage_count >= top2[1].usage_count
    repo.close()


def test_record_usage_increments():
    repo = make_repo()
    ins = make_insight()
    repo.add_insight(ins)
    repo.record_usage(ins.insight_id, was_successful=True)
    repo.record_usage(ins.insight_id, was_successful=False)
    loaded = repo.get_insight(ins.insight_id)
    assert loaded.usage_count == 2
    assert loaded.success_count == 1
    repo.close()


def test_record_usage_nonexistent_no_crash():
    repo = make_repo()
    repo.record_usage(uuid.uuid4(), was_successful=True)  # should not raise
    repo.close()


def test_prune_stale_removes_old_failures():
    import time
    from datetime import timedelta
    repo = make_repo()
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    ins = make_insight(content="Old failing insight")
    ins.creation_time = old_time
    ins.usage_count = 10
    ins.success_count = 1  # success_rate = 0.1, below threshold
    repo.add_insight(ins)

    n = repo.prune_stale(age_days=30, min_success_rate=0.5)
    assert n == 1
    assert repo.get_insight(ins.insight_id) is None
    repo.close()


def test_prune_stale_keeps_successful():
    from datetime import timedelta
    repo = make_repo()
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    ins = make_insight(content="Old successful insight")
    ins.creation_time = old_time
    ins.usage_count = 10
    ins.success_count = 9  # success_rate = 0.9
    repo.add_insight(ins)

    n = repo.prune_stale(age_days=30, min_success_rate=0.5)
    assert n == 0
    assert repo.get_insight(ins.insight_id) is not None
    repo.close()
