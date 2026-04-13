"""
tests/test_episode.py — Episode and EpisodeStore tests
"""

import uuid
from datetime import datetime, timezone

import pytest

from sca.episode import Episode, EpisodeStore


def make_store() -> EpisodeStore:
    return EpisodeStore(db_path=":memory:")


def make_episode(**kwargs) -> Episode:
    defaults = dict(
        domain="coding",
        initial_prompt="Test prompt",
        tags=["test"],
    )
    defaults.update(kwargs)
    return Episode(**defaults)


# ---------------------------------------------------------------------------
# Episode model tests
# ---------------------------------------------------------------------------

def test_episode_default_fields():
    ep = Episode()
    assert isinstance(ep.episode_id, uuid.UUID)
    assert isinstance(ep.start_time, datetime)
    assert ep.end_time is None
    assert ep.context_block_ids == []
    assert ep.prediction_ids == []
    assert ep.calibration_snapshot == {}
    assert ep.metadata == {}
    assert ep.tags == []


def test_episode_with_fields():
    pred_id = uuid.uuid4()
    ep = Episode(
        domain="research",
        initial_prompt="Hello",
        context_block_ids=[1, 2, 3],
        prediction_ids=[pred_id],
        calibration_snapshot={"coding": {"samples": 5}},
        tags=["pilot"],
    )
    assert ep.domain == "research"
    assert ep.initial_prompt == "Hello"
    assert len(ep.context_block_ids) == 3
    assert pred_id in ep.prediction_ids
    assert ep.calibration_snapshot["coding"]["samples"] == 5


def test_episode_optional_fields_none():
    ep = Episode()
    assert ep.domain is None
    assert ep.initial_prompt is None


# ---------------------------------------------------------------------------
# EpisodeStore CRUD
# ---------------------------------------------------------------------------

def test_save_and_load_episode():
    store = make_store()
    ep = make_episode()
    store.save_episode(ep)
    loaded = store.load_episode(ep.episode_id)
    assert loaded is not None
    assert loaded.episode_id == ep.episode_id
    assert loaded.domain == "coding"
    store.close()


def test_load_nonexistent_returns_none():
    store = make_store()
    result = store.load_episode(uuid.uuid4())
    assert result is None
    store.close()


def test_list_episodes_empty():
    store = make_store()
    assert store.list_episodes() == []
    store.close()


def test_list_episodes_returns_all():
    store = make_store()
    for i in range(5):
        store.save_episode(make_episode(initial_prompt=f"prompt {i}"))
    episodes = store.list_episodes()
    assert len(episodes) == 5
    store.close()


def test_list_episodes_pagination():
    store = make_store()
    for i in range(10):
        store.save_episode(make_episode(initial_prompt=f"p{i}"))
    page1 = store.list_episodes(limit=3, offset=0)
    page2 = store.list_episodes(limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 3
    ids1 = {e.episode_id for e in page1}
    ids2 = {e.episode_id for e in page2}
    assert ids1.isdisjoint(ids2)
    store.close()


def test_query_by_domain():
    store = make_store()
    store.save_episode(make_episode(domain="coding"))
    store.save_episode(make_episode(domain="coding"))
    store.save_episode(make_episode(domain="research"))
    coding = store.query_by_domain("coding")
    research = store.query_by_domain("research")
    assert len(coding) == 2
    assert len(research) == 1
    store.close()


def test_save_overwrites_existing():
    store = make_store()
    ep = make_episode()
    store.save_episode(ep)
    ep.domain = "updated"
    store.save_episode(ep)
    loaded = store.load_episode(ep.episode_id)
    assert loaded.domain == "updated"
    store.close()


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------

def test_anonymize_email():
    store = make_store()
    ep = Episode(initial_prompt="Contact user@example.com for info")
    anon = store.anonymize_episode(ep)
    assert "user@example.com" not in (anon.initial_prompt or "")
    assert "[EMAIL]" in (anon.initial_prompt or "")
    assert anon.episode_id == ep.episode_id
    store.close()


def test_anonymize_phone():
    store = make_store()
    ep = Episode(initial_prompt="Call 555-123-4567 now")
    anon = store.anonymize_episode(ep)
    assert "555-123-4567" not in (anon.initial_prompt or "")
    store.close()


def test_anonymize_ssn():
    store = make_store()
    ep = Episode(initial_prompt="SSN is 123-45-6789")
    anon = store.anonymize_episode(ep)
    assert "123-45-6789" not in (anon.initial_prompt or "")
    assert "[SSN]" in (anon.initial_prompt or "")
    store.close()


def test_anonymize_no_pii_unchanged():
    store = make_store()
    original = "Analyze Python function behavior"
    ep = Episode(initial_prompt=original)
    anon = store.anonymize_episode(ep)
    assert anon.initial_prompt == original
    store.close()
