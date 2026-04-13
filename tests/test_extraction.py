"""
tests/test_extraction.py — InsightExtractor tests with mocked LLM and grounding_log
"""

import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from sca.episode import Episode
from sca.extraction import InsightExtractor
from sca.insight import InsightType
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction


def make_prediction(category: str = "general", confidence: float = 0.7) -> Prediction:
    return Prediction(
        statement="The file exists at /tmp/test.py",
        confidence_at_prediction=confidence,
        category=category,
    )


def make_outcome(pred_id: uuid.UUID, match_score: float = 0.8) -> Outcome:
    return Outcome(
        prediction_id=pred_id,
        action_executed=ActionProposal(action_type=ActionType.NO_ACTION),
        actual_result="File found.",
        match_score=match_score,
    )


def make_episode_with_predictions(n: int = 3) -> tuple[Episode, list[Prediction], list[Outcome]]:
    ep = Episode(domain="coding")
    preds = []
    outcomes = []
    for i in range(n):
        p = make_prediction(category=f"category_{i}", confidence=0.5 + i * 0.1)
        o = make_outcome(p.prediction_id, match_score=min(1.0, 0.3 + i * 0.15))
        preds.append(p)
        outcomes.append(o)
        ep.prediction_ids.append(p.prediction_id)
    return ep, preds, outcomes


def make_grounding_log(preds: list[Prediction], outcomes: list[Outcome]):
    log = MagicMock()
    pred_map = {p.prediction_id: p for p in preds}
    outcome_map = {o.prediction_id: o for o in outcomes}

    def get_pred(pid):
        return pred_map.get(pid)

    def get_outcome(pid):
        return outcome_map.get(pid)

    log.get_prediction.side_effect = get_pred
    log.get_outcome_for_prediction.side_effect = get_outcome
    return log


def make_llm(response_json: dict | list | None = None, raw: str = "") -> MagicMock:
    llm = MagicMock()
    if response_json is not None:
        llm.chat.return_value = json.dumps(response_json)
    else:
        llm.chat.return_value = raw
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_bias_patterns_success():
    ep, preds, outcomes = make_episode_with_predictions(4)
    grounding = make_grounding_log(preds, outcomes)
    response = {"biases": [
        {"pattern": "Overconfident on file locations", "evidence": [], "severity": 0.7, "category": "general"}
    ]}
    llm = make_llm(response)
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_bias_patterns(ep))
    assert len(results) == 1
    assert results[0].type == InsightType.BIAS_PATTERN
    assert "Overconfident" in results[0].content


def test_extract_bias_patterns_empty_response():
    ep, preds, outcomes = make_episode_with_predictions(3)
    grounding = make_grounding_log(preds, outcomes)
    llm = make_llm({"biases": []})
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_bias_patterns(ep))
    assert results == []


def test_extract_bias_patterns_parse_failure_returns_empty():
    ep, preds, outcomes = make_episode_with_predictions(3)
    grounding = make_grounding_log(preds, outcomes)
    llm = make_llm(raw="Not valid JSON at all $$$$")
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_bias_patterns(ep))
    assert results == []


def test_extract_successful_strategies_no_high_matches():
    ep, preds, outcomes = make_episode_with_predictions(3)
    # All match scores < HIGH_MATCH_THRESHOLD
    for o in outcomes:
        o.match_score = 0.3
    grounding = make_grounding_log(preds, outcomes)
    llm = make_llm({"strategies": []})
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_successful_strategies(ep))
    assert results == []
    llm.chat.assert_not_called()  # LLM not called if no high-match pairs


def test_extract_successful_strategies_with_high_matches():
    ep, preds, outcomes = make_episode_with_predictions(3)
    outcomes[0].match_score = 0.9
    outcomes[1].match_score = 0.85
    grounding = make_grounding_log(preds, outcomes)
    response = {"strategies": [
        {"strategy": "Use type hints for better accuracy", "evidence": [], "confidence": 0.8, "keywords": ["typing"]}
    ]}
    llm = make_llm(response)
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_successful_strategies(ep))
    assert len(results) == 1
    assert results[0].type == InsightType.SUCCESSFUL_STRATEGY


def test_extract_failure_modes_no_low_matches():
    ep, preds, outcomes = make_episode_with_predictions(2)
    for o in outcomes:
        o.match_score = 0.9
    grounding = make_grounding_log(preds, outcomes)
    llm = make_llm({"failures": []})
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_failure_modes(ep))
    assert results == []
    llm.chat.assert_not_called()


def test_extract_failure_modes_with_low_matches():
    ep, preds, outcomes = make_episode_with_predictions(3)
    outcomes[0].match_score = 0.1
    outcomes[1].match_score = 0.2
    grounding = make_grounding_log(preds, outcomes)
    response = {"failures": [
        {"mode": "Assumes file paths are absolute", "evidence": [], "confidence": 0.65, "category": "file_location"}
    ]}
    llm = make_llm(response)
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_failure_modes(ep))
    assert len(results) == 1
    assert results[0].type == InsightType.FAILURE_MODE


def test_extract_domain_knowledge():
    ep, preds, outcomes = make_episode_with_predictions(3)
    # High confidence AND high match
    preds[0].confidence_at_prediction = 0.9
    outcomes[0].match_score = 0.95
    grounding = make_grounding_log(preds, outcomes)
    response = {"facts": [
        {"fact": "Python f-strings require Python 3.6+", "evidence": [], "confidence": 0.8, "domain": "coding"}
    ]}
    llm = make_llm(response)
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_domain_knowledge(ep))
    assert len(results) == 1
    assert results[0].type == InsightType.DOMAIN_KNOWLEDGE


def test_extract_all_runs_in_parallel():
    ep, preds, outcomes = make_episode_with_predictions(5)
    outcomes[0].match_score = 0.9
    outcomes[1].match_score = 0.9
    preds[0].confidence_at_prediction = 0.9
    outcomes[0].match_score = 0.95
    grounding = make_grounding_log(preds, outcomes)

    # Return different responses each call — use side_effect
    responses = [
        json.dumps({"biases": [{"pattern": "Bias A", "evidence": [], "severity": 0.6, "category": "general"}]}),
        json.dumps({"strategies": [{"strategy": "Strategy B", "evidence": [], "confidence": 0.7, "keywords": []}]}),
        json.dumps({"failures": []}),
        json.dumps({"facts": []}),
    ]
    llm = MagicMock()
    llm.chat.side_effect = responses

    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_all(ep))
    assert isinstance(results, list)
    types = {r.type for r in results}
    assert InsightType.BIAS_PATTERN in types
    assert InsightType.SUCCESSFUL_STRATEGY in types


def test_extract_all_no_predictions():
    ep = Episode(domain="coding")
    grounding = MagicMock()
    llm = make_llm({"biases": []})
    extractor = InsightExtractor(llm_client=llm, grounding_log=grounding)
    results = asyncio.run(extractor.extract_all(ep))
    assert results == []
