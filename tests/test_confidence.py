"""
Tests for sca.confidence — SelfConsistencyScorer, VerifierScorer,
ProvenancePenaltyCalculator, CompositeConfidenceScorer
"""
from __future__ import annotations
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from sca.context import ContextBlock, ContextManager, Provenance


def run(coro):
    return asyncio.run(coro)


def make_manager(*provenance_list):
    manager = ContextManager()
    ids = []
    for prov in provenance_list:
        bid = manager.add(ContextBlock(f"Content for {prov.value}", prov))
        ids.append(bid)
    return manager, ids


def make_mock_llm(response="7"):
    mock = MagicMock()
    mock.chat.return_value = response
    return mock


# ---------------------------------------------------------------------------
# SelfConsistencyScorer
# ---------------------------------------------------------------------------
class TestSelfConsistencyScorer(unittest.TestCase):
    def _make_scorer(self, llm_response="Sample response", n_samples=3):
        from sca.confidence import SelfConsistencyScorer
        llm = make_mock_llm(llm_response)
        scorer = SelfConsistencyScorer(llm_client=llm, n_samples=n_samples)
        return scorer, llm

    def test_score_returns_float_in_range(self):
        from sca.confidence import SelfConsistencyScorer
        from sca.similarity import SemanticSimilarity
        scorer, _ = self._make_scorer()
        with patch.object(SemanticSimilarity, "batch_cosine_similarity", return_value=[0.8, 0.9, 0.85]):
            result = run(scorer.score("What is 2+2?", "4"))
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_sample_temperatures_count(self):
        from sca.confidence import SelfConsistencyScorer
        scorer = SelfConsistencyScorer(make_mock_llm(), n_samples=5)
        temps = scorer._sample_temperatures()
        self.assertEqual(len(temps), 5)

    def test_sample_temperatures_in_range(self):
        from sca.confidence import SelfConsistencyScorer
        low, high = 0.3, 0.9
        scorer = SelfConsistencyScorer(make_mock_llm(), temperature_range=(low, high))
        temps = scorer._sample_temperatures()
        for t in temps:
            self.assertGreaterEqual(t, low)
            self.assertLessEqual(t, high + 1e-9)

    def test_single_sample_uses_midpoint(self):
        from sca.confidence import SelfConsistencyScorer
        scorer = SelfConsistencyScorer(make_mock_llm(), n_samples=1, temperature_range=(0.4, 0.8))
        temps = scorer._sample_temperatures()
        self.assertEqual(len(temps), 1)
        self.assertAlmostEqual(temps[0], 0.6, places=9)

    def test_all_failed_responses_returns_half(self):
        from sca.confidence import SelfConsistencyScorer
        scorer, llm = self._make_scorer()
        llm.chat.side_effect = RuntimeError("API error")
        result = run(scorer.score("prompt", "response"))
        self.assertEqual(result, 0.5)

    def test_high_similarity_yields_high_score(self):
        from sca.confidence import SelfConsistencyScorer
        from sca.similarity import SemanticSimilarity
        scorer, _ = self._make_scorer()
        with patch.object(SemanticSimilarity, "batch_cosine_similarity", return_value=[1.0, 1.0, 1.0]):
            result = run(scorer.score("prompt", "response"))
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_n_samples_3_default(self):
        from sca.confidence import DEFAULT_N_SAMPLES, SelfConsistencyScorer
        scorer = SelfConsistencyScorer(make_mock_llm())
        self.assertEqual(scorer.n_samples, DEFAULT_N_SAMPLES)


# ---------------------------------------------------------------------------
# VerifierScorer
# ---------------------------------------------------------------------------
class TestVerifierScorer(unittest.TestCase):
    def _make_scorer(self, llm_response="7"):
        from sca.confidence import VerifierScorer
        llm = make_mock_llm(llm_response)
        scorer = VerifierScorer(verifier_llm_client=llm)
        return scorer, llm

    def test_parse_integer_response(self):
        scorer, _ = self._make_scorer("7")
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertAlmostEqual(result, 0.7, places=6)

    def test_parse_decimal_response(self):
        scorer, _ = self._make_scorer("8.5")
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertAlmostEqual(result, 0.85, places=6)

    def test_parse_zero_response(self):
        scorer, _ = self._make_scorer("0")
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_parse_ten_response(self):
        scorer, _ = self._make_scorer("10")
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_unparseable_response_returns_fallback(self):
        from sca.confidence import DEFAULT_PARSE_FALLBACK
        scorer, _ = self._make_scorer("no number here")
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertEqual(result, DEFAULT_PARSE_FALLBACK)

    def test_api_failure_returns_fallback(self):
        from sca.confidence import DEFAULT_PARSE_FALLBACK, VerifierScorer
        llm = make_mock_llm()
        llm.chat.side_effect = RuntimeError("fail")
        scorer = VerifierScorer(verifier_llm_client=llm, max_retries=2)
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertEqual(result, DEFAULT_PARSE_FALLBACK)

    def test_retry_eventually_succeeds(self):
        from sca.confidence import VerifierScorer
        llm = make_mock_llm()
        llm.chat.side_effect = [RuntimeError("fail"), "8"]
        scorer = VerifierScorer(verifier_llm_client=llm, max_retries=2)
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertAlmostEqual(result, 0.8, places=6)

    def test_result_in_range(self):
        scorer, _ = self._make_scorer("6")
        result = run(scorer.score("Q?", "A.", "Context."))
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# ProvenancePenaltyCalculator
# ---------------------------------------------------------------------------
class TestProvenancePenaltyCalculator(unittest.TestCase):
    def _make_calc(self, *provs):
        from sca.confidence import ProvenancePenaltyCalculator
        manager, ids = make_manager(*provs)
        calc = ProvenancePenaltyCalculator(context_manager=manager)
        return calc, ids

    def test_external_tool_high_score(self):
        calc, ids = self._make_calc(Provenance.EXTERNAL_TOOL)
        score = calc.compute_penalty(ids)
        self.assertGreater(score, 0.7)

    def test_self_generated_low_score(self):
        calc, ids = self._make_calc(Provenance.SELF_GENERATED)
        score = calc.compute_penalty(ids)
        self.assertLess(score, 0.5)

    def test_empty_ids_returns_neutral(self):
        calc, _ = self._make_calc(Provenance.USER)
        score = calc.compute_penalty([])
        self.assertEqual(score, 0.5)

    def test_mixed_provenances_in_range(self):
        calc, ids = self._make_calc(Provenance.EXTERNAL_TOOL, Provenance.SELF_GENERATED)
        score = calc.compute_penalty(ids)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_knowledge_base_high_score(self):
        calc, ids = self._make_calc(Provenance.KNOWLEDGE_BASE)
        score = calc.compute_penalty(ids)
        self.assertGreater(score, 0.7)

    def test_system_provenance_neutral(self):
        calc, ids = self._make_calc(Provenance.SYSTEM)
        score = calc.compute_penalty(ids)
        self.assertAlmostEqual(score, 0.5, delta=0.05)

    def test_missing_block_id_skipped(self):
        from sca.confidence import ProvenancePenaltyCalculator
        manager = ContextManager()
        calc = ProvenancePenaltyCalculator(context_manager=manager)
        score = calc.compute_penalty([999])
        self.assertEqual(score, 0.5)

    def test_score_always_in_range_mixed(self):
        calc, ids = self._make_calc(
            Provenance.EXTERNAL_TOOL, Provenance.SELF_GENERATED,
            Provenance.DERIVED_INFERENCE, Provenance.USER,
        )
        score = calc.compute_penalty(ids)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_sigmoid_at_zero(self):
        from sca.confidence import ProvenancePenaltyCalculator
        result = ProvenancePenaltyCalculator._sigmoid(0)
        self.assertAlmostEqual(result, 0.5, places=9)

    def test_sigmoid_large_positive(self):
        from sca.confidence import ProvenancePenaltyCalculator
        result = ProvenancePenaltyCalculator._sigmoid(10)
        self.assertGreater(result, 0.99)

    def test_sigmoid_large_negative(self):
        from sca.confidence import ProvenancePenaltyCalculator
        result = ProvenancePenaltyCalculator._sigmoid(-10)
        self.assertLess(result, 0.01)

    def test_derived_inference_penalized(self):
        calc, ids = self._make_calc(Provenance.DERIVED_INFERENCE)
        score = calc.compute_penalty(ids)
        # Should be < 0.5 (penalized)
        self.assertLess(score, 0.5)

    def test_user_provenance_moderately_high(self):
        calc, ids = self._make_calc(Provenance.USER)
        score = calc.compute_penalty(ids)
        self.assertGreater(score, 0.5)


# ---------------------------------------------------------------------------
# CompositeConfidenceScorer
# ---------------------------------------------------------------------------
class TestCompositeConfidenceScorer(unittest.TestCase):
    def _make_composite(self, sc=0.8, verifier=0.7, prov=0.6):
        from sca.confidence import (
            CompositeConfidenceScorer, ProvenancePenaltyCalculator,
            SelfConsistencyScorer, VerifierScorer,
        )
        llm = make_mock_llm("response")
        sc_scorer = SelfConsistencyScorer(llm_client=llm, n_samples=2)
        v_scorer = VerifierScorer(verifier_llm_client=llm)
        manager, ids = make_manager(Provenance.EXTERNAL_TOOL)
        prov_calc = ProvenancePenaltyCalculator(context_manager=manager)

        composite = CompositeConfidenceScorer(
            self_consistency_scorer=sc_scorer,
            verifier_scorer=v_scorer,
            provenance_calculator=prov_calc,
        )

        async def mock_sc(*args, **kwargs): return sc
        async def mock_v(*args, **kwargs): return verifier
        sc_scorer.score = mock_sc
        v_scorer.score = mock_v
        prov_calc.compute_penalty = lambda ids: prov

        return composite, manager, ids

    def test_final_score_in_range(self):
        composite, manager, ids = self._make_composite()
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertGreaterEqual(result.final_score, 0.0)
        self.assertLessEqual(result.final_score, 1.0)

    def test_components_present(self):
        composite, manager, ids = self._make_composite()
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertIn("self_consistency", result.components)
        self.assertIn("verifier", result.components)
        self.assertIn("provenance", result.components)

    def test_weighted_average_correct(self):
        sc, v, prov = 0.8, 0.7, 0.6
        composite, manager, ids = self._make_composite(sc, v, prov)
        expected = 0.4 * sc + 0.4 * v + 0.2 * prov
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertAlmostEqual(result.final_score, expected, places=6)

    def test_reasoning_string_non_empty(self):
        composite, manager, ids = self._make_composite()
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertIsInstance(result.reasoning, str)
        self.assertGreater(len(result.reasoning), 10)

    def test_high_confidence_label(self):
        composite, manager, ids = self._make_composite(sc=1.0, verifier=1.0, prov=1.0)
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertIn("HIGH", result.reasoning)

    def test_low_confidence_label(self):
        composite, manager, ids = self._make_composite(sc=0.0, verifier=0.0, prov=0.0)
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertIn("LOW", result.reasoning)

    def test_medium_confidence_label(self):
        composite, manager, ids = self._make_composite(sc=0.5, verifier=0.55, prov=0.5)
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertIn("MEDIUM", result.reasoning)

    def test_custom_weights_missing_key_raises(self):
        from sca.confidence import (
            CompositeConfidenceScorer, ProvenancePenaltyCalculator,
            SelfConsistencyScorer, VerifierScorer,
        )
        llm = make_mock_llm()
        sc = SelfConsistencyScorer(llm, n_samples=2)
        v = VerifierScorer(llm)
        manager, _ = make_manager(Provenance.USER)
        prov = ProvenancePenaltyCalculator(manager)
        with self.assertRaises(ValueError):
            CompositeConfidenceScorer(sc, v, prov, weights={"self_consistency": 0.5})

    def test_weights_normalized_if_not_sum_to_one(self):
        from sca.confidence import (
            CompositeConfidenceScorer, ProvenancePenaltyCalculator,
            SelfConsistencyScorer, VerifierScorer,
        )
        llm = make_mock_llm()
        sc = SelfConsistencyScorer(llm, n_samples=2)
        v = VerifierScorer(llm)
        manager, _ = make_manager(Provenance.USER)
        prov = ProvenancePenaltyCalculator(manager)
        composite = CompositeConfidenceScorer(
            sc, v, prov,
            weights={"self_consistency": 2.0, "verifier": 2.0, "provenance": 1.0},
        )
        total = sum(composite.weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_result_is_namedtuple(self):
        from sca.confidence import ConfidenceScore
        composite, manager, ids = self._make_composite()
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertIsInstance(result, ConfidenceScore)

    def test_components_values_match_inputs(self):
        sc, v, prov = 0.6, 0.75, 0.4
        composite, manager, ids = self._make_composite(sc, v, prov)
        result = run(composite.score("prompt", "response", manager, ids))
        self.assertAlmostEqual(result.components["self_consistency"], sc, places=6)
        self.assertAlmostEqual(result.components["verifier"], v, places=6)
        self.assertAlmostEqual(result.components["provenance"], prov, places=6)


if __name__ == "__main__":
    unittest.main()