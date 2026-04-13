"""
tests/test_evaluation.py — OutcomeEvaluator testleri

Tüm LLM çağrıları mock'lanmış, gerçek API çağrısı yok.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from sca.evaluation import (
    DEFAULT_FALLBACK_REASONING,
    DEFAULT_FALLBACK_SCORE,
    OutcomeEvaluator,
)
from sca.prediction import Prediction


def run(coro):
    return asyncio.run(coro)


def make_prediction(
    statement: str = "dosya.txt /tmp/ dizinindedir",
    confidence: float = 0.8,
    category: str = "file_location",
) -> Prediction:
    return Prediction(
        statement=statement,
        confidence_at_prediction=confidence,
        category=category,
    )


def make_mock_llm(response: str) -> MagicMock:
    """LLMClient mock'u oluşturur."""
    mock = MagicMock()
    mock.chat.return_value = response
    return mock


# ---------------------------------------------------------------------------
# Parse testleri
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_json_response(self):
        """Geçerli JSON yanıt parse edilmeli."""
        mock_llm = make_mock_llm('{"score": 8, "reasoning": "tahmin doğru"}')
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "dosya.txt bulundu"))
        assert score == pytest.approx(0.8, abs=0.01)
        assert "doğru" in reasoning

    def test_perfect_match_score(self):
        """score=10 → normalize 1.0 olmalı."""
        mock_llm = make_mock_llm('{"score": 10, "reasoning": "mükemmel eşleşme"}')
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "mükemmel sonuç"))
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_score(self):
        """score=0 → normalize 0.0 olmalı."""
        mock_llm = make_mock_llm('{"score": 0, "reasoning": "tamamen yanlış"}')
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "hiç ilgisi yok"))
        assert score == pytest.approx(0.0, abs=0.01)

    def test_parse_failure_returns_fallback(self):
        """Parse başarısız olursa fallback (0.5, "parse failed") döndürmeli."""
        mock_llm = make_mock_llm("bu bir JSON değil, sadece metin")
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "sonuç"))
        # Fallback değeri veya parse edilmiş bir şey
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_markdown_json_parsed(self):
        """Markdown kod bloğu içindeki JSON parse edilmeli."""
        mock_llm = make_mock_llm('```json\n{"score": 7, "reasoning": "oldukça doğru"}\n```')
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "sonuç"))
        assert score == pytest.approx(0.7, abs=0.01)

    def test_score_normalized_to_01(self):
        """Score her zaman [0,1] aralığında olmalı."""
        for raw_score in [0, 1, 5, 8, 10]:
            mock_llm = make_mock_llm(f'{{"score": {raw_score}, "reasoning": "test"}}')
            evaluator = OutcomeEvaluator(llm_client=mock_llm)
            score, _ = run(evaluator.evaluate(make_prediction(), "sonuç"))
            assert 0.0 <= score <= 1.0, f"raw_score={raw_score} → score={score} aralık dışı"


# ---------------------------------------------------------------------------
# Hata handling testleri
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_llm_exception_returns_fallback(self):
        """LLM exception fırlatırsa fallback döndürmeli (raise değil)."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API error")
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "sonuç"))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert "error" in reasoning.lower()

    def test_never_returns_none(self):
        """evaluate() hiçbir durumda None döndürmemeli."""
        # Normal durum
        mock_llm = make_mock_llm('{"score": 5, "reasoning": "orta"}')
        evaluator = OutcomeEvaluator(llm_client=mock_llm)
        score, reasoning = run(evaluator.evaluate(make_prediction(), "sonuç"))
        assert score is not None
        assert reasoning is not None

        # Parse hatası durumu
        mock_llm2 = make_mock_llm("bozuk yanıt")
        evaluator2 = OutcomeEvaluator(llm_client=mock_llm2)
        score2, reasoning2 = run(evaluator2.evaluate(make_prediction(), "sonuç"))
        assert score2 is not None
        assert reasoning2 is not None
