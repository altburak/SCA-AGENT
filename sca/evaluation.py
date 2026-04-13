"""
sca/evaluation.py — Outcome Evaluation

OutcomeEvaluator: Prediction ile gerçek sonucu LLM rubric'iyle karşılaştırır
ve 0-1 arası bir match_score üretir.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sca.prediction import Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SCORE_MAX: float = 10.0
DEFAULT_FALLBACK_SCORE: float = 0.5
DEFAULT_FALLBACK_REASONING: str = "parse failed"

EVALUATOR_PROMPT_TEMPLATE: str = """\
You are evaluating how well an AI agent's prediction matches reality.

PREDICTION (what the agent expected):
{statement}

ACTUAL RESULT (what actually happened):
{actual_result}

RUBRIC (0-10 scale):
  0-2 = Prediction completely wrong or irrelevant
  3-4 = Prediction partially aligned but significant errors
  5-6 = Prediction roughly correct with notable gaps
  7-8 = Prediction mostly accurate, minor discrepancies
  9-10 = Prediction fully aligned with actual result

Consider: specificity, accuracy, relevance.

Respond ONLY in JSON format, no other text:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""


# ---------------------------------------------------------------------------
# OutcomeEvaluator
# ---------------------------------------------------------------------------
class OutcomeEvaluator:
    """LLM rubric tabanlı tahmin-sonuç değerlendiricisi.

    Bir Prediction ve gerçek sonuç string'i alarak 0-1 arası bir
    match_score ve gerekçe üretir.

    Args:
        llm_client: LLMClient nesnesi (büyük model önerilir — Llama 3.3 70B).

    Example:
        >>> evaluator = OutcomeEvaluator(llm_client=client)
        >>> score, reason = asyncio.run(
        ...     evaluator.evaluate(prediction, "Dosya içeriği: hello world")
        ... )
        >>> print(score)  # 0.9
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client
        logger.debug("OutcomeEvaluator başlatıldı.")

    def _build_prompt(self, prediction: Prediction, actual_result: str) -> str:
        """Değerlendirme prompt'unu oluşturur.

        Args:
            prediction: Değerlendirilecek tahmin.
            actual_result: Gerçek sonuç metni.

        Returns:
            Formatlanmış prompt string.
        """
        # Çok uzun actual_result'ı kırp
        truncated = actual_result[:3000] if len(actual_result) > 3000 else actual_result
        return EVALUATOR_PROMPT_TEMPLATE.format(
            statement=prediction.statement,
            actual_result=truncated,
        )

    def _parse_response(self, raw: str) -> tuple[float, str]:
        """LLM yanıtından score ve reasoning çıkarır.

        Parse başarısız olursa default fallback döndürür (asla None değil).

        Args:
            raw: LLM'in ham yanıtı.

        Returns:
            (normalized_score [0,1], reasoning) tuple'ı.
        """
        # JSON bloğunu bulmaya çalış
        raw_stripped = raw.strip()

        # Markdown kod bloğunu temizle
        raw_stripped = re.sub(r"```(?:json)?\s*", "", raw_stripped)
        raw_stripped = raw_stripped.strip().rstrip("`").strip()

        # Önce direkt JSON parse dene
        parsed = self._try_parse_json(raw_stripped)
        if parsed is not None:
            return self._extract_from_parsed(parsed)

        # JSON nesnesini regex ile bul
        json_match = re.search(r"\{[^{}]+\}", raw_stripped, re.DOTALL)
        if json_match:
            parsed = self._try_parse_json(json_match.group())
            if parsed is not None:
                return self._extract_from_parsed(parsed)

        # Sadece sayı varsa
        num_match = re.search(r"\b([0-9]|10)(?:\.[0-9]+)?\b", raw_stripped)
        if num_match:
            try:
                score = float(num_match.group()) / SCORE_MAX
                score = max(0.0, min(1.0, score))
                return (score, f"Score extracted from raw: {raw_stripped[:100]}")
            except (ValueError, ZeroDivisionError):
                pass

        logger.warning("LLM evaluation yanıtı parse edilemedi: %r", raw[:200])
        return (DEFAULT_FALLBACK_SCORE, DEFAULT_FALLBACK_REASONING)

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        """JSON parse dener, başarısız olursa None döner."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _extract_from_parsed(data: dict) -> tuple[float, str]:
        """Parse edilmiş dict'ten score ve reasoning çıkarır."""
        try:
            raw_score = float(data.get("score", DEFAULT_FALLBACK_SCORE * SCORE_MAX))
            reasoning = str(data.get("reasoning", DEFAULT_FALLBACK_REASONING))
            normalized = max(0.0, min(1.0, raw_score / SCORE_MAX))
            return (normalized, reasoning)
        except (TypeError, ValueError, ZeroDivisionError):
            return (DEFAULT_FALLBACK_SCORE, DEFAULT_FALLBACK_REASONING)

    async def evaluate(
        self, prediction: Prediction, actual_result: str
    ) -> tuple[float, str]:
        """Tahmin ile gerçek sonucu karşılaştırır.

        LLM'e rubric tabanlı prompt ile değerlendirme yaptırır.
        Her durumda concrete değer döndürür (None asla).

        Args:
            prediction: Değerlendirilecek tahmin.
            actual_result: Gerçek sonuç metni.

        Returns:
            (match_score [0,1], reasoning_str) tuple'ı.
            Parse başarısız olursa (0.5, "parse failed") döner.
        """
        import asyncio

        prompt = self._build_prompt(prediction, actual_result)
        messages = [{"role": "user", "content": prompt}]

        loop = asyncio.get_event_loop()
        try:
            raw_response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.llm_client.chat(messages),
                ),
                timeout=30.0,
            )
            score, reasoning = self._parse_response(raw_response)
            logger.info(
                "Evaluation tamamlandı: score=%.3f | %s",
                score, reasoning[:80],
            )
            return (score, reasoning)

        except asyncio.TimeoutError:
            logger.warning("Evaluation LLM çağrısı zaman aşımına uğradı.")
            return (DEFAULT_FALLBACK_SCORE, "timeout: evaluation LLM did not respond")

        except Exception as exc:
            logger.warning("Evaluation LLM çağrısı başarısız: %s", exc)
            return (DEFAULT_FALLBACK_SCORE, f"error: {exc}")
