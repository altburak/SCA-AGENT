"""
CSM — Confidence Scoring Module
================================
Her LLM çıktısına 0-1 arası bir eminlik skoru atar.

Üç sinyal:
1. Self-consistency: Aynı soru N kez sorulur, tutarlılık ölçülür.
2. Verifier check: İkinci model faktüel güvenilirliği değerlendirir.
3. Provenance penalty: Context bloklarının kaynağına göre skor ayarlanır.

AOGL entegrasyonu: CompositeConfidenceScorer.score() artık opsiyonel
`category` parametresi alır. Eğer o kategoriye ait bir ConfidenceCalibrator
varsa, final_score kalibre edilir.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import lru_cache
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_N_SAMPLES: int = 3
DEFAULT_TEMPERATURE_RANGE: tuple[float, float] = (0.5, 1.0)
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_PARSE_FALLBACK: float = 0.5
VERIFIER_SCALE_MAX: float = 10.0
MAX_CALLS_PER_SECOND: int = 5

# Provenance ağırlıkları
PROVENANCE_WEIGHTS: dict[str, float] = {
    "EXTERNAL_TOOL": 1.0,
    "KNOWLEDGE_BASE": 1.0,
    "USER": 0.8,
    "SYSTEM": 0.0,
    "DERIVED_INFERENCE": -0.3,
    "SELF_GENERATED": -0.5,
}

DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "self_consistency": 0.4,
    "verifier": 0.4,
    "provenance": 0.2,
}

VERIFIER_PROMPT_TEMPLATE: str = (
    "Given this question: {question}\n\n"
    "This answer: {answer}\n\n"
    "And this context: {context}\n\n"
    "Rate the factual reliability of the answer on a scale of 0-10. "
    "Consider: internal consistency, evidence in context, logical coherence. "
    "Return ONLY a number 0-10."
)


# ---------------------------------------------------------------------------
# ConfidenceScore (namedtuple)
# ---------------------------------------------------------------------------
class ConfidenceScore(NamedTuple):
    """Composite confidence scoring sonucu.

    Attributes:
        final_score: Ağırlıklı birleşim skoru [0, 1].
        components: Alt skor bileşenleri.
        reasoning: İnsan-okunabilir açıklama.
    """

    final_score: float
    components: dict[str, float]
    reasoning: str


# ---------------------------------------------------------------------------
# Rate limiter (basit token bucket)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Basit token-bucket rate limiter."""

    def __init__(self, max_per_second: int = MAX_CALLS_PER_SECOND) -> None:
        self._max = max_per_second
        self._tokens = float(max_per_second)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._max, self._tokens + elapsed * self._max)
            self._last_refill = now
            if self._tokens < 1:
                sleep_time = (1 - self._tokens) / self._max
                await asyncio.sleep(sleep_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


_global_rate_limiter = _RateLimiter(MAX_CALLS_PER_SECOND)


# ---------------------------------------------------------------------------
# SelfConsistencyScorer
# ---------------------------------------------------------------------------
class SelfConsistencyScorer:
    """Aynı prompt'u birden çok kez sorarak tutarlılık ölçer.

    Args:
        llm_client: LLMClient nesnesi.
        n_samples: Kaç kez sorulacağı (default: 3).
        temperature_range: Sıcaklık aralığı (default: (0.5, 1.0)).
        timeout: Her çağrı için maksimum süre (saniye).

    Example:
        >>> scorer = SelfConsistencyScorer(llm_client=client, n_samples=3)
        >>> score = asyncio.run(scorer.score("What is 2+2?", "4"))
    """

    def __init__(
        self,
        llm_client: Any,
        n_samples: int = DEFAULT_N_SAMPLES,
        temperature_range: tuple[float, float] = DEFAULT_TEMPERATURE_RANGE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.llm_client = llm_client
        self.n_samples = n_samples
        self.temperature_range = temperature_range
        self.timeout = timeout
        logger.debug(
            "SelfConsistencyScorer başlatıldı: n_samples=%d, temp_range=%s",
            n_samples,
            temperature_range,
        )

    def _sample_temperatures(self) -> list[float]:
        """n_samples adet eşit aralıklı sıcaklık değeri üretir."""
        low, high = self.temperature_range
        if self.n_samples == 1:
            return [(low + high) / 2]
        step = (high - low) / (self.n_samples - 1)
        return [low + i * step for i in range(self.n_samples)]

    async def _query_once(self, prompt: str, temperature: float) -> Optional[str]:
        """Tek bir LLM çağrısı yapar (async)."""
        await _global_rate_limiter.acquire()
        try:
            loop = asyncio.get_event_loop()
            messages = [{"role": "user", "content": prompt}]
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.llm_client.chat(messages, temperature=temperature),
                ),
                timeout=self.timeout,
            )
            logger.debug("Self-consistency örneği alındı (temp=%.2f).", temperature)
            return response
        except asyncio.TimeoutError:
            logger.warning("Self-consistency çağrısı zaman aşımına uğradı (temp=%.2f).", temperature)
            return None
        except Exception as exc:
            logger.warning("Self-consistency çağrısı başarısız: %s", exc)
            return None

    async def score(self, prompt: str, base_response: str) -> float:
        """Prompt'u n_samples kez sorerek tutarlılık skoru hesaplar.

        Args:
            prompt: Sorulacak prompt.
            base_response: Ana model yanıtı.

        Returns:
            [0, 1] aralığında tutarlılık skoru.
        """
        from sca.similarity import SemanticSimilarity

        temperatures = self._sample_temperatures()
        tasks = [self._query_once(prompt, temp) for temp in temperatures]
        responses = await asyncio.gather(*tasks)

        valid_responses = [r for r in responses if r is not None]
        if not valid_responses:
            logger.warning("Hiç geçerli self-consistency yanıtı alınamadı, 0.5 döndürülüyor.")
            return 0.5

        similarity = SemanticSimilarity()
        scores = similarity.batch_cosine_similarity(base_response, valid_responses)
        avg_score = sum(scores) / len(scores)
        logger.debug(
            "Self-consistency skoru: %.4f (%d/%d geçerli yanıt)",
            avg_score,
            len(valid_responses),
            self.n_samples,
        )
        return avg_score


# ---------------------------------------------------------------------------
# VerifierScorer
# ---------------------------------------------------------------------------
class VerifierScorer:
    """İkinci bir model kullanarak faktüel güvenilirlik skoru hesaplar.

    Args:
        verifier_llm_client: Doğrulayıcı LLMClient (küçük model tercih edilir).
        prompt_template: Verifier prompt şablonu ({question}, {answer}, {context}).
        max_retries: Maksimum deneme sayısı.

    Example:
        >>> scorer = VerifierScorer(verifier_llm_client=client)
        >>> score = asyncio.run(scorer.score("soru", "cevap", "bağlam"))
    """

    def __init__(
        self,
        verifier_llm_client: Any,
        prompt_template: str = VERIFIER_PROMPT_TEMPLATE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.verifier_llm_client = verifier_llm_client
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        logger.debug("VerifierScorer başlatıldı: max_retries=%d", max_retries)

    def _parse_score(self, raw: str) -> float:
        """Ham LLM yanıtından 0-1 arası skor çıkarır."""
        match = re.search(r"\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\b", raw)
        if match:
            try:
                score = float(match.group()) / VERIFIER_SCALE_MAX
                return max(0.0, min(1.0, score))
            except (ValueError, ZeroDivisionError):
                pass
        logger.warning("Verifier skoru parse edilemedi: %r", raw[:100])
        return DEFAULT_PARSE_FALLBACK

    async def score(self, question: str, answer: str, context: str) -> float:
        """Faktüel güvenilirlik skoru hesaplar.

        Args:
            question: Ana soru.
            answer: Değerlendirilecek cevap.
            context: İlgili bağlam metni.

        Returns:
            [0, 1] aralığında güvenilirlik skoru.
        """
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            context=context[:2000],  # Çok uzun context'i kırp
        )
        messages = [{"role": "user", "content": prompt}]

        loop = asyncio.get_event_loop()
        for attempt in range(1, self.max_retries + 1):
            try:
                await _global_rate_limiter.acquire()
                raw = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.verifier_llm_client.chat(messages),
                    ),
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )
                return self._parse_score(raw)
            except asyncio.TimeoutError:
                logger.warning("Verifier çağrısı zaman aşımı (deneme %d/%d).", attempt, self.max_retries)
            except Exception as exc:
                logger.warning("Verifier çağrısı başarısız (deneme %d/%d): %s", attempt, self.max_retries, exc)

        logger.warning("Tüm verifier denemeleri başarısız, %.1f döndürülüyor.", DEFAULT_PARSE_FALLBACK)
        return DEFAULT_PARSE_FALLBACK


# ---------------------------------------------------------------------------
# ProvenancePenaltyCalculator
# ---------------------------------------------------------------------------
class ProvenancePenaltyCalculator:
    """Cevabın dayandığı context bloklarına göre provenance skoru hesaplar.

    Args:
        context_manager: ContextManager referansı.

    Example:
        >>> calc = ProvenancePenaltyCalculator(context_manager=manager)
        >>> score = calc.compute_penalty([0, 1, 2])
    """

    def __init__(self, context_manager: Any) -> None:
        self.context_manager = context_manager
        logger.debug("ProvenancePenaltyCalculator başlatıldı.")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid fonksiyonu: 1 / (1 + e^(-x))."""
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def compute_penalty(self, derived_from_ids: list[int]) -> float:
        """Belirtilen blok ID'leri için provenance skoru hesaplar.

        Args:
            derived_from_ids: Cevabın türetildiği blok ID'leri.

        Returns:
            [0, 1] aralığında provenance güven skoru.
        """
        if not derived_from_ids:
            logger.debug("derived_from_ids boş, nötr skor 0.5 döndürülüyor.")
            return 0.5

        weights: list[float] = []
        for block_id in derived_from_ids:
            try:
                block = self.context_manager.get_block(block_id)
                provenance_name = block.provenance.value
                weight = PROVENANCE_WEIGHTS.get(provenance_name, 0.0)
                weighted = weight * block.confidence
                weights.append(weighted)
                logger.debug(
                    "Blok %d: provenance=%s weight=%.2f conf=%.2f → %.2f",
                    block_id, provenance_name, weight, block.confidence, weighted,
                )
            except KeyError:
                logger.warning("Blok ID %d bulunamadı, atlanıyor.", block_id)

        if not weights:
            return 0.5

        avg_weight = sum(weights) / len(weights)
        score = self._sigmoid(avg_weight * 3)
        logger.debug("Provenance skoru: avg_weight=%.4f → sigmoid → %.4f", avg_weight, score)
        return score


# ---------------------------------------------------------------------------
# CompositeConfidenceScorer
# ---------------------------------------------------------------------------
class CompositeConfidenceScorer:
    """CSM'in ana arayüzü. Üç skorer'ı birleştirir.

    Args:
        self_consistency_scorer: SelfConsistencyScorer nesnesi.
        verifier_scorer: VerifierScorer nesnesi.
        provenance_calculator: ProvenancePenaltyCalculator nesnesi.
        weights: Alt skorların ağırlıkları (toplamı 1.0 olmalı).

    AOGL entegrasyonu:
        CalibrationLearner.apply_to_csm() çağrıldıktan sonra
        _category_calibrators sözlüğü doldurulur. score() metodunda
        category parametresi verildiğinde bu calibrator uygulanır.

    Example:
        >>> scorer = CompositeConfidenceScorer(
        ...     self_consistency_scorer=sc_scorer,
        ...     verifier_scorer=v_scorer,
        ...     provenance_calculator=prov_calc,
        ... )
        >>> result = asyncio.run(scorer.score(prompt, response, manager, [0, 1]))
        >>> print(result.final_score)
    """

    def __init__(
        self,
        self_consistency_scorer: SelfConsistencyScorer,
        verifier_scorer: VerifierScorer,
        provenance_calculator: ProvenancePenaltyCalculator,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.self_consistency_scorer = self_consistency_scorer
        self.verifier_scorer = verifier_scorer
        self.provenance_calculator = provenance_calculator
        self.weights = weights or dict(DEFAULT_COMPOSITE_WEIGHTS)
        # AOGL: category-aware calibrators (CalibrationLearner tarafından doldurulur)
        self._category_calibrators: dict[str, Any] = {}
        self._validate_weights()
        logger.debug("CompositeConfidenceScorer başlatıldı: weights=%s", self.weights)

    def _validate_weights(self) -> None:
        required_keys = {"self_consistency", "verifier", "provenance"}
        if not required_keys.issubset(self.weights.keys()):
            raise ValueError(f"weights şu anahtarları içermeli: {required_keys}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning("Ağırlıkların toplamı 1.0 değil: %.6f. Normalize ediliyor.", total)
            for k in self.weights:
                self.weights[k] /= total

    def _build_reasoning(
        self,
        sc_score: float,
        v_score: float,
        prov_score: float,
        final_score: float,
        calibrated: bool = False,
    ) -> str:
        """İnsan-okunabilir açıklama üretir."""
        confidence_label = (
            "HIGH" if final_score >= 0.75
            else "MEDIUM" if final_score >= 0.50
            else "LOW"
        )
        calib_note = " [calibrated]" if calibrated else ""
        return (
            f"Confidence: {confidence_label} ({final_score:.3f}){calib_note}. "
            f"Self-consistency: {sc_score:.3f} (weight {self.weights['self_consistency']:.2f}), "
            f"Verifier: {v_score:.3f} (weight {self.weights['verifier']:.2f}), "
            f"Provenance: {prov_score:.3f} (weight {self.weights['provenance']:.2f})."
        )

    async def score(
        self,
        prompt: str,
        response: str,
        context_manager: Any,
        derived_from_ids: list[int],
        category: Optional[str] = None,
    ) -> ConfidenceScore:
        """Üç skorer'ı paralel çalıştırır ve birleşik skor döndürür.

        Args:
            prompt: Kullanıcının sorusu.
            response: LLM'in ana yanıtı.
            context_manager: ContextManager nesnesi.
            derived_from_ids: Yanıtın türetildiği blok ID'leri.
            category: Opsiyonel tahmin kategorisi. Verilirse ve o kategori
                için bir ConfidenceCalibrator mevcutsa, final_score kalibre
                edilir. Backward-compatible (default: None).

        Returns:
            ConfidenceScore namedtuple.
        """
        context_text = context_manager.to_prompt() if context_manager else ""

        prov_score = self.provenance_calculator.compute_penalty(derived_from_ids)

        sc_task = self.self_consistency_scorer.score(prompt, response)
        v_task = self.verifier_scorer.score(prompt, response, context_text)

        sc_score, v_score = await asyncio.gather(sc_task, v_task)

        raw_final = (
            self.weights["self_consistency"] * sc_score
            + self.weights["verifier"] * v_score
            + self.weights["provenance"] * prov_score
        )
        raw_final = float(max(0.0, min(1.0, raw_final)))

        # Category-aware calibration (AOGL entegrasyonu)
        calibrated = False
        final_score = raw_final
        if category is not None:
            cal = self._category_calibrators.get(category)
            if cal is not None and cal.is_fitted:
                final_score = cal.apply(raw_final)
                calibrated = True
                logger.debug(
                    "Category '%s' calibration uygulandı: %.4f → %.4f",
                    category, raw_final, final_score,
                )

        reasoning = self._build_reasoning(
            sc_score, v_score, prov_score, final_score, calibrated=calibrated
        )

        logger.info(
            "CompositeConfidenceScorer: final=%.4f sc=%.4f v=%.4f prov=%.4f cat=%s",
            final_score, sc_score, v_score, prov_score, category,
        )

        return ConfidenceScore(
            final_score=final_score,
            components={
                "self_consistency": sc_score,
                "verifier": v_score,
                "provenance": prov_score,
            },
            reasoning=reasoning,
        )