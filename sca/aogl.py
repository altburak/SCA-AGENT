"""
sca/aogl.py — AOGLController (Action-Outcome Grounding Loop)

Ana orkestrasyon bileşeni. Tahmin üretimi, eylem önerisi,
yürütme ve kalibrasyon döngüsünü yönetir.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from sca.actions import ActionExecutor
from sca.calibration import CalibrationLearner
from sca.evaluation import OutcomeEvaluator
from sca.grounding import GroundingLog
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
PROPOSE_ACTION_TIMEOUT: float = 30.0
NO_ACTION_FALLBACK_REASON: str = "LLM eylem önerisi parse edilemedi, NO_ACTION kullanılıyor."


# ---------------------------------------------------------------------------
# AOGLController
# ---------------------------------------------------------------------------
class AOGLController:
    """Action-Outcome Grounding Loop orkestratörü.

    Tahmin üretir, doğrulama eylemi önerir, eylemi yürütür,
    sonucu değerlendirir ve kalibrasyon döngüsünü yönetir.

    Args:
        psm_manager: PSM ContextManager nesnesi.
        csm_scorer: CompositeConfidenceScorer nesnesi.
        action_executor: ActionExecutor nesnesi.
        outcome_evaluator: OutcomeEvaluator nesnesi.
        grounding_log: GroundingLog nesnesi.
        calibration_learner: CalibrationLearner nesnesi.
        action_planner_llm: Eylem önerisi için LLMClient.

    Example:
        >>> ctrl = AOGLController(
        ...     psm_manager=manager,
        ...     csm_scorer=csm,
        ...     action_executor=executor,
        ...     outcome_evaluator=evaluator,
        ...     grounding_log=log,
        ...     calibration_learner=learner,
        ...     action_planner_llm=llm,
        ... )
        >>> pred, outcome = asyncio.run(
        ...     ctrl.run_full_cycle("dosya.txt var", "file_location", 0.8, [0])
        ... )
    """

    def __init__(
        self,
        psm_manager: Any,
        csm_scorer: Any,
        action_executor: ActionExecutor,
        outcome_evaluator: OutcomeEvaluator,
        grounding_log: GroundingLog,
        calibration_learner: CalibrationLearner,
        action_planner_llm: Any,
    ) -> None:
        self.psm_manager = psm_manager
        self.csm_scorer = csm_scorer
        self.action_executor = action_executor
        self.outcome_evaluator = outcome_evaluator
        self.grounding_log = grounding_log
        self.calibration_learner = calibration_learner
        self.action_planner_llm = action_planner_llm
        logger.info("AOGLController başlatıldı.")

    # ------------------------------------------------------------------
    # make_prediction
    # ------------------------------------------------------------------

    def make_prediction(
        self,
        statement: str,
        category: str,
        confidence: float,
        context_block_ids: list[int],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Prediction:
        """Yeni bir Prediction oluşturur ve GroundingLog'a kaydeder.

        Args:
            statement: Tahminin metinsel ifadesi.
            category: Tahmin kategorisi (esnek string).
            confidence: CSM tarafından hesaplanan confidence skoru [0,1].
            context_block_ids: PSM blok ID'leri.
            metadata: İsteğe bağlı ek bilgi.

        Returns:
            Oluşturulmuş ve kaydedilmiş Prediction nesnesi.
        """
        pred = Prediction(
            statement=statement,
            confidence_at_prediction=confidence,
            category=category,
            context_block_ids=context_block_ids,
            metadata=metadata or {},
        )
        self.grounding_log.add_prediction(pred)
        logger.info(
            "Prediction oluşturuldu: %s | cat=%s | conf=%.3f",
            pred.prediction_id, category, confidence,
        )
        return pred

    # ------------------------------------------------------------------
    # propose_action
    # ------------------------------------------------------------------

    def _build_propose_prompt(self, prediction: Prediction) -> str:
        """Eylem öneri prompt'unu oluşturur."""
        return f"""An AI agent made this prediction:
"{prediction.statement}"
Category: {prediction.category}
Confidence: {prediction.confidence_at_prediction:.3f}

Propose an action to VERIFY this prediction. Your options:

READ_FILE: read a file (parameters: {{"path": "..."}})
WEB_FETCH: fetch a URL (parameters: {{"url": "..."}})
EXECUTE_CODE: run Python code (parameters: {{"code": "..."}})
USER_QUESTION: ask the user (parameters: {{"question": "..."}})
SEARCH: search the web (parameters: {{"query": "..."}})
NO_ACTION: cannot be verified automatically (parameters: {{}})

Respond ONLY in JSON format, no other text:
{{
  "action_type": "...",
  "parameters": {{...}},
  "expected_outcome": "what you expect to find",
  "cost_estimate": <estimated seconds as float>,
  "justification": "why this action verifies the prediction"
}}"""

    def _parse_action_proposal(self, raw: str) -> Optional[ActionProposal]:
        """LLM yanıtından ActionProposal parse eder.

        Başarısız olursa None döner (caller NO_ACTION kullanır).
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.strip().rstrip("`").strip()

        # JSON nesnesini bul
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("ActionProposal JSON parse hatası: %s | raw: %r", exc, raw[:200])
            return None

        try:
            action_type_str = str(data.get("action_type", "no_action")).lower()
            # ActionType enum değerine dönüştür
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning("Bilinmeyen action_type: %r", action_type_str)
                return None

            return ActionProposal(
                action_type=action_type,
                parameters=data.get("parameters", {}),
                expected_outcome=str(data.get("expected_outcome", "")),
                cost_estimate=float(data.get("cost_estimate", 1.0)),
                justification=str(data.get("justification", "")),
            )
        except Exception as exc:
            logger.warning("ActionProposal oluşturma hatası: %s", exc)
            return None

    def _no_action_proposal(self, reason: str = NO_ACTION_FALLBACK_REASON) -> ActionProposal:
        """Fallback NO_ACTION proposal döndürür."""
        return ActionProposal(
            action_type=ActionType.NO_ACTION,
            parameters={},
            expected_outcome="",
            cost_estimate=0.0,
            justification=reason,
        )

    async def propose_action(self, prediction: Prediction) -> ActionProposal:
        """Tahmini doğrulamak için eylem önerir.

        LLM'e rubric tabanlı prompt ile eylem tipini ve parametrelerini
        ürettirir. Parse başarısız olursa NO_ACTION döner (asla None değil).

        Args:
            prediction: Doğrulanacak tahmin.

        Returns:
            ActionProposal nesnesi (en kötü ihtimalle NO_ACTION).
        """
        prompt = self._build_propose_prompt(prediction)
        messages = [{"role": "user", "content": prompt}]

        loop = asyncio.get_event_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.action_planner_llm.chat(messages),
                ),
                timeout=PROPOSE_ACTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Eylem önerisi LLM zaman aşımına uğradı.")
            return self._no_action_proposal("Timeout: LLM yanıt vermedi.")
        except Exception as exc:
            logger.warning("Eylem önerisi LLM hatası: %s", exc)
            return self._no_action_proposal(f"LLM hatası: {exc}")

        proposal = self._parse_action_proposal(raw)
        if proposal is None:
            return self._no_action_proposal()

        logger.info(
            "Eylem önerildi: %s | pred=%s",
            proposal.action_type, prediction.prediction_id,
        )
        return proposal

    # ------------------------------------------------------------------
    # execute_and_record
    # ------------------------------------------------------------------

    async def execute_and_record(
        self, prediction: Prediction, proposal: ActionProposal
    ) -> Outcome:
        """Eylemi yürütür, değerlendirir ve GroundingLog'a kaydeder.

        Args:
            prediction: İlgili tahmin.
            proposal: Yürütülecek eylem.

        Returns:
            Oluşturulmuş ve kaydedilmiş Outcome nesnesi.
        """
        result_str, exec_time, cost, error = await self.action_executor.execute(proposal)

        match_score, match_reasoning = await self.outcome_evaluator.evaluate(
            prediction, result_str
        )

        outcome = Outcome(
            prediction_id=prediction.prediction_id,
            action_executed=proposal,
            actual_result=result_str,
            match_score=match_score,
            match_reasoning=match_reasoning,
            execution_time_seconds=exec_time,
            cost_actual=cost,
            error=error,
        )

        self.grounding_log.add_outcome(outcome)
        logger.info(
            "Outcome kaydedildi: pred=%s | match=%.3f",
            prediction.prediction_id, match_score,
        )
        return outcome

    # ------------------------------------------------------------------
    # run_full_cycle
    # ------------------------------------------------------------------

    async def run_full_cycle(
        self,
        statement: str,
        category: str,
        confidence: float,
        context_block_ids: list[int],
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Prediction, Optional[Outcome]]:
        """Tam AOGL döngüsünü çalıştırır.

        make_prediction → propose_action → execute_and_record adımlarını
        sırayla yürütür.

        Eğer propose_action NO_ACTION döndürürse, Outcome oluşturulmaz
        ve tuple'ın ikinci elemanı None olur. Bu, tahmin GroundingLog'da
        "doğrulanmamış" olarak kalır ve kalibrasyon datasına katılmaz.

        Args:
            statement: Tahminin metinsel ifadesi.
            category: Tahmin kategorisi.
            confidence: CSM confidence skoru [0,1].
            context_block_ids: PSM blok ID'leri.
            metadata: İsteğe bağlı ek bilgi.

        Returns:
            (Prediction, Outcome | None) tuple'ı.
            Outcome None ise NO_ACTION veya yürütme hatası demektir.
        """
        pred = self.make_prediction(
            statement=statement,
            category=category,
            confidence=confidence,
            context_block_ids=context_block_ids,
            metadata=metadata,
        )

        proposal = await self.propose_action(pred)

        if proposal.action_type == ActionType.NO_ACTION:
            logger.info(
                "NO_ACTION: Tahmin %s doğrulanamaz. Gerekçe: %s",
                pred.prediction_id, proposal.justification,
            )
            return (pred, None)

        outcome = await self.execute_and_record(pred, proposal)
        return (pred, outcome)

    # ------------------------------------------------------------------
    # update_calibration
    # ------------------------------------------------------------------

    async def update_calibration(self) -> dict[str, Any]:
        """Kalibrasyon modellerini günceller ve CSM'e uygular.

        GroundingLog'daki mevcut veriyi kullanarak CalibrationLearner'ı
        yeniden eğitir ve CompositeConfidenceScorer'a uygular.

        Returns:
            {
                "categories_learned": int,
                "total_samples": int,
                "categories": list[str],
            }
        """
        # SQLite connection'ı thread-safe olsa da, CalibrationLearner
        # sync çalıştırıldığında event loop bloke olmaz (I/O yok, sadece hesaplama)
        calibrators = self.calibration_learner.learn_from_log()
        self.calibration_learner.apply_to_csm(self.csm_scorer)

        # İstatistik
        cal_data = self.grounding_log.get_calibration_data()
        total_samples = sum(len(v) for v in cal_data.values())

        stats = {
            "categories_learned": len(calibrators),
            "total_samples": total_samples,
            "categories": list(calibrators.keys()),
        }
        logger.info("Kalibrasyon güncellendi: %s", stats)
        return stats
