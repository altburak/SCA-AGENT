"""
sca/agent.py — StratifiedAgent

PSM + CSM + AOGL + CED bileşenlerini birleştiren üst düzey agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from sca.actions import create_default_executor
from sca.aogl import AOGLController
from sca.calibration import CalibrationLearner
from sca.confidence import (
    CompositeConfidenceScorer,
    ProvenancePenaltyCalculator,
    SelfConsistencyScorer,
    VerifierScorer,
)
from sca.context import ContextBlock, ContextManager, Provenance
from sca.evaluation import OutcomeEvaluator
from sca.grounding import GroundingLog
from sca.llm import LLMClient
from sca.prediction import Prediction

if TYPE_CHECKING:
    from sca.ced import DistillationOrchestrator
    from sca.episode import Episode

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD: float = 0.5
EXTRACT_PREDICTIONS_TIMEOUT: float = 20.0
DEFAULT_DB_PATH: str = "sca_grounding.db"


class StratifiedAgent:
    """PSM + CSM + AOGL + optional CED agent."""

    def __init__(
        self,
        psm_manager: ContextManager,
        csm_scorer: CompositeConfidenceScorer,
        aogl_controller: AOGLController,
        main_llm: LLMClient,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        ced_orchestrator: Optional["DistillationOrchestrator"] = None,
    ) -> None:
        self.psm_manager = psm_manager
        self.csm_scorer = csm_scorer
        self.aogl_controller = aogl_controller
        self.main_llm = main_llm
        self.low_confidence_threshold = low_confidence_threshold
        self.ced_orchestrator = ced_orchestrator
        self._session_active: bool = False
        self._current_episode: Optional["Episode"] = None
        self._session_prediction_ids: list[uuid.UUID] = []
        self._augmented_system_prompt: Optional[str] = None
        logger.info("StratifiedAgent başlatıldı.")

    async def start_session(
        self,
        initial_prompt: str = "",
        domain_hint: Optional[str] = None,
    ) -> dict:
        """Start a new session, optionally augmenting via CED."""
        from sca.episode import Episode

        self._session_active = True
        self.psm_manager.clear()
        self._session_prediction_ids = []
        self._current_episode = Episode(
            start_time=datetime.now(timezone.utc),
            initial_prompt=initial_prompt or None,
            domain=domain_hint,
        )

        if self.ced_orchestrator is None:
            logger.info("Agent oturumu başlatıldı (CED yok).")
            return {}

        result = self.ced_orchestrator.on_episode_start(
            initial_prompt=initial_prompt,
            domain_hint=domain_hint,
        )
        self._augmented_system_prompt = result.get("augmented_system_prompt")
        logger.info("Agent oturumu başlatıldı (CED).")
        return result

    async def end_session(self) -> dict:
        """End current session, distilling insights if CED is active."""
        self._session_active = False

        if self._current_episode is None:
            return {}

        self._current_episode.end_time = datetime.now(timezone.utc)
        self._current_episode.prediction_ids = list(self._session_prediction_ids)
        self._current_episode.context_block_ids = [
            b.block_id for b in self.psm_manager.blocks
        ]

        if self.ced_orchestrator is None:
            logger.info("Agent oturumu kapatıldı (CED yok).")
            self._current_episode = None
            self._augmented_system_prompt = None
            return {}

        try:
            result = await self.ced_orchestrator.on_episode_end(self._current_episode)
        except Exception as exc:
            logger.warning("on_episode_end failed: %s", exc)
            result = {"insights_extracted": 0, "insights_merged": 0}

        self._current_episode = None
        self._augmented_system_prompt = None
        logger.info("Agent oturumu kapatıldı (CED): %s", result)
        return result

    async def __aenter__(self) -> "StratifiedAgent":
        await self.start_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.end_session()

    async def _extract_predictions(self, response: str) -> list[dict[str, Any]]:
        extract_prompt = f"""Analyze this AI assistant response and identify any verifiable factual predictions or claims.

RESPONSE:
{response[:2000]}

Extract claims that can be verified by reading a file, running code, or searching the web.
Respond ONLY in JSON format:
[
  {{
    "statement": "exact claim that can be verified",
    "category": "file_location|code_behavior|factual_claim|tool_output|data_extraction|general",
    "confidence": <0.0-1.0 estimated confidence>
  }}
]

If there are no verifiable predictions, respond with: []"""

        messages = [{"role": "user", "content": extract_prompt}]
        loop = asyncio.get_event_loop()

        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.main_llm.chat(messages)),
                timeout=EXTRACT_PREDICTIONS_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Tahmin çıkarma başarısız: %s", exc)
            return []

        cleaned = raw.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.strip().rstrip("`").strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if arr_match:
            try:
                data = json.loads(arr_match.group())
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        return []

    async def chat(self, user_message: str) -> str:
        """Process user message and return agent response."""
        user_block = ContextBlock(
            content=user_message,
            provenance=Provenance.USER,
            confidence=1.0,
        )
        user_block_id = self.psm_manager.add(user_block)

        from sca.formatter import PromptFormatter
        formatter = PromptFormatter()
        system_prompt = (
            self._augmented_system_prompt
            if self._augmented_system_prompt
            else formatter.get_system_prompt()
        )
        messages = self.psm_manager.to_messages(system_prompt=system_prompt)
        messages.append({"role": "user", "content": user_message})

        loop = asyncio.get_event_loop()
        try:
            llm_response = await loop.run_in_executor(
                None, lambda: self.main_llm.chat(messages)
            )
        except Exception as exc:
            logger.error("LLM çağrısı başarısız: %s", exc)
            return f"Üzgünüm, bir hata oluştu: {exc}"

        response_block = ContextBlock(
            content=llm_response,
            provenance=Provenance.SELF_GENERATED,
            confidence=0.7,
            derived_from=[user_block_id],
        )
        response_block_id = self.psm_manager.add(response_block)

        try:
            confidence_result = await self.csm_scorer.score(
                prompt=user_message,
                response=llm_response,
                context_manager=self.psm_manager,
                derived_from_ids=[user_block_id],
            )
            final_score = confidence_result.final_score
        except Exception as exc:
            logger.warning("CSM scoring başarısız: %s", exc)
            final_score = 0.7

        predictions_data = await self._extract_predictions(llm_response)

        for pd in predictions_data[:3]:
            try:
                stmt = pd.get("statement", "")
                cat = pd.get("category", "general")
                conf = float(pd.get("confidence", final_score))
                if not stmt:
                    continue

                pred, outcome = await self.aogl_controller.run_full_cycle(
                    statement=stmt,
                    category=cat,
                    confidence=conf,
                    context_block_ids=[user_block_id, response_block_id],
                )

                if self._current_episode is not None:
                    self._session_prediction_ids.append(pred.prediction_id)

                logger.info(
                    "AOGL: tahmin=%s | outcome=%s",
                    pred.prediction_id,
                    outcome.match_score if outcome else "NO_ACTION",
                )
            except Exception as exc:
                logger.warning("AOGL döngüsü başarısız: %s", exc)

        final_response = llm_response
        if final_score < self.low_confidence_threshold:
            warning = (
                f"\n\n⚠️ **Uyarı**: Bu cevabın güven skoru düşük ({final_score:.2f}). "
                "Önemli kararlar için lütfen doğrulamayı düşünün."
            )
            final_response += warning
            logger.info("Düşük confidence uyarısı eklendi: %.3f", final_score)

        return final_response


def create_default_agent(
    api_key: str,
    allowed_dirs: Optional[list[str | Path]] = None,
    db_path: str = DEFAULT_DB_PATH,
    sandbox: bool = True,
    model: str = "groq/llama-3.3-70b-versatile",
    min_samples_for_calibration: int = 20,
    ced_orchestrator: Optional["DistillationOrchestrator"] = None,
) -> StratifiedAgent:
    """Factory: create a fully configured StratifiedAgent."""
    main_llm = LLMClient(model=model, api_key=api_key)
    verifier_llm = LLMClient(model=model, api_key=api_key)
    psm_manager = ContextManager()

    sc_scorer = SelfConsistencyScorer(llm_client=main_llm, n_samples=3)
    v_scorer = VerifierScorer(verifier_llm_client=verifier_llm)
    prov_calc = ProvenancePenaltyCalculator(context_manager=psm_manager)

    csm_scorer = CompositeConfidenceScorer(
        self_consistency_scorer=sc_scorer,
        verifier_scorer=v_scorer,
        provenance_calculator=prov_calc,
    )

    executor = create_default_executor(allowed_dirs=allowed_dirs, sandbox=sandbox)
    evaluator = OutcomeEvaluator(llm_client=main_llm)
    grounding_log = GroundingLog(db_path=db_path)
    calibration_learner = CalibrationLearner(
        grounding_log=grounding_log,
        min_samples_per_category=min_samples_for_calibration,
    )

    aogl_ctrl = AOGLController(
        psm_manager=psm_manager,
        csm_scorer=csm_scorer,
        action_executor=executor,
        outcome_evaluator=evaluator,
        grounding_log=grounding_log,
        calibration_learner=calibration_learner,
        action_planner_llm=main_llm,
    )

    return StratifiedAgent(
        psm_manager=psm_manager,
        csm_scorer=csm_scorer,
        aogl_controller=aogl_ctrl,
        main_llm=main_llm,
        ced_orchestrator=ced_orchestrator,
    )
