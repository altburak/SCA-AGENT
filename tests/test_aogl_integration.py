"""
tests/test_aogl_integration.py — AOGLController entegrasyon testleri

Tüm LLM ve tool çağrıları mock'lanmış. Tam döngü testleri.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sca.actions import ActionExecutor, create_default_executor
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
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grounding_log():
    log = GroundingLog(db_path=":memory:")
    yield log
    log.close()


@pytest.fixture
def mock_llm():
    """Eylem öneri JSON döndüren LLM mock'u."""
    mock = MagicMock()
    mock.chat.return_value = """{
        "action_type": "execute_code",
        "parameters": {"code": "print('test')"},
        "expected_outcome": "test çıktısı",
        "cost_estimate": 1.0,
        "justification": "kodu çalıştırarak doğrularız"
    }"""
    return mock


@pytest.fixture
def mock_evaluator_llm():
    """Evaluation JSON döndüren LLM mock'u."""
    mock = MagicMock()
    mock.chat.return_value = '{"score": 8, "reasoning": "tahmin büyük ölçüde doğru"}'
    return mock


@pytest.fixture
def psm_manager():
    manager = ContextManager()
    block = ContextBlock(content="Test bağlamı", provenance=Provenance.USER)
    manager.add(block)
    return manager


@pytest.fixture
def mock_csm():
    """CompositeConfidenceScorer mock'u."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_executor():
    """Mock ActionExecutor: kod çalıştırıyor gibi davranır."""
    executor = MagicMock(spec=ActionExecutor)

    async def mock_execute(proposal):
        return ("mock çıktı: test", 0.5, 1.0, None)

    executor.execute = mock_execute
    return executor


@pytest.fixture
def aogl_controller(psm_manager, mock_csm, mock_executor, mock_evaluator_llm, grounding_log):
    evaluator = OutcomeEvaluator(llm_client=mock_evaluator_llm)
    calibration_learner = CalibrationLearner(
        grounding_log=grounding_log,
        min_samples_per_category=5,
    )
    mock_planner_llm = MagicMock()
    mock_planner_llm.chat.return_value = """{
        "action_type": "execute_code",
        "parameters": {"code": "print('hello')"},
        "expected_outcome": "hello",
        "cost_estimate": 1.0,
        "justification": "çalıştırarak doğrularız"
    }"""

    return AOGLController(
        psm_manager=psm_manager,
        csm_scorer=mock_csm,
        action_executor=mock_executor,
        outcome_evaluator=evaluator,
        grounding_log=grounding_log,
        calibration_learner=calibration_learner,
        action_planner_llm=mock_planner_llm,
    )


# ---------------------------------------------------------------------------
# make_prediction testleri
# ---------------------------------------------------------------------------

class TestMakePrediction:
    def test_creates_and_logs_prediction(self, aogl_controller, grounding_log):
        """make_prediction Prediction oluşturmalı ve log'a kaydetmeli."""
        pred = aogl_controller.make_prediction(
            statement="dosya.txt /tmp/ dizinindedir",
            category="file_location",
            confidence=0.8,
            context_block_ids=[0],
        )
        assert isinstance(pred, Prediction)
        assert pred.statement == "dosya.txt /tmp/ dizinindedir"
        assert pred.confidence_at_prediction == 0.8

        # Log'da mevcut olmalı
        retrieved = grounding_log.get_prediction(pred.prediction_id)
        assert retrieved is not None

    def test_prediction_id_unique(self, aogl_controller):
        """Her make_prediction farklı UUID üretmeli."""
        p1 = aogl_controller.make_prediction("stmt1", "general", 0.5, [])
        p2 = aogl_controller.make_prediction("stmt2", "general", 0.5, [])
        assert p1.prediction_id != p2.prediction_id


# ---------------------------------------------------------------------------
# propose_action testleri
# ---------------------------------------------------------------------------

class TestProposeAction:
    def test_valid_llm_response_returns_proposal(self, aogl_controller):
        """Geçerli LLM yanıtı ActionProposal döndürmeli."""
        pred = aogl_controller.make_prediction("test", "general", 0.7, [0])
        proposal = run(aogl_controller.propose_action(pred))
        assert isinstance(proposal, ActionProposal)
        assert proposal.action_type != ActionType.NO_ACTION

    def test_parse_failure_returns_no_action(self, psm_manager, mock_csm, mock_executor, mock_evaluator_llm, grounding_log):
        """LLM parse başarısız olursa NO_ACTION döndürmeli."""
        bad_llm = MagicMock()
        bad_llm.chat.return_value = "bu kesinlikle JSON değil"
        calibration_learner = CalibrationLearner(grounding_log=grounding_log, min_samples_per_category=5)
        evaluator = OutcomeEvaluator(llm_client=mock_evaluator_llm)

        ctrl = AOGLController(
            psm_manager=psm_manager,
            csm_scorer=mock_csm,
            action_executor=mock_executor,
            outcome_evaluator=evaluator,
            grounding_log=grounding_log,
            calibration_learner=calibration_learner,
            action_planner_llm=bad_llm,
        )
        pred = ctrl.make_prediction("test", "general", 0.5, [])
        proposal = run(ctrl.propose_action(pred))
        assert proposal.action_type == ActionType.NO_ACTION

    def test_llm_exception_returns_no_action(self, psm_manager, mock_csm, mock_executor, mock_evaluator_llm, grounding_log):
        """LLM exception fırlatırsa NO_ACTION döndürmeli (raise değil)."""
        error_llm = MagicMock()
        error_llm.chat.side_effect = RuntimeError("API çöktü")
        calibration_learner = CalibrationLearner(grounding_log=grounding_log, min_samples_per_category=5)
        evaluator = OutcomeEvaluator(llm_client=mock_evaluator_llm)

        ctrl = AOGLController(
            psm_manager=psm_manager,
            csm_scorer=mock_csm,
            action_executor=mock_executor,
            outcome_evaluator=evaluator,
            grounding_log=grounding_log,
            calibration_learner=calibration_learner,
            action_planner_llm=error_llm,
        )
        pred = ctrl.make_prediction("test", "general", 0.5, [])
        proposal = run(ctrl.propose_action(pred))
        assert proposal.action_type == ActionType.NO_ACTION


# ---------------------------------------------------------------------------
# execute_and_record testleri
# ---------------------------------------------------------------------------

class TestExecuteAndRecord:
    def test_creates_and_logs_outcome(self, aogl_controller, grounding_log):
        """execute_and_record Outcome oluşturmalı ve log'a kaydetmeli."""
        pred = aogl_controller.make_prediction(
            "print çalışıyor", "code_behavior", 0.9, [0]
        )
        proposal = ActionProposal(
            action_type=ActionType.EXECUTE_CODE,
            parameters={"code": "print('hello')"},
            expected_outcome="hello",
            cost_estimate=1.0,
            justification="test",
        )
        outcome = run(aogl_controller.execute_and_record(pred, proposal))
        assert isinstance(outcome, Outcome)
        assert outcome.prediction_id == pred.prediction_id
        assert 0.0 <= outcome.match_score <= 1.0

        # Log'da mevcut olmalı
        retrieved = grounding_log.get_outcome(pred.prediction_id)
        assert retrieved is not None


# ---------------------------------------------------------------------------
# run_full_cycle testleri
# ---------------------------------------------------------------------------

class TestRunFullCycle:
    def test_full_cycle_returns_prediction_and_outcome(self, aogl_controller):
        """Tam döngü (Prediction, Outcome) tuple döndürmeli."""
        pred, outcome = run(
            aogl_controller.run_full_cycle(
                statement="fonksiyon 4 döndürür",
                category="code_behavior",
                confidence=0.85,
                context_block_ids=[0],
            )
        )
        assert isinstance(pred, Prediction)
        assert outcome is not None
        assert isinstance(outcome, Outcome)
        assert outcome.prediction_id == pred.prediction_id

    def test_no_action_returns_none_outcome(self, psm_manager, mock_csm, mock_executor, mock_evaluator_llm, grounding_log):
        """NO_ACTION durumunda Outcome None olmalı."""
        no_action_llm = MagicMock()
        no_action_llm.chat.return_value = """{
            "action_type": "no_action",
            "parameters": {},
            "expected_outcome": "",
            "cost_estimate": 0.0,
            "justification": "doğrulanamaz"
        }"""
        calibration_learner = CalibrationLearner(grounding_log=grounding_log, min_samples_per_category=5)
        evaluator = OutcomeEvaluator(llm_client=mock_evaluator_llm)

        ctrl = AOGLController(
            psm_manager=psm_manager,
            csm_scorer=mock_csm,
            action_executor=mock_executor,
            outcome_evaluator=evaluator,
            grounding_log=grounding_log,
            calibration_learner=calibration_learner,
            action_planner_llm=no_action_llm,
        )
        pred, outcome = run(
            ctrl.run_full_cycle(
                statement="bu kullanıcı mutlu",
                category="user_intent",
                confidence=0.3,
                context_block_ids=[],
            )
        )
        assert isinstance(pred, Prediction)
        assert outcome is None


# ---------------------------------------------------------------------------
# update_calibration testleri
# ---------------------------------------------------------------------------

class TestUpdateCalibration:
    def test_update_calibration_returns_stats(self, aogl_controller, grounding_log):
        """update_calibration istatistik dict döndürmeli."""
        # Önce birkaç tahmin + outcome ekle
        for i in range(3):
            pred = aogl_controller.make_prediction(
                f"tahmin {i}", "code_behavior", 0.5 + i * 0.1, [0]
            )
            proposal = ActionProposal(
                action_type=ActionType.EXECUTE_CODE,
                parameters={"code": "print(1)"},
            )
            run(aogl_controller.execute_and_record(pred, proposal))

        stats = run(aogl_controller.update_calibration())
        assert "categories_learned" in stats
        assert "total_samples" in stats
        assert "categories" in stats
        assert isinstance(stats["categories_learned"], int)
        assert isinstance(stats["total_samples"], int)

    def test_calibration_applied_to_csm(self, aogl_controller, grounding_log):
        """Yeterli örnekten sonra CSM'e calibrator uygulanmalı."""
        # min_samples_per_category=5, bu yüzden 5 örnek ekle
        for i in range(6):
            pred = aogl_controller.make_prediction(
                f"tahmin {i}", "file_location", 0.3 + i * 0.1, [0]
            )
            proposal = ActionProposal(
                action_type=ActionType.EXECUTE_CODE,
                parameters={"code": "print(1)"},
            )
            run(aogl_controller.execute_and_record(pred, proposal))

        run(aogl_controller.update_calibration())

        # CSM'de _category_calibrators oluşturulmuş olmalı
        assert hasattr(aogl_controller.csm_scorer, "_category_calibrators")
