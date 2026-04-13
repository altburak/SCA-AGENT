"""
tests/test_prediction.py — Prediction, ActionProposal, Outcome testleri
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from sca.prediction import ActionProposal, ActionType, Outcome, Prediction


# ---------------------------------------------------------------------------
# Prediction testleri
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_basic_creation(self):
        """Temel Prediction oluşturma."""
        pred = Prediction(
            statement="config.py /etc/app/ dizinindedir",
            confidence_at_prediction=0.8,
            category="file_location",
            context_block_ids=[0, 1],
        )
        assert pred.statement == "config.py /etc/app/ dizinindedir"
        assert pred.confidence_at_prediction == 0.8
        assert pred.category == "file_location"
        assert isinstance(pred.prediction_id, uuid.UUID)
        assert isinstance(pred.timestamp, datetime)

    def test_auto_generated_uuid(self):
        """Her Prediction farklı UUID almalı."""
        p1 = Prediction(statement="a", confidence_at_prediction=0.5, category="general")
        p2 = Prediction(statement="b", confidence_at_prediction=0.5, category="general")
        assert p1.prediction_id != p2.prediction_id

    def test_confidence_boundary_values(self):
        """Confidence sınır değerleri: 0.0 ve 1.0 geçerli."""
        p0 = Prediction(statement="test", confidence_at_prediction=0.0, category="general")
        assert p0.confidence_at_prediction == 0.0
        p1 = Prediction(statement="test", confidence_at_prediction=1.0, category="general")
        assert p1.confidence_at_prediction == 1.0

    def test_confidence_out_of_range_raises(self):
        """Confidence [0,1] dışı ValueError fırlatmalı."""
        with pytest.raises(Exception):
            Prediction(statement="test", confidence_at_prediction=1.5, category="general")
        with pytest.raises(Exception):
            Prediction(statement="test", confidence_at_prediction=-0.1, category="general")

    def test_empty_statement_raises(self):
        """Boş statement ValueError fırlatmalı."""
        with pytest.raises(Exception):
            Prediction(statement="", confidence_at_prediction=0.5, category="general")
        with pytest.raises(Exception):
            Prediction(statement="   ", confidence_at_prediction=0.5, category="general")

    def test_empty_category_raises(self):
        """Boş category ValueError fırlatmalı."""
        with pytest.raises(Exception):
            Prediction(statement="test", confidence_at_prediction=0.5, category="")

    def test_category_lowercased(self):
        """Category otomatik lowercase yapılmalı."""
        pred = Prediction(statement="test", confidence_at_prediction=0.5, category="FILE_LOCATION")
        assert pred.category == "file_location"

    def test_serialization_roundtrip(self):
        """model_dump → model_validate round-trip çalışmalı."""
        pred = Prediction(
            statement="fonksiyon 42 döndürür",
            confidence_at_prediction=0.75,
            category="code_behavior",
            context_block_ids=[0, 2, 5],
            metadata={"source": "llm_output"},
        )
        data = pred.model_dump(mode="json")
        restored = Prediction.model_validate(data)
        assert restored.prediction_id == pred.prediction_id
        assert restored.statement == pred.statement
        assert restored.confidence_at_prediction == pred.confidence_at_prediction
        assert restored.category == pred.category
        assert restored.context_block_ids == pred.context_block_ids

    def test_json_serialization(self):
        """JSON serialize/deserialize round-trip çalışmalı."""
        pred = Prediction(
            statement="test statement",
            confidence_at_prediction=0.6,
            category="factual_claim",
        )
        json_str = pred.model_dump_json()
        restored = Prediction.model_validate_json(json_str)
        assert restored.prediction_id == pred.prediction_id


# ---------------------------------------------------------------------------
# ActionProposal testleri
# ---------------------------------------------------------------------------

class TestActionProposal:
    def test_basic_creation(self):
        """Temel ActionProposal oluşturma."""
        proposal = ActionProposal(
            action_type=ActionType.READ_FILE,
            parameters={"path": "/tmp/test.txt"},
            expected_outcome="Dosyanın ilk satırı 'hello'",
            cost_estimate=0.1,
            justification="Dosyayı okuyarak doğrulayabiliriz",
        )
        assert proposal.action_type == ActionType.READ_FILE
        assert proposal.parameters["path"] == "/tmp/test.txt"
        assert proposal.cost_estimate == 0.1

    def test_all_action_types(self):
        """Tüm ActionType değerleri geçerli olmalı."""
        for at in ActionType:
            proposal = ActionProposal(action_type=at, parameters={})
            assert proposal.action_type == at

    def test_no_action_type(self):
        """NO_ACTION tipi geçerli olmalı."""
        proposal = ActionProposal(action_type=ActionType.NO_ACTION)
        assert proposal.action_type == ActionType.NO_ACTION

    def test_serialization_roundtrip(self):
        """model_dump round-trip."""
        proposal = ActionProposal(
            action_type=ActionType.EXECUTE_CODE,
            parameters={"code": "print(2+2)"},
            expected_outcome="4",
            cost_estimate=5.0,
            justification="Kodu çalıştırarak doğrularız",
        )
        data = proposal.model_dump(mode="json")
        restored = ActionProposal.model_validate(data)
        assert restored.action_type == proposal.action_type
        assert restored.parameters == proposal.parameters


# ---------------------------------------------------------------------------
# Outcome testleri
# ---------------------------------------------------------------------------

class TestOutcome:
    def _make_proposal(self) -> ActionProposal:
        return ActionProposal(
            action_type=ActionType.READ_FILE,
            parameters={"path": "/tmp/test.txt"},
        )

    def test_basic_creation(self):
        """Temel Outcome oluşturma."""
        pred_id = uuid.uuid4()
        outcome = Outcome(
            prediction_id=pred_id,
            action_executed=self._make_proposal(),
            actual_result="Dosya içeriği: hello world",
            match_score=0.9,
            match_reasoning="Tahmin doğru",
        )
        assert outcome.prediction_id == pred_id
        assert outcome.match_score == 0.9
        assert isinstance(outcome.outcome_id, uuid.UUID)

    def test_match_score_boundaries(self):
        """match_score 0.0 ve 1.0 geçerli."""
        pred_id = uuid.uuid4()
        o0 = Outcome(
            prediction_id=pred_id,
            action_executed=self._make_proposal(),
            match_score=0.0,
        )
        assert o0.match_score == 0.0
        o1 = Outcome(
            prediction_id=pred_id,
            action_executed=self._make_proposal(),
            match_score=1.0,
        )
        assert o1.match_score == 1.0

    def test_match_score_out_of_range_raises(self):
        """match_score [0,1] dışı ValueError fırlatmalı."""
        pred_id = uuid.uuid4()
        with pytest.raises(Exception):
            Outcome(
                prediction_id=pred_id,
                action_executed=self._make_proposal(),
                match_score=1.5,
            )

    def test_error_field_optional(self):
        """error field None veya string olabilmeli."""
        pred_id = uuid.uuid4()
        o = Outcome(
            prediction_id=pred_id,
            action_executed=self._make_proposal(),
            error=None,
        )
        assert o.error is None
        o2 = Outcome(
            prediction_id=pred_id,
            action_executed=self._make_proposal(),
            error="Dosya bulunamadı",
        )
        assert o2.error == "Dosya bulunamadı"

    def test_serialization_roundtrip(self):
        """model_dump round-trip."""
        pred_id = uuid.uuid4()
        outcome = Outcome(
            prediction_id=pred_id,
            action_executed=self._make_proposal(),
            actual_result="sonuç",
            match_score=0.75,
            match_reasoning="kısmen doğru",
            execution_time_seconds=1.5,
            cost_actual=0.5,
        )
        data = outcome.model_dump(mode="json")
        restored = Outcome.model_validate(data)
        assert restored.outcome_id == outcome.outcome_id
        assert restored.match_score == outcome.match_score
        assert restored.execution_time_seconds == outcome.execution_time_seconds
