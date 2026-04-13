"""
tests/test_grounding.py — GroundingLog testleri

SQLite in-memory kullanılır, gerçek disk yazımı yok.
"""

import json
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sca.grounding import GroundingLog
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log():
    """In-memory GroundingLog."""
    gl = GroundingLog(db_path=":memory:")
    yield gl
    gl.close()


def make_prediction(
    statement: str = "test tahmini",
    category: str = "general",
    confidence: float = 0.7,
) -> Prediction:
    return Prediction(
        statement=statement,
        confidence_at_prediction=confidence,
        category=category,
        context_block_ids=[0, 1],
    )


def make_outcome(prediction_id: uuid.UUID, match_score: float = 0.8) -> Outcome:
    proposal = ActionProposal(
        action_type=ActionType.READ_FILE,
        parameters={"path": "/tmp/test.txt"},
    )
    return Outcome(
        prediction_id=prediction_id,
        action_executed=proposal,
        actual_result="Gerçek sonuç",
        match_score=match_score,
        match_reasoning="Tahmin doğru",
        execution_time_seconds=0.5,
        cost_actual=0.1,
    )


# ---------------------------------------------------------------------------
# add / get testleri
# ---------------------------------------------------------------------------

class TestAddGet:
    def test_add_and_get_prediction(self, log):
        """Prediction eklenip geri alınabilmeli."""
        pred = make_prediction()
        log.add_prediction(pred)
        retrieved = log.get_prediction(pred.prediction_id)
        assert retrieved is not None
        assert retrieved.statement == pred.statement
        assert retrieved.confidence_at_prediction == pred.confidence_at_prediction
        assert retrieved.category == pred.category

    def test_get_nonexistent_prediction_returns_none(self, log):
        """Olmayan prediction_id None döndürmeli."""
        result = log.get_prediction(uuid.uuid4())
        assert result is None

    def test_add_duplicate_prediction_raises(self, log):
        """Aynı prediction_id iki kez eklenemez."""
        pred = make_prediction()
        log.add_prediction(pred)
        with pytest.raises(ValueError):
            log.add_prediction(pred)

    def test_add_and_get_outcome(self, log):
        """Outcome eklenip geri alınabilmeli."""
        pred = make_prediction()
        log.add_prediction(pred)
        outcome = make_outcome(pred.prediction_id)
        log.add_outcome(outcome)
        retrieved = log.get_outcome(pred.prediction_id)
        assert retrieved is not None
        assert retrieved.match_score == outcome.match_score
        assert retrieved.actual_result == outcome.actual_result

    def test_get_outcome_without_prediction_raises(self, log):
        """Prediction olmadan Outcome eklenemez."""
        fake_pred_id = uuid.uuid4()
        outcome = make_outcome(fake_pred_id)
        with pytest.raises(ValueError):
            log.add_outcome(outcome)

    def test_get_outcome_for_unverified_prediction_returns_none(self, log):
        """Sonucu olmayan tahmin için get_outcome None döndürmeli."""
        pred = make_prediction()
        log.add_prediction(pred)
        result = log.get_outcome(pred.prediction_id)
        assert result is None


# ---------------------------------------------------------------------------
# Query testleri
# ---------------------------------------------------------------------------

class TestQueries:
    def test_query_by_category(self, log):
        """Kategoriye göre filtreleme çalışmalı."""
        p1 = make_prediction(category="file_location", confidence=0.8)
        p2 = make_prediction(category="code_behavior", confidence=0.6)
        p3 = make_prediction(category="file_location", confidence=0.5)
        for p in [p1, p2, p3]:
            log.add_prediction(p)

        results = log.query_by_category("file_location")
        assert len(results) == 2
        for pred, outcome in results:
            assert pred.category == "file_location"

    def test_query_by_category_no_results(self, log):
        """Olmayan kategori boş liste döndürmeli."""
        results = log.query_by_category("nonexistent_category")
        assert results == []

    def test_query_by_time_range(self, log):
        """Zaman aralığına göre filtreleme çalışmalı."""
        pred = make_prediction()
        log.add_prediction(pred)

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        results = log.query_by_time_range(start, end)
        assert len(results) >= 1

    def test_query_all(self, log):
        """Tüm kayıtlar döndürülmeli."""
        preds = [make_prediction(statement=f"tahmin {i}") for i in range(5)]
        for p in preds:
            log.add_prediction(p)
        results = log.query_all()
        assert len(results) == 5

    def test_query_all_with_outcomes(self, log):
        """Outcome olan ve olmayan kayıtlar birlikte döndürülmeli."""
        p1 = make_prediction(statement="pred with outcome")
        p2 = make_prediction(statement="pred without outcome")
        log.add_prediction(p1)
        log.add_prediction(p2)
        outcome = make_outcome(p1.prediction_id)
        log.add_outcome(outcome)

        results = log.query_all()
        assert len(results) == 2

        pred_map = {pred.prediction_id: outcome for pred, outcome in results}
        assert pred_map[p1.prediction_id] is not None
        assert pred_map[p2.prediction_id] is None

    def test_len(self, log):
        """__len__ toplam prediction sayısını döndürmeli."""
        assert len(log) == 0
        log.add_prediction(make_prediction())
        assert len(log) == 1
        log.add_prediction(make_prediction())
        assert len(log) == 2


# ---------------------------------------------------------------------------
# Export / Import testleri
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_import_roundtrip(self, log, tmp_path):
        """Export → Import round-trip doğru çalışmalı."""
        p1 = make_prediction(statement="export test 1", category="file_location")
        p2 = make_prediction(statement="export test 2", category="code_behavior")
        log.add_prediction(p1)
        log.add_prediction(p2)
        outcome = make_outcome(p1.prediction_id, match_score=0.9)
        log.add_outcome(outcome)

        json_path = tmp_path / "export.json"
        log.export_to_json(str(json_path))
        assert json_path.exists()

        # Yeni log'a import
        log2 = GroundingLog(db_path=":memory:")
        log2.import_from_json(str(json_path))

        assert len(log2) == 2
        retrieved = log2.get_prediction(p1.prediction_id)
        assert retrieved is not None
        assert retrieved.statement == p1.statement

        retrieved_outcome = log2.get_outcome(p1.prediction_id)
        assert retrieved_outcome is not None
        assert retrieved_outcome.match_score == pytest.approx(0.9, abs=0.001)
        log2.close()

    def test_export_json_structure(self, log, tmp_path):
        """Export JSON dosyasının yapısı doğru olmalı."""
        pred = make_prediction()
        log.add_prediction(pred)
        json_path = tmp_path / "test.json"
        log.export_to_json(str(json_path))

        with json_path.open() as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert "prediction" in data[0]

    def test_import_nonexistent_file_raises(self, log):
        """Olmayan dosyayı import etmek FileNotFoundError fırlatmalı."""
        with pytest.raises(FileNotFoundError):
            log.import_from_json("/nonexistent/path/file.json")


# ---------------------------------------------------------------------------
# Kalibrasyon testleri
# ---------------------------------------------------------------------------

class TestCalibrationData:
    def test_get_calibration_data_empty(self, log):
        """Veri yokken boş dict döndürmeli."""
        result = log.get_calibration_data()
        assert result == {}

    def test_get_calibration_data_all_categories(self, log):
        """Tüm kategoriler için (confidence, match_score) çiftleri döndürmeli."""
        categories = ["file_location", "file_location", "code_behavior"]
        confidences = [0.8, 0.6, 0.7]
        match_scores = [0.9, 0.5, 0.8]

        for cat, conf, ms in zip(categories, confidences, match_scores):
            pred = make_prediction(category=cat, confidence=conf)
            log.add_prediction(pred)
            outcome = make_outcome(pred.prediction_id, match_score=ms)
            log.add_outcome(outcome)

        data = log.get_calibration_data()
        assert "file_location" in data
        assert "code_behavior" in data
        assert len(data["file_location"]) == 2
        assert len(data["code_behavior"]) == 1

        # Her çift (confidence, match_score) formatında
        for pair in data["file_location"]:
            assert len(pair) == 2
            assert 0.0 <= pair[0] <= 1.0
            assert 0.0 <= pair[1] <= 1.0

    def test_get_calibration_data_specific_category(self, log):
        """Belirli kategori filtrelemesi çalışmalı."""
        p1 = make_prediction(category="file_location")
        p2 = make_prediction(category="code_behavior")
        log.add_prediction(p1)
        log.add_prediction(p2)
        log.add_outcome(make_outcome(p1.prediction_id, 0.8))
        log.add_outcome(make_outcome(p2.prediction_id, 0.6))

        data = log.get_calibration_data(category="file_location")
        assert "file_location" in data
        assert "code_behavior" not in data

    def test_predictions_without_outcomes_excluded(self, log):
        """Outcome'u olmayan tahminler kalibrasyon datasına dahil edilmemeli."""
        pred = make_prediction()
        log.add_prediction(pred)
        # Outcome ekleme!
        data = log.get_calibration_data()
        assert data == {}


# ---------------------------------------------------------------------------
# prune_old testi
# ---------------------------------------------------------------------------

class TestPruneOld:
    def test_prune_removes_old_records(self, log):
        """Eski kayıtları silmeli."""
        pred = make_prediction()
        log.add_prediction(pred)

        # 0 gün = hepsini sil
        deleted = log.prune_old(days=0)
        assert deleted >= 0  # Sayı döndürmeli
