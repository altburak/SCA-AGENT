"""
sca/grounding.py — GroundingLog (SQLite Persistence)

SQLAlchemy ORM tabanlı kalıcı depolama.
Prediction ve Outcome nesnelerini saklar, sorgular ve kalibrasyon verisi üretir.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    delete,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from sca.prediction import ActionProposal, ActionType, Outcome, Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH: str = "sca_grounding.db"
CALIBRATION_BIN_COUNT: int = 5  # Rapor için bin sayısı


# ---------------------------------------------------------------------------
# ORM Base & Models
# ---------------------------------------------------------------------------
class _Base(DeclarativeBase):
    pass


class _PredictionRow(_Base):
    """predictions tablosu ORM modeli."""

    __tablename__ = "predictions"

    prediction_id = Column(String(36), primary_key=True)
    statement = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    category = Column(String(128), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    context_block_ids = Column(Text, nullable=False, default="[]")  # JSON
    metadata_json = Column(Text, nullable=False, default="{}")  # JSON

    outcome = relationship(
        "_OutcomeRow",
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )


class _OutcomeRow(_Base):
    """outcomes tablosu ORM modeli."""

    __tablename__ = "outcomes"

    outcome_id = Column(String(36), primary_key=True)
    prediction_id = Column(String(36), ForeignKey("predictions.prediction_id"), nullable=False)
    action_type = Column(String(64), nullable=False)
    action_params = Column(Text, nullable=False, default="{}")  # JSON
    action_expected_outcome = Column(Text, nullable=False, default="")
    action_cost_estimate = Column(Float, nullable=False, default=0.0)
    action_justification = Column(Text, nullable=False, default="")
    actual_result = Column(Text, nullable=False, default="")
    match_score = Column(Float, nullable=False, default=0.0)
    match_reasoning = Column(Text, nullable=False, default="")
    timestamp = Column(DateTime, nullable=False)
    execution_time = Column(Float, nullable=False, default=0.0)
    cost_actual = Column(Float, nullable=False, default=0.0)
    error = Column(Text, nullable=True)

    prediction = relationship("_PredictionRow", back_populates="outcome")


# ---------------------------------------------------------------------------
# Dönüşüm Yardımcıları
# ---------------------------------------------------------------------------
def _prediction_to_row(pred: Prediction) -> _PredictionRow:
    """Prediction pydantic → ORM satırı."""
    return _PredictionRow(
        prediction_id=str(pred.prediction_id),
        statement=pred.statement,
        confidence=pred.confidence_at_prediction,
        category=pred.category,
        timestamp=pred.timestamp.replace(tzinfo=None),  # SQLite naive datetime
        context_block_ids=json.dumps(pred.context_block_ids),
        metadata_json=json.dumps(pred.metadata),
    )


def _row_to_prediction(row: _PredictionRow) -> Prediction:
    """ORM satırı → Prediction pydantic."""
    ts = row.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return Prediction(
        prediction_id=uuid.UUID(row.prediction_id),
        statement=row.statement,
        confidence_at_prediction=row.confidence,
        category=row.category,
        timestamp=ts,
        context_block_ids=json.loads(row.context_block_ids),
        metadata=json.loads(row.metadata_json),
    )


def _outcome_to_row(outcome: Outcome) -> _OutcomeRow:
    """Outcome pydantic → ORM satırı."""
    return _OutcomeRow(
        outcome_id=str(outcome.outcome_id),
        prediction_id=str(outcome.prediction_id),
        action_type=outcome.action_executed.action_type.value,
        action_params=json.dumps(outcome.action_executed.parameters),
        action_expected_outcome=outcome.action_executed.expected_outcome,
        action_cost_estimate=outcome.action_executed.cost_estimate,
        action_justification=outcome.action_executed.justification,
        actual_result=outcome.actual_result,
        match_score=outcome.match_score,
        match_reasoning=outcome.match_reasoning,
        timestamp=outcome.timestamp.replace(tzinfo=None),
        execution_time=outcome.execution_time_seconds,
        cost_actual=outcome.cost_actual,
        error=outcome.error,
    )


def _row_to_outcome(row: _OutcomeRow) -> Outcome:
    """ORM satırı → Outcome pydantic."""
    ts = row.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    proposal = ActionProposal(
        action_type=ActionType(row.action_type),
        parameters=json.loads(row.action_params),
        expected_outcome=row.action_expected_outcome,
        cost_estimate=row.action_cost_estimate,
        justification=row.action_justification,
    )
    return Outcome(
        outcome_id=uuid.UUID(row.outcome_id),
        prediction_id=uuid.UUID(row.prediction_id),
        action_executed=proposal,
        actual_result=row.actual_result,
        match_score=row.match_score,
        match_reasoning=row.match_reasoning,
        timestamp=ts,
        execution_time_seconds=row.execution_time,
        cost_actual=row.cost_actual,
        error=row.error,
    )


# ---------------------------------------------------------------------------
# GroundingLog
# ---------------------------------------------------------------------------
class GroundingLog:
    """AOGL için SQLite tabanlı kalıcı log.

    Prediction ve Outcome nesnelerini depolar, sorgular ve kalibrasyon
    için veri üretir.

    Args:
        db_path: SQLite veritabanı yolu. ":memory:" in-memory için.

    Example:
        >>> log = GroundingLog(db_path=":memory:")
        >>> log.add_prediction(pred)
        >>> log.add_outcome(outcome)
        >>> data = log.get_calibration_data()
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        if db_path == ":memory:":
            connection_string = "sqlite:///:memory:"
        else:
            connection_string = f"sqlite:///{db_path}"
        self._engine = create_engine(
            connection_string,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("GroundingLog başlatıldı: %s", db_path)

    # ------------------------------------------------------------------
    # Yazma
    # ------------------------------------------------------------------

    def add_prediction(self, prediction: Prediction) -> None:
        """Bir tahmin kaydeder.

        Args:
            prediction: Kaydedilecek Prediction nesnesi.

        Raises:
            ValueError: Aynı prediction_id zaten varsa.
        """
        with self._Session() as session:
            existing = session.get(_PredictionRow, str(prediction.prediction_id))
            if existing is not None:
                raise ValueError(
                    f"prediction_id {prediction.prediction_id} zaten mevcut."
                )
            row = _prediction_to_row(prediction)
            session.add(row)
            session.commit()
        logger.debug("Prediction kaydedildi: %s", prediction.prediction_id)

    def add_outcome(self, outcome: Outcome) -> None:
        """Bir sonuç kaydeder.

        Args:
            outcome: Kaydedilecek Outcome nesnesi.

        Raises:
            ValueError: prediction_id yoksa.
        """
        with self._Session() as session:
            pred_row = session.get(_PredictionRow, str(outcome.prediction_id))
            if pred_row is None:
                raise ValueError(
                    f"prediction_id {outcome.prediction_id} bulunamadı."
                )
            row = _outcome_to_row(outcome)
            session.merge(row)
            session.commit()
        logger.debug("Outcome kaydedildi: %s", outcome.outcome_id)

    # ------------------------------------------------------------------
    # Okuma
    # ------------------------------------------------------------------

    def get_prediction(self, prediction_id: uuid.UUID) -> Optional[Prediction]:
        """ID ile tahmin getirir.

        Args:
            prediction_id: Aranan tahmin UUID'si.

        Returns:
            Bulunan Prediction veya None.
        """
        with self._Session() as session:
            row = session.get(_PredictionRow, str(prediction_id))
            if row is None:
                return None
            return _row_to_prediction(row)

    def get_outcome(self, prediction_id: uuid.UUID) -> Optional[Outcome]:
        """Tahmin ID'sine göre sonuç getirir.

        Args:
            prediction_id: İlgili tahmin UUID'si.

        Returns:
            Bulunan Outcome veya None.
        """
        with self._Session() as session:
            pred_row = session.get(_PredictionRow, str(prediction_id))
            if pred_row is None or pred_row.outcome is None:
                return None
            return _row_to_outcome(pred_row.outcome)

    # ------------------------------------------------------------------
    # Sorgular
    # ------------------------------------------------------------------

    def query_by_category(
        self, category: str
    ) -> list[tuple[Prediction, Optional[Outcome]]]:
        """Belirli kategorideki tahmin-sonuç çiftlerini getirir.

        Args:
            category: Filtrelenecek kategori string.

        Returns:
            (Prediction, Outcome | None) tuple listesi.
        """
        with self._Session() as session:
            rows = (
                session.query(_PredictionRow)
                .filter(_PredictionRow.category == category.lower())
                .all()
            )
            result = []
            for row in rows:
                pred = _row_to_prediction(row)
                outcome = _row_to_outcome(row.outcome) if row.outcome else None
                result.append((pred, outcome))
        return result

    def query_by_time_range(
        self, start: datetime, end: datetime
    ) -> list[tuple[Prediction, Optional[Outcome]]]:
        """Belirli zaman aralığındaki kayıtları getirir.

        Args:
            start: Başlangıç zamanı (inclusive).
            end: Bitiş zamanı (inclusive).

        Returns:
            (Prediction, Outcome | None) tuple listesi.
        """
        start_naive = start.replace(tzinfo=None)
        end_naive = end.replace(tzinfo=None)
        with self._Session() as session:
            rows = (
                session.query(_PredictionRow)
                .filter(
                    _PredictionRow.timestamp >= start_naive,
                    _PredictionRow.timestamp <= end_naive,
                )
                .all()
            )
            result = []
            for row in rows:
                pred = _row_to_prediction(row)
                outcome = _row_to_outcome(row.outcome) if row.outcome else None
                result.append((pred, outcome))
        return result

    def query_all(self) -> list[tuple[Prediction, Optional[Outcome]]]:
        """Tüm tahmin-sonuç çiftlerini getirir.

        Returns:
            (Prediction, Outcome | None) tuple listesi.
        """
        with self._Session() as session:
            rows = session.query(_PredictionRow).all()
            result = []
            for row in rows:
                pred = _row_to_prediction(row)
                outcome = _row_to_outcome(row.outcome) if row.outcome else None
                result.append((pred, outcome))
        return result

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_to_json(self, path: str | Path) -> None:
        """Tüm kayıtları JSON dosyasına aktarır.

        Args:
            path: Çıktı dosyası yolu.
        """
        path = Path(path)
        records = self.query_all()
        data: list[dict[str, Any]] = []
        for pred, outcome in records:
            entry: dict[str, Any] = {"prediction": pred.model_dump(mode="json")}
            if outcome:
                entry["outcome"] = outcome.model_dump(mode="json")
            else:
                entry["outcome"] = None
            data.append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        logger.info("GroundingLog %s dosyasına aktarıldı (%d kayıt).", path, len(data))

    def import_from_json(self, path: str | Path) -> None:
        """JSON dosyasından kayıtları içe aktarır.

        Mevcut kayıtların üzerine merge eder (duplicate_id'ler atlanır).

        Args:
            path: İçe aktarılacak JSON dosyası.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        imported_preds = 0
        imported_outcomes = 0

        for entry in data:
            pred_data = entry.get("prediction")
            if not pred_data:
                continue
            try:
                pred = Prediction.model_validate(pred_data)
                try:
                    self.add_prediction(pred)
                    imported_preds += 1
                except ValueError:
                    pass  # Zaten var

                outcome_data = entry.get("outcome")
                if outcome_data:
                    outcome = Outcome.model_validate(outcome_data)
                    try:
                        self.add_outcome(outcome)
                        imported_outcomes += 1
                    except (ValueError, Exception):
                        pass
            except Exception as exc:
                logger.warning("Kayıt import edilemedi: %s", exc)

        logger.info(
            "JSON import tamamlandı: %d prediction, %d outcome.",
            imported_preds, imported_outcomes,
        )

    # ------------------------------------------------------------------
    # Kalibrasyon
    # ------------------------------------------------------------------

    def get_calibration_data(
        self, category: Optional[str] = None
    ) -> dict[str, list[tuple[float, float]]]:
        """Kalibrasyon için (confidence, match_score) çiftlerini döndürür.

        Sadece hem prediction hem outcome olan kayıtları dahil eder.

        Args:
            category: Filtrelenecek kategori. None ise tüm kategoriler ayrı ayrı.

        Returns:
            {category: [(confidence, match_score), ...]} dict.
            Her giriş bir (tahmin_anı_güven, gerçekleşen_skor) çiftidir.
        """
        records = self.query_all()
        result: dict[str, list[tuple[float, float]]] = {}

        for pred, outcome in records:
            if outcome is None:
                continue
            if category is not None and pred.category != category.lower():
                continue
            cat = pred.category
            if cat not in result:
                result[cat] = []
            result[cat].append((pred.confidence_at_prediction, outcome.match_score))

        return result

    # ------------------------------------------------------------------
    # Temizlik
    # ------------------------------------------------------------------

    def prune_old(self, days: int) -> int:
        """Belirtilen günden daha eski kayıtları siler.

        Args:
            days: Kaç günden eski kayıtlar silinsin.

        Returns:
            Silinen prediction sayısı.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._Session() as session:
            stmt = delete(_PredictionRow).where(
                _PredictionRow.timestamp < cutoff
            )
            result = session.execute(stmt)
            session.commit()
            count = result.rowcount
        logger.info("prune_old(%d gün): %d kayıt silindi.", days, count)
        return count

    def close(self) -> None:
        """Veritabanı bağlantısını kapatır."""
        self._engine.dispose()
        logger.info("GroundingLog kapatıldı.")

    def __len__(self) -> int:
        """Toplam prediction sayısını döndürür."""
        with self._Session() as session:
            return session.query(_PredictionRow).count()
