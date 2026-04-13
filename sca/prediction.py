"""
sca/prediction.py — AOGL Veri Sınıfları

Prediction, ActionProposal ve Outcome pydantic modellerini tanımlar.
Bu sınıflar AOGL döngüsünün temel veri taşıyıcılarıdır.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0
MATCH_SCORE_MIN: float = 0.0
MATCH_SCORE_MAX: float = 1.0


# ---------------------------------------------------------------------------
# ActionType Enum
# ---------------------------------------------------------------------------
class ActionType(str, Enum):
    """Bir tahmini doğrulamak için kullanılabilecek eylem türleri.

    Attributes:
        READ_FILE: Yerel dosya okuma.
        WEB_FETCH: URL içeriği çekme.
        EXECUTE_CODE: Python kodu çalıştırma.
        TOOL_CALL: Harici araç çağrısı.
        USER_QUESTION: Kullanıcıya soru sorma.
        SEARCH: Web araması yapma.
        NO_ACTION: Otomatik doğrulama mümkün değil.
    """

    READ_FILE = "read_file"
    WEB_FETCH = "web_fetch"
    EXECUTE_CODE = "execute_code"
    TOOL_CALL = "tool_call"
    USER_QUESTION = "user_question"
    SEARCH = "search"
    NO_ACTION = "no_action"


# ---------------------------------------------------------------------------
# ActionProposal
# ---------------------------------------------------------------------------
class ActionProposal(BaseModel):
    """Bir tahmini doğrulamak için önerilen eylemin tanımı.

    Args:
        action_type: Eylem türü (ActionType enum).
        parameters: Eyleme özgü parametreler.
        expected_outcome: Beklenen sonucun metinsel açıklaması.
        cost_estimate: Yaklaşık maliyet (saniye veya token birimi).
        justification: Bu eylemin seçilme gerekçesi.

    Example:
        >>> proposal = ActionProposal(
        ...     action_type=ActionType.READ_FILE,
        ...     parameters={"path": "/tmp/test.txt"},
        ...     expected_outcome="Dosyanın ilk satırı 'hello' içeriyor",
        ...     cost_estimate=0.1,
        ...     justification="Dosya içeriğini okuyarak tahmini doğrulayabiliriz",
        ... )
    """

    action_type: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str = Field(default="")
    cost_estimate: float = Field(default=0.0, ge=0.0)
    justification: str = Field(default="")

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class Prediction(BaseModel):
    """Agent'in ürettiği bir tahmini temsil eder.

    Args:
        prediction_id: Benzersiz UUID (otomatik üretilir).
        statement: Tahminin metinsel ifadesi.
        confidence_at_prediction: Tahmin anındaki CSM skoru [0, 1].
        category: Tahmin kategorisi (esnek string).
        timestamp: Oluşturma zamanı (UTC).
        context_block_ids: PSM'deki dayanak blok ID'leri.
        proposed_action: Doğrulama için önerilen eylem (opsiyonel).
        metadata: Esnek ek bilgi.

    Raises:
        ValueError: confidence [0,1] dışındaysa, statement veya category boşsa.

    Example:
        >>> pred = Prediction(
        ...     statement="config.py dosyası /etc/app/ dizinindedir",
        ...     confidence_at_prediction=0.8,
        ...     category="file_location",
        ...     context_block_ids=[0, 1],
        ... )
    """

    prediction_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    statement: str
    confidence_at_prediction: float
    category: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context_block_ids: list[int] = Field(default_factory=list)
    proposed_action: Optional[ActionProposal] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @field_validator("confidence_at_prediction")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Confidence [0, 1] aralığında olmalı."""
        if not (CONFIDENCE_MIN <= v <= CONFIDENCE_MAX):
            raise ValueError(
                f"confidence_at_prediction {v!r} geçersiz; "
                f"[{CONFIDENCE_MIN}, {CONFIDENCE_MAX}] aralığında olmalı."
            )
        return v

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, v: str) -> str:
        """Statement boş olamaz."""
        if not v or not v.strip():
            raise ValueError("statement boş olamaz.")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Category boş olamaz."""
        if not v or not v.strip():
            raise ValueError("category boş olamaz.")
        return v.strip().lower()


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------
class Outcome(BaseModel):
    """Bir tahmini doğrulamak için yürütülen eylemin sonucu.

    Args:
        outcome_id: Benzersiz UUID (otomatik üretilir).
        prediction_id: İlgili tahmin UUID'si (foreign key).
        action_executed: Yürütülen eylem proposal'ı.
        actual_result: Eylemin gerçek sonucunun metin temsili.
        match_score: Tahmin ile sonuç örtüşme skoru [0, 1].
        match_reasoning: LLM'in match_score gerekçesi.
        timestamp: Oluşturma zamanı (UTC).
        execution_time_seconds: Eylemin yürütülme süresi.
        cost_actual: Gerçek maliyet (token veya saniye).
        error: Eylem başarısız olduysa hata mesajı.

    Example:
        >>> outcome = Outcome(
        ...     prediction_id=pred.prediction_id,
        ...     action_executed=proposal,
        ...     actual_result="Dosyanın ilk satırı: 'hello world'",
        ...     match_score=0.9,
        ...     match_reasoning="Tahmin ile gerçek sonuç büyük ölçüde örtüşüyor",
        ... )
    """

    outcome_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    prediction_id: uuid.UUID
    action_executed: ActionProposal
    actual_result: str = Field(default="")
    match_score: float = Field(default=0.0)
    match_reasoning: str = Field(default="")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_seconds: float = Field(default=0.0, ge=0.0)
    cost_actual: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None

    model_config = {"frozen": False}

    @field_validator("match_score")
    @classmethod
    def validate_match_score(cls, v: float) -> float:
        """match_score [0, 1] aralığında olmalı."""
        if not (MATCH_SCORE_MIN <= v <= MATCH_SCORE_MAX):
            raise ValueError(
                f"match_score {v!r} geçersiz; "
                f"[{MATCH_SCORE_MIN}, {MATCH_SCORE_MAX}] aralığında olmalı."
            )
        return v
