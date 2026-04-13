"""
sca/insight.py — Insight Model and Repository

InsightType: Categorizes the kind of learned knowledge.
Insight: A single piece of distilled knowledge from past episodes.
InsightRepository: SQLAlchemy-backed storage with query, merge, and prune.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INSIGHT_DB_PATH: str = "sca_insights.db"
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0


# ---------------------------------------------------------------------------
# InsightType
# ---------------------------------------------------------------------------
class InsightType(str, Enum):
    """Categories of distilled knowledge.

    Attributes:
        BIAS_PATTERN: The agent is over/under-confident in a specific area.
        SUCCESSFUL_STRATEGY: An approach that reliably works for a task type.
        FAILURE_MODE: A pattern that reliably causes prediction failures.
        DOMAIN_KNOWLEDGE: A learned fact specific to a domain.
    """

    BIAS_PATTERN = "bias_pattern"
    SUCCESSFUL_STRATEGY = "successful_strategy"
    FAILURE_MODE = "failure_mode"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


# ---------------------------------------------------------------------------
# Pydantic Model
# ---------------------------------------------------------------------------
class Insight(BaseModel):
    """A single piece of distilled cross-episode knowledge.

    Args:
        insight_id: Auto-generated UUID.
        type: InsightType classification.
        content: Natural language description of the insight.
        evidence: Supporting episode or prediction IDs (as strings).
        confidence: Confidence in this insight itself (0-1).
        applicability: Conditions under which this insight applies.
            Keys: domain (Optional[str]), category (Optional[str]),
            keywords (list[str]).
        source_episode_ids: UUIDs of source episodes.
        creation_time: When this insight was created.
        last_validated: When this insight was last confirmed.
        usage_count: How many times this insight was applied.
        success_count: How many applications were successful.
    """

    insight_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: InsightType
    content: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5)
    applicability: dict = Field(default_factory=lambda: {"domain": None, "category": None, "keywords": []})
    source_episode_ids: list[uuid.UUID] = Field(default_factory=list)
    creation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_validated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = Field(default=0)
    success_count: int = Field(default=0)

    model_config = {"frozen": False}

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Content must not be empty."""
        if not v or not v.strip():
            raise ValueError("Insight content cannot be empty.")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Confidence must be in [0, 1]."""
        if not (CONFIDENCE_MIN <= v <= CONFIDENCE_MAX):
            raise ValueError(f"confidence {v!r} must be in [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}].")
        return v

    @property
    def success_rate(self) -> float:
        """Ratio of successful applications. Returns 0 if never used."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------
class _InsightBase(DeclarativeBase):
    pass


class _InsightRow(_InsightBase):
    """insights table ORM model."""

    __tablename__ = "insights"

    insight_id = Column(String(36), primary_key=True)
    type = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=False, default="[]")
    confidence = Column(Float, nullable=False, default=0.5)
    applicability_json = Column(Text, nullable=False, default="{}")
    source_episode_ids_json = Column(Text, nullable=False, default="[]")
    creation_time = Column(DateTime, nullable=False)
    last_validated = Column(DateTime, nullable=False)
    usage_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _insight_to_row(ins: Insight) -> _InsightRow:
    return _InsightRow(
        insight_id=str(ins.insight_id),
        type=ins.type.value,
        content=ins.content,
        evidence_json=json.dumps(ins.evidence),
        confidence=ins.confidence,
        applicability_json=json.dumps(ins.applicability),
        source_episode_ids_json=json.dumps([str(e) for e in ins.source_episode_ids]),
        creation_time=ins.creation_time,
        last_validated=ins.last_validated,
        usage_count=ins.usage_count,
        success_count=ins.success_count,
    )


def _row_to_insight(row: _InsightRow) -> Insight:
    return Insight(
        insight_id=uuid.UUID(row.insight_id),
        type=InsightType(row.type),
        content=row.content,
        evidence=json.loads(row.evidence_json or "[]"),
        confidence=row.confidence,
        applicability=json.loads(row.applicability_json or "{}"),
        source_episode_ids=[uuid.UUID(e) for e in json.loads(row.source_episode_ids_json or "[]")],
        creation_time=row.creation_time,
        last_validated=row.last_validated,
        usage_count=row.usage_count,
        success_count=row.success_count,
    )


# ---------------------------------------------------------------------------
# InsightRepository
# ---------------------------------------------------------------------------
class InsightRepository:
    """SQLAlchemy-backed storage for Insight objects.

    Supports rich querying, pruning stale insights, and merging
    semantically similar ones.

    Args:
        db_path: SQLite file path (default: sca_insights.db).

    Example:
        >>> repo = InsightRepository(db_path=":memory:")
        >>> ins = Insight(type=InsightType.BIAS_PATTERN, content="Overconfident on math")
        >>> repo.add_insight(ins)
        >>> results = repo.query_by_type(InsightType.BIAS_PATTERN)
    """

    def __init__(self, db_path: str = DEFAULT_INSIGHT_DB_PATH) -> None:
        self.db_path = db_path
        connection_str = (
            f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite:///:memory:"
        )
        self._engine = create_engine(connection_str, echo=False)
        _InsightBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.debug("InsightRepository initialized: db=%s", db_path)

    def add_insight(self, insight: Insight) -> None:
        """Persist an Insight.

        Args:
            insight: Insight to add.
        """
        with self._Session() as session:
            existing = session.get(_InsightRow, str(insight.insight_id))
            row = _insight_to_row(insight)
            if existing:
                session.delete(existing)
                session.flush()
            session.add(row)
            session.commit()
        logger.debug("Insight added: %s", insight.insight_id)

    def get_insight(self, insight_id: uuid.UUID) -> Optional[Insight]:
        """Retrieve an Insight by UUID.

        Args:
            insight_id: UUID to look up.

        Returns:
            Insight if found, None otherwise.
        """
        with self._Session() as session:
            row = session.get(_InsightRow, str(insight_id))
            return _row_to_insight(row) if row else None

    def query_by_type(self, type: InsightType) -> list[Insight]:
        """Return all insights of a given type.

        Args:
            type: InsightType to filter by.

        Returns:
            List of matching Insights.
        """
        with self._Session() as session:
            rows = (
                session.query(_InsightRow)
                .filter(_InsightRow.type == type.value)
                .all()
            )
            return [_row_to_insight(r) for r in rows]

    def query_by_applicability(
        self,
        domain: Optional[str],
        category: Optional[str],
        keywords: list[str],
    ) -> list[Insight]:
        """Search insights by applicability filters.

        Scores each insight by how well it matches domain, category, and keywords,
        then returns them sorted by relevance (descending).

        Args:
            domain: Domain to match (None = no filter).
            category: Category to match (None = no filter).
            keywords: Keywords for overlap scoring.

        Returns:
            List of Insights sorted by relevance score descending.
        """
        with self._Session() as session:
            rows = session.query(_InsightRow).all()

        scored: list[tuple[int, Insight]] = []
        for row in rows:
            ins = _row_to_insight(row)
            score = 0
            app = ins.applicability

            # Domain match
            if domain and app.get("domain") == domain:
                score += 3

            # Category match
            if category and app.get("category") == category:
                score += 2

            # Keyword overlap
            ins_kw = [k.lower() for k in app.get("keywords", [])]
            for kw in keywords:
                if kw.lower() in ins_kw:
                    score += 1

            # Include if any match, or if insight has no applicability restrictions
            if score > 0 or (not app.get("domain") and not app.get("category") and not app.get("keywords")):
                scored.append((score, ins))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ins for _, ins in scored]

    def get_top_k(self, k: int, criterion: str = "confidence") -> list[Insight]:
        """Return top-k insights by a given criterion.

        Args:
            k: Number of insights to return.
            criterion: One of "confidence", "usage_count", "success_rate", "recency".

        Returns:
            Top k Insights.
        """
        with self._Session() as session:
            rows = session.query(_InsightRow).all()

        insights = [_row_to_insight(r) for r in rows]

        if criterion == "confidence":
            insights.sort(key=lambda i: i.confidence, reverse=True)
        elif criterion == "usage_count":
            insights.sort(key=lambda i: i.usage_count, reverse=True)
        elif criterion == "success_rate":
            insights.sort(key=lambda i: i.success_rate, reverse=True)
        elif criterion == "recency":
            insights.sort(key=lambda i: i.creation_time, reverse=True)
        else:
            logger.warning("Unknown criterion: %r, using confidence.", criterion)
            insights.sort(key=lambda i: i.confidence, reverse=True)

        return insights[:k]

    def record_usage(self, insight_id: uuid.UUID, was_successful: bool) -> None:
        """Increment usage and optionally success counters.

        Args:
            insight_id: UUID of the insight to update.
            was_successful: Whether applying this insight was successful.
        """
        with self._Session() as session:
            row = session.get(_InsightRow, str(insight_id))
            if row is None:
                logger.warning("record_usage: insight %s not found.", insight_id)
                return
            row.usage_count += 1
            if was_successful:
                row.success_count += 1
            session.commit()

    def prune_stale(self, age_days: int, min_success_rate: float) -> int:
        """Delete old insights that have low success rates.

        Args:
            age_days: Insights older than this (in days) are candidates for pruning.
            min_success_rate: Insights with success_rate below this are pruned.

        Returns:
            Number of insights deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)
        deleted = 0

        with self._Session() as session:
            rows = (
                session.query(_InsightRow)
                .filter(_InsightRow.creation_time < cutoff)
                .all()
            )
            for row in rows:
                ins = _row_to_insight(row)
                if ins.usage_count > 0 and ins.success_rate < min_success_rate:
                    session.delete(row)
                    deleted += 1
            session.commit()

        logger.info("Pruned %d stale insights.", deleted)
        return deleted

    def merge_similar(self, similarity_threshold: float = 0.85) -> int:
        """Merge semantically similar insights using SemanticSimilarity.

        For each pair of similar insights, the older one is deleted and its
        evidence is merged into the newer one.

        Args:
            similarity_threshold: Cosine similarity threshold for merging.

        Returns:
            Number of insights deleted (merged away).
        """
        from sca.similarity import SemanticSimilarity

        with self._Session() as session:
            rows = session.query(_InsightRow).all()

        insights = [_row_to_insight(r) for r in rows]
        if len(insights) < 2:
            return 0

        sim = SemanticSimilarity()
        to_delete: set[str] = set()
        updates: list[Insight] = []

        for i in range(len(insights)):
            if str(insights[i].insight_id) in to_delete:
                continue
            for j in range(i + 1, len(insights)):
                if str(insights[j].insight_id) in to_delete:
                    continue
                try:
                    score = sim.cosine_similarity(insights[i].content, insights[j].content)
                except Exception as exc:
                    logger.warning("Similarity computation failed: %s", exc)
                    continue

                if score >= similarity_threshold:
                    # Keep newer, delete older
                    if insights[i].creation_time >= insights[j].creation_time:
                        keeper, gone = insights[i], insights[j]
                    else:
                        keeper, gone = insights[j], insights[i]

                    # Merge evidence
                    merged_evidence = list(set(keeper.evidence + gone.evidence + [str(gone.insight_id)]))
                    keeper.evidence = merged_evidence
                    keeper.usage_count += gone.usage_count
                    keeper.success_count += gone.success_count

                    to_delete.add(str(gone.insight_id))
                    updates.append(keeper)
                    logger.debug(
                        "Merging insight %s into %s (sim=%.3f)",
                        gone.insight_id, keeper.insight_id, score
                    )

        # Apply updates and deletions
        for ins in updates:
            self.add_insight(ins)

        with self._Session() as session:
            for iid in to_delete:
                row = session.get(_InsightRow, iid)
                if row:
                    session.delete(row)
            session.commit()

        logger.info("Merged %d similar insights.", len(to_delete))
        return len(to_delete)

    def close(self) -> None:
        """Dispose the database engine."""
        self._engine.dispose()
        logger.debug("InsightRepository closed.")

    def __enter__(self) -> "InsightRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass
