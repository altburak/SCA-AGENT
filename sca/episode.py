"""
sca/episode.py — Episode Data Model and Store

Episode: Represents a single agent session with its context, predictions,
and calibration state.
EpisodeStore: SQLAlchemy-backed persistence for episodes.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

if TYPE_CHECKING:
    from sca.grounding import GroundingLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EPISODE_DB_PATH: str = "sca_episodes.db"

# PII regex patterns
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
]


# ---------------------------------------------------------------------------
# Pydantic Model
# ---------------------------------------------------------------------------
class Episode(BaseModel):
    """Represents a single agent session.

    Args:
        episode_id: Auto-generated UUID.
        start_time: Session start time (UTC).
        end_time: Session end time (UTC).
        initial_prompt: The opening request for this session.
        context_block_ids: PSM block IDs added during the session.
        prediction_ids: AOGL prediction UUIDs produced during the session.
        calibration_snapshot: Calibration state at session end.
            Format: {category: {samples: N, raw_to_calibrated: {...}}}
        domain: Session domain hint ("coding", "research", "qa", "general").
        metadata: Arbitrary extra info.
        tags: Arbitrary string tags.
    """

    episode_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    initial_prompt: Optional[str] = None
    context_block_ids: list[int] = Field(default_factory=list)
    prediction_ids: list[uuid.UUID] = Field(default_factory=list)
    calibration_snapshot: dict[str, dict] = Field(default_factory=dict)
    domain: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------
class _EpisodeBase(DeclarativeBase):
    pass


class _EpisodeRow(_EpisodeBase):
    """episodes table ORM model."""

    __tablename__ = "episodes"

    episode_id = Column(String(36), primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    initial_prompt = Column(Text, nullable=True)
    context_block_ids_json = Column(Text, nullable=False, default="[]")
    prediction_ids_json = Column(Text, nullable=False, default="[]")
    calibration_snapshot_json = Column(Text, nullable=False, default="{}")
    domain = Column(String(128), nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")
    tags_json = Column(Text, nullable=False, default="[]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _episode_to_row(ep: Episode) -> _EpisodeRow:
    return _EpisodeRow(
        episode_id=str(ep.episode_id),
        start_time=ep.start_time,
        end_time=ep.end_time,
        initial_prompt=ep.initial_prompt,
        context_block_ids_json=json.dumps(ep.context_block_ids),
        prediction_ids_json=json.dumps([str(p) for p in ep.prediction_ids]),
        calibration_snapshot_json=json.dumps(ep.calibration_snapshot),
        domain=ep.domain,
        metadata_json=json.dumps(ep.metadata),
        tags_json=json.dumps(ep.tags),
    )


def _row_to_episode(row: _EpisodeRow) -> Episode:
    return Episode(
        episode_id=uuid.UUID(row.episode_id),
        start_time=row.start_time,
        end_time=row.end_time,
        initial_prompt=row.initial_prompt,
        context_block_ids=json.loads(row.context_block_ids_json or "[]"),
        prediction_ids=[uuid.UUID(p) for p in json.loads(row.prediction_ids_json or "[]")],
        calibration_snapshot=json.loads(row.calibration_snapshot_json or "{}"),
        domain=row.domain,
        metadata=json.loads(row.metadata_json or "{}"),
        tags=json.loads(row.tags_json or "[]"),
    )


# ---------------------------------------------------------------------------
# EpisodeStore
# ---------------------------------------------------------------------------
class EpisodeStore:
    """SQLAlchemy-backed persistence for Episode objects.

    Args:
        db_path: SQLite database file path.
        grounding_log: Optional GroundingLog reference for match_score queries.

    Example:
        >>> store = EpisodeStore(db_path=":memory:")
        >>> ep = Episode(domain="coding")
        >>> store.save_episode(ep)
        >>> loaded = store.load_episode(ep.episode_id)
    """

    def __init__(
        self,
        db_path: str = DEFAULT_EPISODE_DB_PATH,
        grounding_log: Optional["GroundingLog"] = None,
    ) -> None:
        self.db_path = db_path
        self.grounding_log = grounding_log
        connection_str = (
            f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite:///:memory:"
        )
        self._engine = create_engine(connection_str, echo=False)
        _EpisodeBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.debug("EpisodeStore initialized: db=%s", db_path)

    def save_episode(self, episode: Episode) -> None:
        """Persist an Episode to the database.

        Args:
            episode: Episode to save.
        """
        with self._Session() as session:
            existing = session.get(_EpisodeRow, str(episode.episode_id))
            row = _episode_to_row(episode)
            if existing:
                session.delete(existing)
                session.flush()
            session.add(row)
            session.commit()
        logger.info("Episode saved: %s", episode.episode_id)

    def load_episode(self, episode_id: uuid.UUID) -> Optional[Episode]:
        """Load an Episode by its UUID.

        Args:
            episode_id: UUID of the episode.

        Returns:
            Episode if found, None otherwise.
        """
        with self._Session() as session:
            row = session.get(_EpisodeRow, str(episode_id))
            if row is None:
                return None
            return _row_to_episode(row)

    def list_episodes(self, limit: int = 50, offset: int = 0) -> list[Episode]:
        """List episodes with pagination.

        Args:
            limit: Max number of episodes to return.
            offset: Number of episodes to skip.

        Returns:
            List of Episodes ordered by start_time descending.
        """
        with self._Session() as session:
            rows = (
                session.query(_EpisodeRow)
                .order_by(_EpisodeRow.start_time.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [_row_to_episode(r) for r in rows]

    def query_by_domain(self, domain: str) -> list[Episode]:
        """Return all episodes matching the given domain.

        Args:
            domain: Domain string to filter by.

        Returns:
            List of matching Episodes.
        """
        with self._Session() as session:
            rows = (
                session.query(_EpisodeRow)
                .filter(_EpisodeRow.domain == domain)
                .order_by(_EpisodeRow.start_time.desc())
                .all()
            )
            return [_row_to_episode(r) for r in rows]

    def query_by_outcome_quality(self, min_avg_match_score: float) -> list[Episode]:
        """Return episodes whose average outcome match_score meets the threshold.

        Requires a GroundingLog to compute per-episode match scores.
        If no grounding_log is attached, returns all episodes.

        Args:
            min_avg_match_score: Minimum average match_score (0-1).

        Returns:
            Filtered list of Episodes.
        """
        if self.grounding_log is None:
            logger.warning("No grounding_log attached; returning all episodes.")
            return self.list_episodes(limit=1000)

        all_episodes = self.list_episodes(limit=1000)
        result: list[Episode] = []

        for ep in all_episodes:
            if not ep.prediction_ids:
                continue
            scores: list[float] = []
            for pred_id in ep.prediction_ids:
                outcome = self.grounding_log.get_outcome_for_prediction(pred_id)
                if outcome is not None:
                    scores.append(outcome.match_score)
            if scores:
                avg = sum(scores) / len(scores)
                if avg >= min_avg_match_score:
                    result.append(ep)

        return result

    def anonymize_episode(self, episode: Episode) -> Episode:
        """Return a copy of the episode with PII removed from text fields.

        Applies regex-based scrubbing for email, phone, and SSN patterns.

        Args:
            episode: Episode to anonymize.

        Returns:
            New Episode with PII replaced by placeholder tokens.
        """
        def _scrub(text: Optional[str]) -> Optional[str]:
            if text is None:
                return None
            for pattern, replacement in _PII_PATTERNS:
                text = re.sub(pattern, replacement, text)
            return text

        return Episode(
            episode_id=episode.episode_id,
            start_time=episode.start_time,
            end_time=episode.end_time,
            initial_prompt=_scrub(episode.initial_prompt),
            context_block_ids=list(episode.context_block_ids),
            prediction_ids=list(episode.prediction_ids),
            calibration_snapshot=episode.calibration_snapshot.copy(),
            domain=episode.domain,
            metadata=episode.metadata.copy(),
            tags=list(episode.tags),
        )

    def close(self) -> None:
        """Dispose the database engine."""
        self._engine.dispose()
        logger.debug("EpisodeStore closed.")

    def __enter__(self) -> "EpisodeStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass
