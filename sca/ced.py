"""
sca/ced.py — Cross-Episode Distillation (CED)

DistillationOrchestrator: Coordinates episode storage, insight extraction,
and prompt augmentation across session boundaries.

LoRADistillationHook: Placeholder for future LoRA adapter training.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from sca.augmentation import PromptAugmenter
from sca.episode import Episode, EpisodeStore
from sca.extraction import InsightExtractor
from sca.insight import Insight, InsightRepository, InsightType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MERGE_SIMILARITY_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# DistillationOrchestrator
# ---------------------------------------------------------------------------
class DistillationOrchestrator:
    """Orchestrates cross-episode knowledge distillation.

    On session end: saves the episode, extracts insights, and merges
    similar ones. On session start: loads relevant insights and produces
    an augmented system prompt with few-shot examples.

    Args:
        episode_store: EpisodeStore for session persistence.
        insight_extractor: InsightExtractor for mining lessons.
        insight_repository: InsightRepository for knowledge storage.
        prompt_augmenter: PromptAugmenter for in-context injection.
        merge_similarity_threshold: Cosine similarity above which two
            insights are considered duplicates (default: 0.85).

    Example:
        >>> orch = DistillationOrchestrator(store, extractor, repo, augmenter)
        >>> stats = asyncio.run(orch.on_episode_end(episode))
        >>> context = orch.on_episode_start("Help me debug Python code", "coding")
    """

    def __init__(
        self,
        episode_store: EpisodeStore,
        insight_extractor: InsightExtractor,
        insight_repository: InsightRepository,
        prompt_augmenter: PromptAugmenter,
        merge_similarity_threshold: float = DEFAULT_MERGE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.episode_store = episode_store
        self.insight_extractor = insight_extractor
        self.insight_repository = insight_repository
        self.prompt_augmenter = prompt_augmenter
        self.merge_similarity_threshold = merge_similarity_threshold
        logger.info("DistillationOrchestrator initialized.")

    def __del__(self) -> None:
        """Best-effort cleanup of underlying stores."""
        try:
            self.episode_store.close()
        except Exception:
            pass
        try:
            self.insight_repository.close()
        except Exception:
            pass

    async def on_episode_end(self, episode: Episode) -> dict:
        """Process a completed episode: save, extract insights, and merge.

        Steps:
        1. Save episode to EpisodeStore.
        2. Extract all insights via InsightExtractor.
        3. Add each insight to InsightRepository.
        4. Merge similar insights.

        Args:
            episode: Completed Episode to process.

        Returns:
            Dict with keys:
            - insights_extracted (int): Number of new insights found.
            - insights_merged (int): Number of insights merged/deleted.
        """
        # Step 1: Save episode
        self.episode_store.save_episode(episode)
        logger.info("Episode %s saved.", episode.episode_id)

        # Step 2: Extract insights
        try:
            insights = await self.insight_extractor.extract_all(episode)
        except Exception as exc:
            logger.warning("Insight extraction failed: %s", exc)
            insights = []

        # Step 3: Store each insight
        for insight in insights:
            try:
                self.insight_repository.add_insight(insight)
            except Exception as exc:
                logger.warning("Failed to store insight: %s", exc)

        n_extracted = len(insights)

        # Step 4: Merge similar
        try:
            n_merged = self.insight_repository.merge_similar(
                similarity_threshold=self.merge_similarity_threshold
            )
        except Exception as exc:
            logger.warning("Insight merge failed: %s", exc)
            n_merged = 0

        result = {"insights_extracted": n_extracted, "insights_merged": n_merged}
        logger.info("on_episode_end complete: %s", result)
        return result

    def on_episode_start(
        self,
        initial_prompt: str,
        domain_hint: Optional[str] = None,
    ) -> dict:
        """Prepare context for a new session.

        Loads relevant insights based on the initial prompt and domain,
        produces an augmented system prompt, and selects few-shot examples.

        Args:
            initial_prompt: The opening request for this session.
            domain_hint: Optional domain tag ("coding", "research", etc.).

        Returns:
            Dict with keys:
            - augmented_system_prompt (str): System prompt with learned context.
            - few_shot_examples (list[dict]): Chat-format few-shot examples.
            - applicable_insight_ids (list[str]): UUIDs of applied insights.
        """
        # Build context for applicability matching
        keywords = [w for w in initial_prompt.lower().split() if len(w) > 3][:10]
        current_context = {
            "domain": domain_hint,
            "category": None,
            "keywords": keywords,
        }

        # Get base system prompt
        from sca.formatter import PromptFormatter
        formatter = PromptFormatter()
        base_prompt = formatter.get_system_prompt()

        # Augment
        augmented = self.prompt_augmenter.augment_system_prompt(
            base_system_prompt=base_prompt,
            current_context=current_context,
        )

        # Few-shot examples
        few_shot = self.prompt_augmenter.select_few_shot_examples(
            task_description=initial_prompt, n_examples=2
        )

        # Find applicable insight IDs (for tracking)
        relevant = self.insight_repository.query_by_applicability(
            domain=domain_hint,
            category=None,
            keywords=keywords,
        )
        applicable_ids = [str(ins.insight_id) for ins in relevant[:10]]

        result = {
            "augmented_system_prompt": augmented,
            "few_shot_examples": few_shot,
            "applicable_insight_ids": applicable_ids,
        }
        logger.info(
            "on_episode_start: %d insights applicable, %d few-shot examples.",
            len(applicable_ids), len(few_shot)
        )
        return result

    async def get_statistics(self) -> dict:
        """Return summary statistics about episodes and insights.

        Returns:
            Dict with keys:
            - total_episodes (int)
            - total_insights (int)
            - insights_by_type (dict[str, int])
            - avg_insights_per_episode (float)
            - most_used_insights (list[dict]): Top 3 by usage_count.
        """
        episodes = self.episode_store.list_episodes(limit=10000)
        total_episodes = len(episodes)

        by_type: dict[str, int] = {}
        for ins_type in InsightType:
            count = len(self.insight_repository.query_by_type(ins_type))
            by_type[ins_type.value] = count

        total_insights = sum(by_type.values())
        avg = total_insights / total_episodes if total_episodes > 0 else 0.0

        top_insights = self.insight_repository.get_top_k(k=3, criterion="usage_count")
        most_used = [
            {
                "insight_id": str(ins.insight_id),
                "type": ins.type.value,
                "content": ins.content[:80],
                "usage_count": ins.usage_count,
                "success_rate": round(ins.success_rate, 3),
            }
            for ins in top_insights
        ]

        return {
            "total_episodes": total_episodes,
            "total_insights": total_insights,
            "insights_by_type": by_type,
            "avg_insights_per_episode": round(avg, 2),
            "most_used_insights": most_used,
        }


# ---------------------------------------------------------------------------
# LoRADistillationHook (placeholder)
# ---------------------------------------------------------------------------
class LoRADistillationHook(ABC):
    """Abstract hook for future LoRA adapter distillation.

    Future work: distill insights into LoRA adapter.
    Requires local model (Llama/Mistral) + peft library. Not
    implemented in Phase 1.
    """

    @abstractmethod
    def train_lora_from_insights(self, insights: list[Insight]) -> Path:
        """Train a LoRA adapter from accumulated insights.

        Args:
            insights: List of Insight objects to distill into the adapter.

        Returns:
            Path to the trained LoRA adapter directory.

        Raises:
            NotImplementedError: Always — this is a Phase 2 feature.
        """
        raise NotImplementedError(
            "Future work: distill insights into LoRA adapter. "
            "Requires local model (Llama/Mistral) + peft library. "
            "Not implemented in Phase 1."
        )
