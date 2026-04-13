"""
sca/augmentation.py — PromptAugmenter

Augments LLM system prompts with relevant learned insights and
selects few-shot examples from past episodes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sca.insight import Insight, InsightRepository, InsightType

if TYPE_CHECKING:
    from sca.episode import EpisodeStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_INSIGHTS_PER_SECTION: int = 3
LEARNED_SECTION_HEADER: str = "\n\n--- LEARNED FROM EXPERIENCE ---"
LEARNED_SECTION_FOOTER: str = "--- END LEARNED ---"


# ---------------------------------------------------------------------------
# PromptAugmenter
# ---------------------------------------------------------------------------
class PromptAugmenter:
    """Augments system prompts with relevant learned insights.

    Retrieves applicable insights from the repository and injects them
    into the system prompt as structured sections, enabling in-context
    learning from previous sessions.

    Args:
        insight_repository: InsightRepository to query.
        episode_store: Optional EpisodeStore for few-shot example selection.
        max_insights_per_section: Max insights to include per section (default: 3).

    Example:
        >>> augmenter = PromptAugmenter(repo, episode_store=store)
        >>> augmented = augmenter.augment_system_prompt(base_prompt, context)
    """

    def __init__(
        self,
        insight_repository: InsightRepository,
        episode_store: Optional["EpisodeStore"] = None,
        max_insights_per_section: int = DEFAULT_MAX_INSIGHTS_PER_SECTION,
    ) -> None:
        self.insight_repository = insight_repository
        self.episode_store = episode_store
        self.max_insights_per_section = max_insights_per_section
        logger.debug("PromptAugmenter initialized.")

    def augment_system_prompt(
        self, base_system_prompt: str, current_context: dict
    ) -> str:
        """Augment a system prompt with relevant learned insights.

        Queries the repository for insights matching the current context
        and appends them as structured sections. If no relevant insights
        exist, returns the base prompt unchanged.

        Args:
            base_system_prompt: Original system prompt.
            current_context: Dict with optional keys:
                - domain (str): Current task domain.
                - category (str): Current prediction category.
                - keywords (list[str]): Relevant keywords.

        Returns:
            Augmented system prompt, or base prompt if no insights match.
        """
        domain = current_context.get("domain")
        category = current_context.get("category")
        keywords = current_context.get("keywords", [])

        relevant = self.insight_repository.query_by_applicability(
            domain=domain,
            category=category,
            keywords=keywords if isinstance(keywords, list) else [],
        )

        if not relevant:
            return base_system_prompt

        # Partition by type
        biases = [i for i in relevant if i.type == InsightType.BIAS_PATTERN][
            : self.max_insights_per_section
        ]
        strategies = [
            i for i in relevant if i.type == InsightType.SUCCESSFUL_STRATEGY
        ][: self.max_insights_per_section]
        failures = [i for i in relevant if i.type == InsightType.FAILURE_MODE][
            : self.max_insights_per_section
        ]
        facts = [i for i in relevant if i.type == InsightType.DOMAIN_KNOWLEDGE][
            : self.max_insights_per_section
        ]

        sections: list[str] = []

        if biases:
            lines = ["CAUTIONS (known biases to avoid):"]
            for ins in biases:
                lines.append(f"- {ins.content}")
            sections.append("\n".join(lines))

        if strategies:
            lines = ["STRATEGIES (approaches that have worked):"]
            for ins in strategies:
                lines.append(f"- {ins.content}")
            sections.append("\n".join(lines))

        if failures:
            lines = ["KNOWN FAILURE MODES (avoid these patterns):"]
            for ins in failures:
                lines.append(f"- {ins.content}")
            sections.append("\n".join(lines))

        if facts:
            lines = ["KNOWN FACTS (verified domain knowledge):"]
            for ins in facts:
                lines.append(f"- {ins.content}")
            sections.append("\n".join(lines))

        if not sections:
            return base_system_prompt

        learned_block = (
            LEARNED_SECTION_HEADER
            + "\n"
            + "\n\n".join(sections)
            + "\n"
            + LEARNED_SECTION_FOOTER
        )

        augmented = base_system_prompt + learned_block
        logger.info(
            "Augmented system prompt with %d insights (%d biases, %d strategies, "
            "%d failures, %d facts).",
            len(biases) + len(strategies) + len(failures) + len(facts),
            len(biases), len(strategies), len(failures), len(facts),
        )
        return augmented

    def select_few_shot_examples(
        self,
        task_description: str,
        n_examples: int = 2,
    ) -> list[dict[str, str]]:
        """Select few-shot examples from past episodes.

        Retrieves similar past tasks, mixing successful and failed examples
        for balanced in-context learning.

        Args:
            task_description: Description of the current task.
            n_examples: Number of examples to return (default: 2).

        Returns:
            List of chat-format message dicts:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            Returns empty list if no suitable examples are found.
        """
        if self.episode_store is None:
            return []

        try:
            episodes = self.episode_store.list_episodes(limit=50)
        except Exception as exc:
            logger.warning("Could not retrieve episodes for few-shot: %s", exc)
            return []

        if not episodes:
            return []

        # Simple relevance: keyword overlap in initial_prompt
        task_words = set(task_description.lower().split())
        scored: list[tuple[int, Any]] = []
        for ep in episodes:
            if ep.initial_prompt:
                ep_words = set(ep.initial_prompt.lower().split())
                overlap = len(task_words & ep_words)
                scored.append((overlap, ep))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [ep for _, ep in scored[:n_examples]]

        messages: list[dict[str, str]] = []
        for ep in selected:
            if ep.initial_prompt:
                messages.append({"role": "user", "content": ep.initial_prompt})
                # Use a brief summary as the assistant response
                domain_info = ep.domain or "general"
                n_preds = len(ep.prediction_ids)
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"[Past session in domain '{domain_info}' with "
                            f"{n_preds} predictions logged.]"
                        ),
                    }
                )

        return messages
