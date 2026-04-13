"""
sca/extraction.py — InsightExtractor

Extracts Insight objects from a completed Episode by analyzing
prediction-outcome pairs with an LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sca.episode import Episode
from sca.insight import Insight, InsightType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIGH_MATCH_THRESHOLD: float = 0.8
LOW_MATCH_THRESHOLD: float = 0.4
DEFAULT_INSIGHT_CONFIDENCE: float = 0.6
MAX_PREDICTIONS_FOR_EXTRACTION: int = 30


# ---------------------------------------------------------------------------
# InsightExtractor
# ---------------------------------------------------------------------------
class InsightExtractor:
    """Extracts Insight objects from an Episode using LLM analysis.

    Analyzes prediction-outcome pairs to identify bias patterns,
    successful strategies, failure modes, and domain knowledge.

    Args:
        llm_client: LLMClient instance (large model recommended).
        grounding_log: GroundingLog to retrieve prediction-outcome data.

    Example:
        >>> extractor = InsightExtractor(llm_client=client, grounding_log=log)
        >>> insights = asyncio.run(extractor.extract_all(episode))
    """

    def __init__(self, llm_client: Any, grounding_log: Any) -> None:
        self.llm_client = llm_client
        self.grounding_log = grounding_log
        logger.debug("InsightExtractor initialized.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_pred_outcome_pairs(
        self, episode: Episode
    ) -> list[dict[str, Any]]:
        """Collect prediction-outcome pairs for this episode.

        Args:
            episode: Source episode.

        Returns:
            List of dicts with prediction_id, statement, confidence,
            category, match_score fields.
        """
        pairs: list[dict[str, Any]] = []
        for pred_id in episode.prediction_ids[:MAX_PREDICTIONS_FOR_EXTRACTION]:
            try:
                pred = self.grounding_log.get_prediction(pred_id)
                outcome = self.grounding_log.get_outcome_for_prediction(pred_id)
                if pred is not None:
                    pairs.append(
                        {
                            "prediction_id": str(pred_id),
                            "statement": pred.statement,
                            "confidence": pred.confidence_at_prediction,
                            "category": pred.category,
                            "match_score": outcome.match_score if outcome else None,
                            "actual_result": outcome.actual_result if outcome else "",
                        }
                    )
            except Exception as exc:
                logger.debug("Could not load prediction %s: %s", pred_id, exc)
        return pairs

    def _format_pred_list(self, pairs: list[dict[str, Any]]) -> str:
        """Format prediction-outcome pairs for LLM prompts."""
        lines = []
        for i, p in enumerate(pairs, 1):
            ms = f"{p['match_score']:.2f}" if p["match_score"] is not None else "N/A"
            lines.append(
                f"{i}. [{p['prediction_id'][:8]}] "
                f"Category={p['category']} | "
                f"Confidence={p['confidence']:.2f} | "
                f"MatchScore={ms}\n"
                f"   Statement: {p['statement'][:120]}"
            )
        return "\n".join(lines)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Synchronous LLM call wrapper."""
        return self.llm_client.chat(messages)

    async def _async_llm(self, prompt: str) -> str:
        """Async LLM call via executor."""
        messages = [{"role": "user", "content": prompt}]
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._call_llm(messages)),
                timeout=45.0,
            )
        except Exception as exc:
            logger.warning("LLM call failed in extractor: %s", exc)
            return ""

    def _parse_json_response(self, raw: str, top_key: str) -> list[dict]:
        """Parse LLM JSON response, return list under top_key.

        Returns empty list on any parse failure.
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.strip().rstrip("`").strip()

        # Try direct parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                items = data.get(top_key, [])
                return items if isinstance(items, list) else []
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                items = data.get(top_key, [])
                return items if isinstance(items, list) else []
            except (json.JSONDecodeError, ValueError):
                pass

        logger.debug("JSON parse failed for top_key=%r. Raw (first 200): %r", top_key, raw[:200])
        return []

    def _make_insight(
        self,
        insight_type: InsightType,
        content: str,
        evidence: list[str],
        confidence: float,
        episode: Episode,
        domain: str | None = None,
        category: str | None = None,
        keywords: list[str] | None = None,
    ) -> Insight:
        """Construct an Insight from extracted fields."""
        return Insight(
            type=insight_type,
            content=content.strip(),
            evidence=evidence,
            confidence=float(max(0.0, min(1.0, confidence))),
            applicability={
                "domain": domain or episode.domain,
                "category": category,
                "keywords": keywords or [],
            },
            source_episode_ids=[episode.episode_id],
            creation_time=datetime.now(timezone.utc),
            last_validated=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    async def extract_bias_patterns(self, episode: Episode) -> list[Insight]:
        """Extract systematic bias patterns from the episode.

        Identifies cases where the agent's confidence consistently
        mismatches its actual accuracy.

        Args:
            episode: Episode to analyze.

        Returns:
            List of BIAS_PATTERN Insights (empty on parse failure).
        """
        pairs = self._gather_pred_outcome_pairs(episode)
        if not pairs:
            return []

        prompt = f"""You are analyzing an AI agent's reasoning session.

Here are {len(pairs)} predictions the agent made, with their confidence
scores and actual outcomes (match_score):

{self._format_pred_list(pairs)}

Identify SYSTEMATIC BIASES. A bias is a pattern where the
agent's confidence doesn't match its actual accuracy in a
particular category or type of prediction.

Respond in JSON:
{{
  "biases": [
    {{
      "pattern": "brief description",
      "evidence": ["prediction IDs supporting this"],
      "severity": 0.0,
      "category": "applicable category or 'general'"
    }}
  ]
}}

If no clear biases exist, return {{"biases": []}}."""

        raw = await self._async_llm(prompt)
        items = self._parse_json_response(raw, "biases")
        insights: list[Insight] = []

        for item in items:
            pattern = item.get("pattern", "").strip()
            if not pattern:
                continue
            confidence = float(item.get("severity", DEFAULT_INSIGHT_CONFIDENCE))
            confidence = max(0.1, min(0.9, confidence))
            try:
                ins = self._make_insight(
                    insight_type=InsightType.BIAS_PATTERN,
                    content=pattern,
                    evidence=item.get("evidence", []),
                    confidence=confidence,
                    episode=episode,
                    category=item.get("category"),
                )
                insights.append(ins)
            except Exception as exc:
                logger.debug("Skipping bias insight due to error: %s", exc)

        logger.info("Extracted %d bias patterns from episode %s.", len(insights), episode.episode_id)
        return insights

    async def extract_successful_strategies(self, episode: Episode) -> list[Insight]:
        """Extract strategies from high-scoring predictions.

        Analyzes predictions with match_score > HIGH_MATCH_THRESHOLD
        to find common successful approaches.

        Args:
            episode: Episode to analyze.

        Returns:
            List of SUCCESSFUL_STRATEGY Insights.
        """
        pairs = self._gather_pred_outcome_pairs(episode)
        high_pairs = [p for p in pairs if p["match_score"] is not None and p["match_score"] > HIGH_MATCH_THRESHOLD]

        if not high_pairs:
            return []

        prompt = f"""You are analyzing an AI agent's successful predictions.

Here are {len(high_pairs)} high-accuracy predictions (match_score > {HIGH_MATCH_THRESHOLD}):

{self._format_pred_list(high_pairs)}

Identify the COMMON SUCCESSFUL STRATEGIES. A strategy is an approach
or reasoning pattern that contributed to accurate predictions.

Respond in JSON:
{{
  "strategies": [
    {{
      "strategy": "brief description of what worked",
      "evidence": ["prediction IDs supporting this"],
      "confidence": 0.0,
      "keywords": ["relevant", "keywords"]
    }}
  ]
}}

If no clear patterns emerge, return {{"strategies": []}}."""

        raw = await self._async_llm(prompt)
        items = self._parse_json_response(raw, "strategies")
        insights: list[Insight] = []

        for item in items:
            strategy = item.get("strategy", "").strip()
            if not strategy:
                continue
            try:
                ins = self._make_insight(
                    insight_type=InsightType.SUCCESSFUL_STRATEGY,
                    content=strategy,
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", DEFAULT_INSIGHT_CONFIDENCE)),
                    episode=episode,
                    keywords=item.get("keywords", []),
                )
                insights.append(ins)
            except Exception as exc:
                logger.debug("Skipping strategy insight: %s", exc)

        logger.info("Extracted %d successful strategies.", len(insights))
        return insights

    async def extract_failure_modes(self, episode: Episode) -> list[Insight]:
        """Extract common failure patterns from low-scoring predictions.

        Analyzes predictions with match_score < LOW_MATCH_THRESHOLD
        to identify recurring failure modes.

        Args:
            episode: Episode to analyze.

        Returns:
            List of FAILURE_MODE Insights.
        """
        pairs = self._gather_pred_outcome_pairs(episode)
        low_pairs = [p for p in pairs if p["match_score"] is not None and p["match_score"] < LOW_MATCH_THRESHOLD]

        if not low_pairs:
            return []

        prompt = f"""You are analyzing an AI agent's failed predictions.

Here are {len(low_pairs)} low-accuracy predictions (match_score < {LOW_MATCH_THRESHOLD}):

{self._format_pred_list(low_pairs)}

Identify COMMON FAILURE MODES. A failure mode is a pattern of reasoning
or assumption that reliably leads to wrong predictions.

Respond in JSON:
{{
  "failures": [
    {{
      "mode": "brief description of what went wrong",
      "evidence": ["prediction IDs"],
      "confidence": 0.0,
      "category": "affected category or 'general'"
    }}
  ]
}}

If no clear patterns, return {{"failures": []}}."""

        raw = await self._async_llm(prompt)
        items = self._parse_json_response(raw, "failures")
        insights: list[Insight] = []

        for item in items:
            mode = item.get("mode", "").strip()
            if not mode:
                continue
            try:
                ins = self._make_insight(
                    insight_type=InsightType.FAILURE_MODE,
                    content=mode,
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", DEFAULT_INSIGHT_CONFIDENCE)),
                    episode=episode,
                    category=item.get("category"),
                )
                insights.append(ins)
            except Exception as exc:
                logger.debug("Skipping failure insight: %s", exc)

        logger.info("Extracted %d failure modes.", len(insights))
        return insights

    async def extract_domain_knowledge(self, episode: Episode) -> list[Insight]:
        """Extract domain-specific facts learned during the episode.

        Looks at high-confidence, high-accuracy predictions and external
        tool results to identify domain knowledge.

        Args:
            episode: Episode to analyze.

        Returns:
            List of DOMAIN_KNOWLEDGE Insights.
        """
        pairs = self._gather_pred_outcome_pairs(episode)
        # High confidence AND high match = likely domain facts
        strong_pairs = [
            p for p in pairs
            if p["match_score"] is not None
            and p["match_score"] > HIGH_MATCH_THRESHOLD
            and p["confidence"] > 0.7
        ]

        if not strong_pairs:
            return []

        prompt = f"""You are analyzing what domain-specific knowledge an AI agent demonstrated.

Here are {len(strong_pairs)} high-confidence, accurate predictions from the session:

{self._format_pred_list(strong_pairs)}

Session domain: {episode.domain or "unknown"}

Identify DOMAIN-SPECIFIC FACTS that the agent correctly knew.
These are facts specific to a domain that would be useful to remember.

Respond in JSON:
{{
  "facts": [
    {{
      "fact": "a specific domain fact that was demonstrated",
      "evidence": ["prediction IDs"],
      "confidence": 0.0,
      "domain": "the specific domain this applies to"
    }}
  ]
}}

If no specific domain facts are identifiable, return {{"facts": []}}."""

        raw = await self._async_llm(prompt)
        items = self._parse_json_response(raw, "facts")
        insights: list[Insight] = []

        for item in items:
            fact = item.get("fact", "").strip()
            if not fact:
                continue
            try:
                ins = self._make_insight(
                    insight_type=InsightType.DOMAIN_KNOWLEDGE,
                    content=fact,
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", DEFAULT_INSIGHT_CONFIDENCE)),
                    episode=episode,
                    domain=item.get("domain") or episode.domain,
                )
                insights.append(ins)
            except Exception as exc:
                logger.debug("Skipping domain knowledge insight: %s", exc)

        logger.info("Extracted %d domain knowledge insights.", len(insights))
        return insights

    async def extract_all(self, episode: Episode) -> list[Insight]:
        """Run all four extraction methods in parallel.

        Args:
            episode: Episode to analyze.

        Returns:
            Combined list of all extracted Insights.
        """
        results = await asyncio.gather(
            self.extract_bias_patterns(episode),
            self.extract_successful_strategies(episode),
            self.extract_failure_modes(episode),
            self.extract_domain_knowledge(episode),
            return_exceptions=True,
        )

        all_insights: list[Insight] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Extraction method raised exception: %s", result)
            elif isinstance(result, list):
                all_insights.extend(result)

        logger.info(
            "Total insights extracted from episode %s: %d",
            episode.episode_id, len(all_insights)
        )
        return all_insights
