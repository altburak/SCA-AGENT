"""
SCA — Stratified Cognitive Agent
=================================
PSM (Provenance-Stratified Memory) + CSM (Confidence Scoring) +
AOGL (Action-Outcome Grounding Loop) + CED (Cross-Episode Distillation)
"""

# PSM — Module 1
from sca.context import ContextBlock, ContextManager, Provenance
from sca.formatter import PromptFormatter

# CSM — Module 2
from sca.confidence import (
    CompositeConfidenceScorer,
    ConfidenceScore,
    ProvenancePenaltyCalculator,
    SelfConsistencyScorer,
    VerifierScorer,
)
from sca.calibration import CalibrationLearner, ConfidenceCalibrator
from sca.llm import LLMClient
from sca.similarity import SemanticSimilarity

# AOGL — Module 3
from sca.prediction import ActionProposal, ActionType, Outcome, Prediction
from sca.actions import (
    ActionExecutor,
    BaseTool,
    CodeExecutorTool,
    FileReaderTool,
    SearchTool,
    UserAskerTool,
    WebFetcherTool,
    create_default_executor,
)
from sca.grounding import GroundingLog
from sca.evaluation import OutcomeEvaluator
from sca.aogl import AOGLController
from sca.agent import StratifiedAgent, create_default_agent

# CED — Module 4
from sca.episode import Episode, EpisodeStore
from sca.insight import Insight, InsightRepository, InsightType
from sca.extraction import InsightExtractor
from sca.augmentation import PromptAugmenter
from sca.ced import DistillationOrchestrator, LoRADistillationHook

__all__ = [
    # PSM
    "ContextBlock", "ContextManager", "Provenance", "PromptFormatter",
    # CSM
    "CompositeConfidenceScorer", "ConfidenceScore", "ProvenancePenaltyCalculator",
    "SelfConsistencyScorer", "VerifierScorer", "CalibrationLearner",
    "ConfidenceCalibrator", "LLMClient", "SemanticSimilarity",
    # AOGL
    "ActionProposal", "ActionType", "Outcome", "Prediction",
    "ActionExecutor", "BaseTool", "CodeExecutorTool", "FileReaderTool",
    "SearchTool", "UserAskerTool", "WebFetcherTool", "create_default_executor",
    "GroundingLog", "OutcomeEvaluator", "AOGLController",
    "StratifiedAgent", "create_default_agent",
    # CED
    "Episode", "EpisodeStore",
    "Insight", "InsightRepository", "InsightType",
    "InsightExtractor",
    "PromptAugmenter",
    "DistillationOrchestrator", "LoRADistillationHook",
]

__version__ = "0.4.0"
