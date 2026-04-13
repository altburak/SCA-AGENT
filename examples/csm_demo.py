"""
CSM Demo — Gerçek Groq çağrılarıyla 4 senaryo
===============================================
Çalıştırma: GROQ_API_KEY ortam değişkeni tanımlı olmalı.
  python examples/csm_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

from sca.calibration import ConfidenceCalibrator
from sca.confidence import (
    CompositeConfidenceScorer,
    ProvenancePenaltyCalculator,
    SelfConsistencyScorer,
    VerifierScorer,
)
from sca.context import ContextBlock, ContextManager, Provenance
from sca.llm import LLMClient

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def color_score(score: float) -> str:
    if score >= 0.75:
        color = GREEN
        label = "HIGH"
    elif score >= 0.50:
        color = YELLOW
        label = "MEDIUM"
    else:
        color = RED
        label = "LOW"
    return f"{color}{BOLD}{label} ({score:.3f}){RESET}"


def print_result(scenario: str, result) -> None:
    print(f"\n{'=' * 60}")
    print(f"{BLUE}{BOLD}Senaryo: {scenario}{RESET}")
    print(f"{'=' * 60}")
    print(f"  Nihai Skor    : {color_score(result.final_score)}")
    print(f"  Self-Consist  : {result.components['self_consistency']:.3f}")
    print(f"  Verifier      : {result.components['verifier']:.3f}")
    print(f"  Provenance    : {result.components['provenance']:.3f}")
    print(f"  Açıklama      : {result.reasoning}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def build_scorer(
    main_client: LLMClient,
    verifier_client: LLMClient,
    context_manager: ContextManager,
    n_samples: int = 2,
) -> CompositeConfidenceScorer:
    sc_scorer = SelfConsistencyScorer(
        llm_client=main_client,
        n_samples=n_samples,
        temperature_range=(0.5, 0.9),
    )
    v_scorer = VerifierScorer(verifier_llm_client=verifier_client)
    prov_calc = ProvenancePenaltyCalculator(context_manager=context_manager)

    return CompositeConfidenceScorer(
        self_consistency_scorer=sc_scorer,
        verifier_scorer=v_scorer,
        provenance_calculator=prov_calc,
    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
async def scenario_a(main_client, verifier_client):
    """A) Yüksek eminlik: doğru, iyi-kaynaklı cevap."""
    manager = ContextManager()
    ids = [
        manager.add(ContextBlock(
            "Python was created by Guido van Rossum and first released in 1991.",
            Provenance.EXTERNAL_TOOL,
            confidence=0.99,
        )),
        manager.add(ContextBlock(
            "Python 3.0 was released in 2008 and is not backward compatible with Python 2.",
            Provenance.KNOWLEDGE_BASE,
            confidence=0.95,
        )),
    ]

    question = "Who created Python and when was it first released?"
    response = (
        "Python was created by Guido van Rossum and first released in 1991. "
        "Python 3, released in 2008, brought significant improvements but broke backward compatibility."
    )

    scorer = build_scorer(main_client, verifier_client, manager)
    return await scorer.score(question, response, manager, ids)


async def scenario_b(main_client, verifier_client):
    """B) Düşük eminlik: uydurulmuş, SELF_GENERATED tabanlı."""
    manager = ContextManager()
    ids = [
        manager.add(ContextBlock(
            "I believe the Eiffel Tower was built in 1756 by Napoleon Bonaparte as a war monument.",
            Provenance.SELF_GENERATED,
            confidence=0.3,
        )),
        manager.add(ContextBlock(
            "The tower is approximately 500 meters tall and made of copper.",
            Provenance.DERIVED_INFERENCE,
            confidence=0.25,
        )),
    ]

    question = "When was the Eiffel Tower built and by whom?"
    response = (
        "The Eiffel Tower was built in 1756 by Napoleon Bonaparte as a war monument. "
        "It stands at 500 meters tall and is made of copper."
    )

    scorer = build_scorer(main_client, verifier_client, manager)
    return await scorer.score(question, response, manager, ids)


async def scenario_c(main_client, verifier_client):
    """C) Orta eminlik: kısmen doğru cevap."""
    manager = ContextManager()
    ids = [
        manager.add(ContextBlock(
            "The speed of light in vacuum is approximately 299,792 km/s.",
            Provenance.KNOWLEDGE_BASE,
            confidence=0.98,
        )),
        manager.add(ContextBlock(
            "I think light might also sometimes travel faster under certain exotic conditions.",
            Provenance.SELF_GENERATED,
            confidence=0.4,
        )),
    ]

    question = "What is the speed of light?"
    response = (
        "The speed of light in vacuum is about 300,000 km/s. "
        "However, under some exotic quantum conditions, it may exceed this value."
    )

    scorer = build_scorer(main_client, verifier_client, manager)
    return await scorer.score(question, response, manager, ids)


async def scenario_d(main_client, verifier_client):
    """D) Kalibrasyon dışı: verifier yüksek, self-consistency düşük."""
    manager = ContextManager()
    ids = [
        manager.add(ContextBlock(
            "Machine learning is a subset of artificial intelligence.",
            Provenance.KNOWLEDGE_BASE,
            confidence=0.9,
        )),
    ]

    question = "What is machine learning?"
    # Deliberately vague response to create inconsistency between verifier and self-consistency
    response = (
        "Machine learning involves algorithms. It can be supervised or not. "
        "Sometimes it uses data. Neural networks may or may not be involved."
    )

    # Custom weights: verifier heavily weighted
    sc_scorer = SelfConsistencyScorer(main_client, n_samples=3, temperature_range=(0.3, 0.95))
    v_scorer = VerifierScorer(verifier_client)
    prov_calc = ProvenancePenaltyCalculator(manager)
    scorer = CompositeConfidenceScorer(
        sc_scorer, v_scorer, prov_calc,
        weights={"self_consistency": 0.2, "verifier": 0.6, "provenance": 0.2},
    )
    return await scorer.score(question, response, manager, ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(f"{RED}GROQ_API_KEY ortam değişkeni tanımlı değil!{RESET}")
        print("export GROQ_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"\n{BOLD}CSM Demo — Confidence Scoring Module{RESET}")
    print("Groq Llama modelleri ile 4 senaryo test ediliyor...\n")

   
    main_client = LLMClient(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=512,
        api_key=api_key,
    )


    verifier_client = LLMClient(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=512,
        api_key=api_key,
    )

    calibrator = ConfidenceCalibrator()

    scenarios = [
        ("A — Yüksek Eminlik (doğru, iyi kaynaklı)", scenario_a),
        ("B — Düşük Eminlik (uydurulmuş, SELF_GENERATED)", scenario_b),
        ("C — Orta Eminlik (kısmen doğru)", scenario_c),
        ("D — Kalibrasyon Dışı (verifier vs self-consistency çatışması)", scenario_d),
    ]

    for name, fn in scenarios:
        try:
            result = await fn(main_client, verifier_client)
            calibrated = calibrator.apply(result.final_score)
            print_result(name, result)
            print(f"  Kalibre Skor  : {calibrated:.3f} (identity — henüz kalibrasyon verisi yok)")
        except Exception as exc:
            print(f"\n{RED}Senaryo '{name}' başarısız: {exc}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())