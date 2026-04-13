"""
PSM Demo — Provenance-Stratified Memory canlı demosunu gösterir.

Çalıştırma:
    python examples/psm_demo.py

Gereksinimler:
    - GROQ_API_KEY ortam değişkeni veya .env dosyası
    - pip install litellm python-dotenv
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Proje kökünü import yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from sca.context import ContextBlock, ContextManager, Provenance
from sca.llm import LLMClient

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

SEPARATOR = "=" * 70


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_response(label: str, response: str) -> None:
    print(f"\n[{label}]\n{response}\n")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def scenario_a_conflict(client: LLMClient) -> None:
    """Model önceki (yanlış) kendi çıktısını harici doğrulanmış bilgiyle karşılaştırır."""
    print_section("SENARYO A: Çelişki — SELF_GENERATED vs EXTERNAL_TOOL")
    print(
        "Senaryo: Model daha önce İstanbul'un Türkiye'nin başkenti olduğunu\n"
        "söylemiş (hatalı bir SELF_GENERATED çıktı). Ardından harici araç\n"
        "doğru cevabı getirdi. PSM bunu nasıl çözüyor?\n"
    )

    mgr = ContextManager()

    mgr.add(
        ContextBlock(
            content=(
                "Türkiye'nin başkenti İstanbul'dur. Nüfusu yaklaşık 15 milyon "
                "olup ülkenin en büyük ve en önemli şehridir."
            ),
            provenance=Provenance.SELF_GENERATED,
            confidence=0.6,
            metadata={"turn": 1, "note": "Bu çıktı hatalıydı."},
        )
    )


    mgr.add(
        ContextBlock(
            content=(
                "Wikipedia/Britannica: Türkiye'nin başkenti Ankara'dır. "
                "Ankara, 1923'te Türkiye Cumhuriyeti'nin ilanıyla başkent oldu. "
                "Nüfusu yaklaşık 5.7 milyondur."
            ),
            provenance=Provenance.EXTERNAL_TOOL,
            confidence=1.0,
            metadata={"tool": "web_search", "source": "wikipedia.org"},
        )
    )

    question = (
        "Türkiye'nin başkenti neresidir? "
        "Bağlamda çelişen bilgiler var — hangisine güvenmeli ve neden?"
    )

    response = client.chat_with_context(mgr, question, format_mode="default")
    print_response("LLM YANITI — Senaryo A", response)



# ---------------------------------------------------------------------------

def scenario_b_low_confidence(client: LLMClient) -> None:
    """Düşük confidence'lı DERIVED_INFERENCE bloğunun etkisini gösterir."""
    print_section("SENARYO B: Düşük Güven — DERIVED_INFERENCE Uyarısı")
    print(
        "Senaryo: Bir finans analizi görevi. Bir blok yüksek güvenle\n"
        "doğrulanmış piyasa verisini içerirken, başka bir blok bu veriden\n"
        "düşük güvenle türetilmiş bir tahmin içeriyor.\n"
    )

    mgr = ContextManager()


    user_bid = mgr.add(
        ContextBlock(
            content="Şirketimizin önümüzdeki çeyrek için gelir tahminini anlat.",
            provenance=Provenance.USER,
        )
    )

    market_bid = mgr.add(
        ContextBlock(
            content=(
                "Q3 2024 sektör raporu: Teknoloji sektöründe ortalama büyüme %12. "
                "Rakip firma X: %9 büyüme. Rakip firma Y: %14 büyüme. "
                "Kaynak: Bloomberg Terminal, 2024-10-01."
            ),
            provenance=Provenance.EXTERNAL_TOOL,
            confidence=0.95,
            metadata={"tool": "bloomberg_api", "date": "2024-10-01"},
        )
    )

    mgr.add(
        ContextBlock(
            content=(
                "Sektör ortalamasını ve şirketin geçmiş performansını baz alarak "
                "tahmini büyüme oranı %10-13 aralığında olabilir. "
                "Ancak makroekonomik belirsizlikler bu tahmini önemli ölçüde etkiler."
            ),
            provenance=Provenance.DERIVED_INFERENCE,
            confidence=0.35,
            derived_from=[market_bid],
            metadata={"method": "linear_extrapolation"},
        )
    )

    question = (
        "Bu bağlamı değerlendirerek şirket için Q4 gelir tahmini sun. "
        "Hangi bilgilere güvendiğini ve hangileri için ne kadar emin olduğunu açıkla."
    )

    response = client.chat_with_context(mgr, question, format_mode="default")
    print_response("LLM YANITI — Senaryo B", response)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def scenario_c_baseline_comparison(client: LLMClient) -> None:
    """Aynı içerik, provenance etiketli vs etiketsiz (minimal mod) karşılaştırması."""
    print_section("SENARYO C: Baseline — Provenance Var vs Yok")
    print(
        "Senaryo: Tıbbi bilgi senaryosu. Aynı içerik iki kez gönderiliyor:\n"
        "  (1) Provenance etiketleriyle (default mod)\n"
        "  (2) Etiket olmadan (minimal mod — kontrol grubu)\n"
        "Modelin yanıt kalitesinin nasıl değiştiğini gözlemleyin.\n"
    )

    # Ortak içerik
    blocks_config = [
        {
            "content": (
                "İlaç prospektüsü: Ibuprofen 400mg — günlük maksimum doz 1200mg. "
                "Böbrek hastalığı olanlarda kontrendikedir."
            ),
            "provenance": Provenance.KNOWLEDGE_BASE,
            "confidence": 0.99,
            "metadata": {"source": "ilaç_prospektüsü"},
        },
        {
            "content": (
                "Hasta: 65 yaşında, kronik böbrek hastalığı Evre 3, "
                "mevcut ilaçları: metformin 500mg, lisinopril 10mg."
            ),
            "provenance": Provenance.USER,
            "confidence": 1.0,
        },
        {
            "content": (
                "Bir önceki yanımda ibuprofenin bu hasta için uygun olabileceğini "
                "söyledim, ancak bunu doğrulamam gerekiyor."
            ),
            "provenance": Provenance.SELF_GENERATED,
            "confidence": 0.4,
        },
    ]

    question = (
        "Bu hasta için ibuprofen kullanımı uygun mudur? "
        "Gerekçeni açıkla."
    )


    print("── 1. PROVENANCE ETİKETLİ (default mod) ──")
    mgr_with = ContextManager()
    for cfg in blocks_config:
        mgr_with.add(
            ContextBlock(
                content=cfg["content"],
                provenance=cfg["provenance"],
                confidence=cfg.get("confidence", 1.0),
                metadata=cfg.get("metadata", {}),
            )
        )
    response_with = client.chat_with_context(mgr_with, question, format_mode="default")
    print_response("LLM YANITI — Provenance Etiketli", response_with)


    print("── 2. ETİKETSİZ (minimal mod — kontrol grubu) ──")
    mgr_without = ContextManager()
    for cfg in blocks_config:
        mgr_without.add(
            ContextBlock(
                content=cfg["content"],
                provenance=cfg["provenance"],
                confidence=cfg.get("confidence", 1.0),
            )
        )
    response_without = client.chat_with_context(
        mgr_without, question, format_mode="minimal", inject_system=False
    )
    print_response("LLM YANITI — Etiketsiz (Kontrol)", response_without)

    print(
        "\n[ANALİZ]\n"
        "• Etiketli yanıtta modelin SELF_GENERATED bloğunu şüpheyle ele aldığını\n"
        "  ve KNOWLEDGE_BASE + USER bloklarına öncelik verdiğini gözlemleyin.\n"
        "• Etiketsiz yanıtta model bilgileri eşit ağırlıkta değerlendirmiş olabilir.\n"
    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("  PSM (Provenance-Stratified Memory) — Canlı Demo")
    print("  Model: groq/llama-3.3-70b-versatile")
    print(SEPARATOR)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "\n[HATA] GROQ_API_KEY ortam değişkeni tanımlı değil.\n"
            "Lütfen proje kökünde bir .env dosyası oluşturun:\n"
            "  GROQ_API_KEY=gsk_...\n"
        )
        sys.exit(1)

    try:
        client = LLMClient()
    except Exception as exc:
        print(f"\n[HATA] LLMClient başlatılamadı: {exc}")
        sys.exit(1)

    scenario_a_conflict(client)
    scenario_b_low_confidence(client)
    scenario_c_baseline_comparison(client)

    print(f"\n{SEPARATOR}")
    print("  Demo tamamlandı.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
