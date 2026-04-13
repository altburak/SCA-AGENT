"""
PSM (Provenance-Stratified Memory) — Temel Veri Yapıları
==========================================================
ContextBlock ve ContextManager sınıfları.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0
DEFAULT_CONFIDENCE: float = 1.0


# ---------------------------------------------------------------------------
# Provenance Enum
# ---------------------------------------------------------------------------
class Provenance(str, Enum):
    """Bir ContextBlock'un bilgi kaynağını tanımlar.

    Attributes:
        USER: Doğrudan kullanıcıdan gelen girdi.
        EXTERNAL_TOOL: Bir araç (API, veritabanı, arama motoru vb.) çıktısı.
        SELF_GENERATED: Modelin önceki turda ürettiği çıktı.
        DERIVED_INFERENCE: Diğer bloklardan çıkarım yapılarak türetilen bilgi.
        KNOWLEDGE_BASE: Statik, doğrulanmış bir bilgi tabanından gelen bilgi.
        SYSTEM: Sistem direktifleri ve konfigürasyon talimatları.
    """

    USER = "USER"
    EXTERNAL_TOOL = "EXTERNAL_TOOL"
    SELF_GENERATED = "SELF_GENERATED"
    DERIVED_INFERENCE = "DERIVED_INFERENCE"
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    SYSTEM = "SYSTEM"


# ---------------------------------------------------------------------------
# ContextBlock
# ---------------------------------------------------------------------------
@dataclass
class ContextBlock:
    """Provenance etiketli tek bir bağlam birimi.

    Her ContextBlock, bir bilgi parçasını kaynağıyla birlikte taşır.
    Bu sayede LLM, kendi geçmiş çıktılarını dışarıdan gelen
    doğrulanmış bilgilerden ayırt edebilir.

    Args:
        content: Bilginin metin içeriği.
        provenance: Bilginin kaynağını tanımlayan Provenance enum değeri.
        confidence: Eminlik skoru, 0.0 ile 1.0 arasında (default: 1.0).
        timestamp: Bloğun oluşturulma zamanı (default: şu an, UTC).
        derived_from: Bu bloğun türetildiği önceki blok ID'leri.
        block_id: Benzersiz kimlik (ContextManager tarafından atanır).
        metadata: Araç adı, kaynak URL vb. ek bilgiler.

    Raises:
        ValueError: confidence 0-1 aralığı dışındaysa.
        ValueError: content boş string ise.
    """

    content: str
    provenance: Provenance
    confidence: float = DEFAULT_CONFIDENCE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    derived_from: list[int] = field(default_factory=list)
    block_id: int = field(default=-1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("ContextBlock.content boş olamaz.")
        if not (CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX):
            raise ValueError(
                f"confidence {self.confidence!r} geçersiz; "
                f"[{CONFIDENCE_MIN}, {CONFIDENCE_MAX}] aralığında olmalı."
            )
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"provenance bir Provenance enum değeri olmalı, "
                f"aldık: {type(self.provenance)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Bloğu JSON-serileştirilebilir dict'e çevirir.

        Returns:
            Bloğun tüm alanlarını içeren dict.
        """
        return {
            "block_id": self.block_id,
            "content": self.content,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "derived_from": self.derived_from,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextBlock":
        """Dict'ten ContextBlock oluşturur.

        Args:
            data: to_dict() çıktısıyla uyumlu dict.

        Returns:
            Yeniden oluşturulmuş ContextBlock.

        Raises:
            KeyError: Zorunlu alan eksikse.
            ValueError: Geçersiz provenance değeri varsa.
        """
        return cls(
            content=data["content"],
            provenance=Provenance(data["provenance"]),
            confidence=data.get("confidence", DEFAULT_CONFIDENCE),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            derived_from=data.get("derived_from", []),
            block_id=data.get("block_id", -1),
            metadata=data.get("metadata", {}),
        )

    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        """Bloğun düşük güven skoruna sahip olup olmadığını döndürür.

        Args:
            threshold: Eşik değeri (default: 0.5).

        Returns:
            confidence < threshold ise True.
        """
        return self.confidence < threshold


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------
class ContextManager:
    """PSM'in merkezi bellek yöneticisi.

    ContextBlock'ları sıralı bir listede saklar, ekler, siler ve
    filtreler. Ayrıca blokları LiteLLM-uyumlu mesaj formatına veya
    düz prompt metnine dönüştürür.

    Example:
        >>> manager = ContextManager()
        >>> bid = manager.add(ContextBlock("Merhaba", Provenance.USER))
        >>> manager.get_block(bid).content
        'Merhaba'
    """

    def __init__(self) -> None:
        self._blocks: list[ContextBlock] = []
        self._next_id: int = 0
        logger.debug("ContextManager başlatıldı.")

    # ------------------------------------------------------------------
    # CRUD işlemleri
    # ------------------------------------------------------------------

    def add(self, block: ContextBlock) -> int:
        """Yeni bir blok ekler ve atanan block_id'yi döndürür.

        Args:
            block: Eklenecek ContextBlock.

        Returns:
            Atanan block_id (int).

        Raises:
            TypeError: block bir ContextBlock değilse.
        """
        if not isinstance(block, ContextBlock):
            raise TypeError(f"Beklenen ContextBlock, aldık: {type(block)}")
        block.block_id = self._next_id
        self._next_id += 1
        self._blocks.append(block)
        logger.debug("Blok eklendi: id=%d provenance=%s", block.block_id, block.provenance)
        return block.block_id

    def get_block(self, block_id: int) -> ContextBlock:
        """ID'ye göre blok döndürür.

        Args:
            block_id: Aranacak blok kimliği.

        Returns:
            Bulunan ContextBlock.

        Raises:
            KeyError: block_id mevcut değilse.
        """
        for block in self._blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(f"block_id={block_id} bulunamadı.")

    def get_by_provenance(self, provenance: Provenance) -> list[ContextBlock]:
        """Belirli provenance tipindeki tüm blokları filtreler.

        Args:
            provenance: Filtrelenecek Provenance değeri.

        Returns:
            Eşleşen ContextBlock listesi (boş olabilir).

        Raises:
            TypeError: provenance bir Provenance değilse.
        """
        if not isinstance(provenance, Provenance):
            raise TypeError(f"Beklenen Provenance enum, aldık: {type(provenance)}")
        result = [b for b in self._blocks if b.provenance == provenance]
        logger.debug("%s provenance için %d blok bulundu.", provenance, len(result))
        return result

    def remove(self, block_id: int) -> None:
        """Belirtilen ID'li bloğu siler.

        Args:
            block_id: Silinecek blok kimliği.

        Raises:
            KeyError: block_id mevcut değilse.
        """
        original_len = len(self._blocks)
        self._blocks = [b for b in self._blocks if b.block_id != block_id]
        if len(self._blocks) == original_len:
            raise KeyError(f"block_id={block_id} bulunamadı, silinemedi.")
        logger.debug("Blok silindi: id=%d", block_id)

    def clear(self) -> None:
        """Tüm blokları temizler. _next_id sıfırlanmaz (monoton artar)."""
        count = len(self._blocks)
        self._blocks.clear()
        logger.debug("%d blok temizlendi.", count)

    # ------------------------------------------------------------------
    # Sorgular
    # ------------------------------------------------------------------

    @property
    def blocks(self) -> list[ContextBlock]:
        """Tüm blokların kopyasını döndürür (salt okunur görünüm)."""
        return list(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __bool__(self) -> bool:
        return bool(self._blocks)

    # ------------------------------------------------------------------
    # Prompt / Mesaj dönüşümleri
    # ------------------------------------------------------------------

    def to_prompt(self) -> str:
        """Tüm blokları düz metin prompt'a dönüştürür.

        PromptFormatter.format_all() çağrısına kısayol.

        Returns:
            Formatlanmış prompt metni.
        """
        from sca.formatter import PromptFormatter

        formatter = PromptFormatter()
        return formatter.format_all(self)

    def to_messages(self, system_prompt: Optional[str] = None) -> list[dict[str, str]]:
        """Blokları LiteLLM chat format (list of dicts) olarak döndürür.

        Her blok ayrı bir 'user' mesajı olarak temsil edilir.
        system_prompt verilirse listeye ilk sıraya eklenir.

        Args:
            system_prompt: İsteğe bağlı sistem mesajı.

        Returns:
            [{"role": "...", "content": "..."}, ...] formatında liste.
        """
        from sca.formatter import PromptFormatter

        formatter = PromptFormatter()
        messages: list[dict[str, str]] = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        for block in self._blocks:
            if block.provenance == Provenance.SYSTEM:
                role = "system"
            elif block.provenance == Provenance.USER:
                role = "user"
            else:
                role = "user"  # LLM'in context'i okuyacağı yer

            messages.append({"role": role, "content": formatter.format_block(block)})

        logger.debug("to_messages(): %d mesaj oluşturuldu.", len(messages))
        return messages

    # ------------------------------------------------------------------
    # Kalıcılık
    # ------------------------------------------------------------------

    def save_to_json(self, path: str | Path) -> None:
        """Tüm blokları JSON dosyasına yazar.

        Args:
            path: Yazılacak dosya yolu.

        Raises:
            OSError: Dosya yazılamazsa.
        """
        path = Path(path)
        payload = {
            "version": "1.0",
            "next_id": self._next_id,
            "blocks": [b.to_dict() for b in self._blocks],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("ContextManager %s dosyasına kaydedildi (%d blok).", path, len(self._blocks))

    def load_from_json(self, path: str | Path) -> None:
        """JSON dosyasından blokları yükler. Mevcut blokları temizler.

        Args:
            path: Okunacak dosya yolu.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            json.JSONDecodeError: Dosya geçerli JSON değilse.
            KeyError: Beklenen alanlar eksikse.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._blocks.clear()
        self._next_id = payload.get("next_id", 0)
        for block_data in payload.get("blocks", []):
            self._blocks.append(ContextBlock.from_dict(block_data))

        logger.info("%s dosyasından %d blok yüklendi.", path, len(self._blocks))
