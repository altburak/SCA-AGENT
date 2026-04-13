"""
PSM — Prompt Formatteri
========================
ContextBlock'ları LLM için formatlar. ContextManager'dan bağımsız tasarlanmıştır;
farklı formatlama stratejileri (default, xml, minimal) destekler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sca.context import ContextManager

from sca.context import ContextBlock, Provenance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format sabitleri
# ---------------------------------------------------------------------------
FORMAT_DEFAULT = "default"
FORMAT_XML = "xml"
FORMAT_MINIMAL = "minimal"
FormatMode = Literal["default", "xml", "minimal"]

BLOCK_SEPARATOR = "\n\n"
CONFIDENCE_PRECISION = 2

# ---------------------------------------------------------------------------
# System prompt şablonları
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_EXPLANATION: str = """\
You are operating within a Provenance-Stratified Memory (PSM) system. \
Every piece of information in your context is tagged with a provenance label \
that indicates its origin. You MUST treat these labels as authoritative metadata \
and reason accordingly.

Provenance labels and their meanings:

[USER] — Direct input from the human user. Treat as the primary source of \
intent and requirements.

[EXTERNAL_TOOL] — Output from an external tool, API, database query, or web \
search. This information has been retrieved from an external system and is \
considered verified at retrieval time. Prefer this over SELF_GENERATED claims \
when there is a conflict.

[SELF_GENERATED] — YOUR OWN previous outputs from an earlier turn. CRITICAL: \
This content was produced by you and may contain reasoning errors, \
hallucinations, or outdated inferences. Do NOT treat it as ground truth. \
Validate it against EXTERNAL_TOOL or USER blocks before relying on it.

[DERIVED_INFERENCE] — Content derived by combining or inferring from other \
blocks. This content is one step removed from primary sources and carries \
compounded uncertainty. Always check the confidence score.

[KNOWLEDGE_BASE] — Static, curated knowledge retrieved from a trusted \
knowledge base. Generally reliable, but may be outdated.

[SYSTEM] — Operational directives and configuration instructions. Follow these.

Confidence scores (conf: X.XX) range from 0.0 (highly uncertain) to 1.0 \
(fully verified). When conf < 0.5, treat the block as tentative and flag \
uncertainty in your response. When SELF_GENERATED blocks conflict with \
EXTERNAL_TOOL blocks, the EXTERNAL_TOOL block takes precedence.

Always reason transparently about provenance when answering. If your answer \
depends on a SELF_GENERATED block with low confidence, say so explicitly.\
"""

MINIMAL_SYSTEM_EXPLANATION: str = """\
You are an AI assistant. Answer the user's questions based on the provided context.\
"""

XML_SYSTEM_EXPLANATION: str = """\
You are operating within a Provenance-Stratified Memory (PSM) system. \
Context blocks are structured as XML elements with provenance and confidence \
attributes. Use these attributes to reason about information reliability. \
SELF_GENERATED blocks are your own previous outputs — treat them critically. \
EXTERNAL_TOOL blocks represent verified external data — prefer them in conflicts.\
"""


# ---------------------------------------------------------------------------
# PromptFormatter
# ---------------------------------------------------------------------------
class PromptFormatter:
    """ContextBlock'ları farklı format modlarında LLM'e uygun metne çevirir.

    Üç mod desteklenir:
    - ``"default"``: İnsan okunabilir ``[PROVENANCE — conf: X.XX]`` başlıkları.
    - ``"xml"``: XML etiketleriyle yapılandırılmış format.
    - ``"minimal"``: Etiket yok, sadece içerik (kontrol grubu için).

    Args:
        mode: Kullanılacak format modu (default: "default").

    Example:
        >>> from sca.context import ContextBlock, Provenance
        >>> fmt = PromptFormatter()
        >>> block = ContextBlock("Hava sıcak.", Provenance.EXTERNAL_TOOL)
        >>> print(fmt.format_block(block))
        [EXTERNAL_TOOL — conf: 1.00]
        Hava sıcak.
    """

    def __init__(self, mode: FormatMode = FORMAT_DEFAULT) -> None:
        self._validate_mode(mode)
        self.mode: FormatMode = mode
        logger.debug("PromptFormatter mode=%s ile başlatıldı.", mode)

    # ------------------------------------------------------------------
    # Validasyon
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mode(mode: str) -> None:
        valid = {FORMAT_DEFAULT, FORMAT_XML, FORMAT_MINIMAL}
        if mode not in valid:
            raise ValueError(f"Geçersiz format modu: {mode!r}. Geçerli: {valid}")

    # ------------------------------------------------------------------
    # Tek blok formatlama
    # ------------------------------------------------------------------

    def format_block(self, block: ContextBlock) -> str:
        """Tek bir ContextBlock'u seçili moda göre formatlar.

        Args:
            block: Formatlanacak ContextBlock.

        Returns:
            Formatlanmış metin.

        Raises:
            TypeError: block bir ContextBlock değilse.
        """
        if not isinstance(block, ContextBlock):
            raise TypeError(f"Beklenen ContextBlock, aldık: {type(block)}")

        if self.mode == FORMAT_DEFAULT:
            return self._format_default(block)
        if self.mode == FORMAT_XML:
            return self._format_xml(block)
        if self.mode == FORMAT_MINIMAL:
            return self._format_minimal(block)
        # Bu noktaya asla gelinmemeli; __init__ validasyon yapar.
        raise RuntimeError(f"Bilinmeyen mod: {self.mode!r}")  # pragma: no cover

    def _format_default(self, block: ContextBlock) -> str:
        conf = round(block.confidence, CONFIDENCE_PRECISION)
        header = f"[{block.provenance.value} — conf: {conf:.2f}]"
        lines = [header, block.content]
        if block.derived_from:
            lines.append(f"(derived from block IDs: {block.derived_from})")
        if block.is_low_confidence():
            lines.append("⚠ LOW CONFIDENCE — treat as tentative.")
        return "\n".join(lines)

    def _format_xml(self, block: ContextBlock) -> str:
        conf = round(block.confidence, CONFIDENCE_PRECISION)
        attrs = (
            f'provenance="{block.provenance.value}" '
            f'conf="{conf:.2f}" '
            f'id="{block.block_id}"'
        )
        if block.derived_from:
            attrs += f' derived_from="{block.derived_from}"'
        xml = (
            f"<context_block {attrs}>\n"
            f"  {block.content}\n"
            f"</context_block>"
        )
        return xml

    def _format_minimal(self, block: ContextBlock) -> str:
        return block.content

    # ------------------------------------------------------------------
    # Tüm blokları formatlama
    # ------------------------------------------------------------------

    def format_all(self, manager: "ContextManager") -> str:
        """ContextManager'daki tüm blokları tek bir metne birleştirir.

        Args:
            manager: Blokları içeren ContextManager.

        Returns:
            Tüm formatlanmış blokların birleşimi.
        """
        if not manager.blocks:
            return ""
        parts = [self.format_block(b) for b in manager.blocks]
        return BLOCK_SEPARATOR.join(parts)

    # ------------------------------------------------------------------
    # System prompt erişicileri
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Seçili moda uygun sistem açıklamasını döndürür.

        Returns:
            System prompt metni.
        """
        if self.mode == FORMAT_DEFAULT:
            return DEFAULT_SYSTEM_EXPLANATION
        if self.mode == FORMAT_XML:
            return XML_SYSTEM_EXPLANATION
        if self.mode == FORMAT_MINIMAL:
            return MINIMAL_SYSTEM_EXPLANATION
        raise RuntimeError(f"Bilinmeyen mod: {self.mode!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Format değiştirme
    # ------------------------------------------------------------------

    def set_mode(self, mode: FormatMode) -> None:
        """Çalışma zamanında format modunu değiştirir.

        Args:
            mode: Yeni format modu.

        Raises:
            ValueError: Geçersiz mod verilirse.
        """
        self._validate_mode(mode)
        self.mode = mode
        logger.debug("PromptFormatter modu %s olarak değiştirildi.", mode)
