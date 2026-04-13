"""
PromptFormatter için pytest test paketi.
"""

from __future__ import annotations

import pytest

from sca.context import ContextBlock, ContextManager, Provenance
from sca.formatter import (
    FORMAT_DEFAULT,
    FORMAT_MINIMAL,
    FORMAT_XML,
    DEFAULT_SYSTEM_EXPLANATION,
    MINIMAL_SYSTEM_EXPLANATION,
    XML_SYSTEM_EXPLANATION,
    PromptFormatter,
)


# ===========================================================================
# Yardımcı fabrika
# ===========================================================================


def make_block(
    content: str = "Test",
    provenance: Provenance = Provenance.USER,
    confidence: float = 1.0,
    derived_from: list[int] | None = None,
) -> ContextBlock:
    block = ContextBlock(
        content=content,
        provenance=provenance,
        confidence=confidence,
        derived_from=derived_from or [],
    )
    block.block_id = 0
    return block


# ===========================================================================
# PromptFormatter — Başlatma
# ===========================================================================


class TestPromptFormatterInit:
    def test_default_mode(self):
        fmt = PromptFormatter()
        assert fmt.mode == FORMAT_DEFAULT

    def test_explicit_modes(self):
        for mode in (FORMAT_DEFAULT, FORMAT_XML, FORMAT_MINIMAL):
            fmt = PromptFormatter(mode=mode)
            assert fmt.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Geçersiz format modu"):
            PromptFormatter(mode="invalid_mode")  # type: ignore[arg-type]

    def test_set_mode_valid(self):
        fmt = PromptFormatter()
        fmt.set_mode(FORMAT_XML)
        assert fmt.mode == FORMAT_XML

    def test_set_mode_invalid_raises(self):
        fmt = PromptFormatter()
        with pytest.raises(ValueError):
            fmt.set_mode("bad_mode")  # type: ignore[arg-type]


# ===========================================================================
# format_block — DEFAULT modu
# ===========================================================================


class TestFormatBlockDefault:
    def setup_method(self):
        self.fmt = PromptFormatter(mode=FORMAT_DEFAULT)

    def test_contains_provenance_label(self):
        block = make_block(provenance=Provenance.EXTERNAL_TOOL)
        result = self.fmt.format_block(block)
        assert "[EXTERNAL_TOOL" in result

    def test_contains_confidence(self):
        block = make_block(confidence=0.75)
        result = self.fmt.format_block(block)
        assert "0.75" in result

    def test_contains_content(self):
        block = make_block(content="Önemli bilgi")
        result = self.fmt.format_block(block)
        assert "Önemli bilgi" in result

    def test_low_confidence_warning_shown(self):
        block = make_block(confidence=0.3)
        result = self.fmt.format_block(block)
        assert "LOW CONFIDENCE" in result

    def test_high_confidence_no_warning(self):
        block = make_block(confidence=0.9)
        result = self.fmt.format_block(block)
        assert "LOW CONFIDENCE" not in result

    def test_derived_from_shown(self):
        block = make_block(derived_from=[1, 2, 3])
        result = self.fmt.format_block(block)
        assert "derived from" in result.lower()
        assert "1" in result

    def test_non_block_raises(self):
        with pytest.raises(TypeError):
            self.fmt.format_block("bu string")  # type: ignore[arg-type]


# ===========================================================================
# format_block — XML modu
# ===========================================================================


class TestFormatBlockXml:
    def setup_method(self):
        self.fmt = PromptFormatter(mode=FORMAT_XML)

    def test_has_opening_tag(self):
        block = make_block()
        result = self.fmt.format_block(block)
        assert "<context_block" in result

    def test_has_closing_tag(self):
        block = make_block()
        result = self.fmt.format_block(block)
        assert "</context_block>" in result

    def test_provenance_attribute_present(self):
        block = make_block(provenance=Provenance.SELF_GENERATED)
        result = self.fmt.format_block(block)
        assert 'provenance="SELF_GENERATED"' in result

    def test_conf_attribute_present(self):
        block = make_block(confidence=0.55)
        result = self.fmt.format_block(block)
        assert "conf=" in result
        assert "0.55" in result

    def test_content_inside_tags(self):
        block = make_block(content="XML içeriği")
        result = self.fmt.format_block(block)
        assert "XML içeriği" in result

    def test_derived_from_in_attributes_when_present(self):
        block = make_block(derived_from=[5, 6])
        result = self.fmt.format_block(block)
        assert "derived_from" in result


# ===========================================================================
# format_block — MINIMAL modu
# ===========================================================================


class TestFormatBlockMinimal:
    def setup_method(self):
        self.fmt = PromptFormatter(mode=FORMAT_MINIMAL)

    def test_no_provenance_label(self):
        block = make_block(provenance=Provenance.EXTERNAL_TOOL)
        result = self.fmt.format_block(block)
        assert "[EXTERNAL_TOOL" not in result

    def test_only_content_returned(self):
        block = make_block(content="Sadece içerik")
        result = self.fmt.format_block(block)
        assert result == "Sadece içerik"

    def test_no_confidence_shown(self):
        block = make_block(confidence=0.3)
        result = self.fmt.format_block(block)
        assert "conf" not in result


# ===========================================================================
# format_all
# ===========================================================================


class TestFormatAll:
    def test_empty_manager_returns_empty_string(self):
        fmt = PromptFormatter()
        mgr = ContextManager()
        assert fmt.format_all(mgr) == ""

    def test_single_block(self):
        fmt = PromptFormatter()
        mgr = ContextManager()
        mgr.add(ContextBlock("Tek blok", Provenance.USER))
        result = fmt.format_all(mgr)
        assert "Tek blok" in result

    def test_multiple_blocks_all_present(self):
        fmt = PromptFormatter()
        mgr = ContextManager()
        mgr.add(ContextBlock("A", Provenance.USER))
        mgr.add(ContextBlock("B", Provenance.EXTERNAL_TOOL))
        mgr.add(ContextBlock("C", Provenance.SELF_GENERATED))
        result = fmt.format_all(mgr)
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_blocks_separated_by_newlines(self):
        fmt = PromptFormatter()
        mgr = ContextManager()
        mgr.add(ContextBlock("X", Provenance.USER))
        mgr.add(ContextBlock("Y", Provenance.USER))
        result = fmt.format_all(mgr)
        assert "\n\n" in result


# ===========================================================================
# get_system_prompt
# ===========================================================================


class TestGetSystemPrompt:
    def test_default_mode_returns_default_explanation(self):
        fmt = PromptFormatter(mode=FORMAT_DEFAULT)
        assert fmt.get_system_prompt() == DEFAULT_SYSTEM_EXPLANATION

    def test_xml_mode_returns_xml_explanation(self):
        fmt = PromptFormatter(mode=FORMAT_XML)
        assert fmt.get_system_prompt() == XML_SYSTEM_EXPLANATION

    def test_minimal_mode_returns_minimal_explanation(self):
        fmt = PromptFormatter(mode=FORMAT_MINIMAL)
        assert fmt.get_system_prompt() == MINIMAL_SYSTEM_EXPLANATION

    def test_default_system_explanation_mentions_self_generated(self):
        assert "SELF_GENERATED" in DEFAULT_SYSTEM_EXPLANATION

    def test_default_system_explanation_mentions_external_tool(self):
        assert "EXTERNAL_TOOL" in DEFAULT_SYSTEM_EXPLANATION

    def test_default_system_explanation_is_non_trivial(self):
        # En az 200 karakter olmalı — detaylı açıklama gerekiyor
        assert len(DEFAULT_SYSTEM_EXPLANATION) > 200


# ===========================================================================
# to_messages (ContextManager)
# ===========================================================================


class TestToMessages:
    def test_returns_list_of_dicts(self):
        mgr = ContextManager()
        mgr.add(ContextBlock("Merhaba", Provenance.USER))
        msgs = mgr.to_messages()
        assert isinstance(msgs, list)
        assert all(isinstance(m, dict) for m in msgs)

    def test_each_message_has_role_and_content(self):
        mgr = ContextManager()
        mgr.add(ContextBlock("X", Provenance.USER))
        for msg in mgr.to_messages():
            assert "role" in msg
            assert "content" in msg

    def test_system_prompt_injected_first(self):
        mgr = ContextManager()
        mgr.add(ContextBlock("X", Provenance.USER))
        msgs = mgr.to_messages(system_prompt="Sistem direktifi")
        assert msgs[0]["role"] == "system"
        assert "Sistem direktifi" in msgs[0]["content"]

    def test_no_system_prompt_no_system_message(self):
        mgr = ContextManager()
        mgr.add(ContextBlock("X", Provenance.USER))
        msgs = mgr.to_messages(system_prompt=None)
        assert all(m["role"] != "system" for m in msgs)
