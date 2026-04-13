"""
ContextBlock ve ContextManager için pytest test paketi.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sca.context import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DEFAULT_CONFIDENCE,
    ContextBlock,
    ContextManager,
    Provenance,
)


# ===========================================================================
# Yardımcı fabrikalar
# ===========================================================================


def make_block(
    content: str = "Test içeriği",
    provenance: Provenance = Provenance.USER,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ContextBlock:
    return ContextBlock(content=content, provenance=provenance, confidence=confidence)


# ===========================================================================
# ContextBlock testleri
# ===========================================================================


class TestContextBlockCreation:
    """ContextBlock yaratma senaryoları."""

    def test_default_confidence(self):
        block = make_block()
        assert block.confidence == DEFAULT_CONFIDENCE

    def test_custom_confidence_valid_min(self):
        block = make_block(confidence=CONFIDENCE_MIN)
        assert block.confidence == CONFIDENCE_MIN

    def test_custom_confidence_valid_max(self):
        block = make_block(confidence=CONFIDENCE_MAX)
        assert block.confidence == CONFIDENCE_MAX

    def test_custom_confidence_mid(self):
        block = make_block(confidence=0.73)
        assert block.confidence == pytest.approx(0.73)

    def test_invalid_confidence_above_max(self):
        with pytest.raises(ValueError, match="confidence"):
            make_block(confidence=1.01)

    def test_invalid_confidence_below_min(self):
        with pytest.raises(ValueError, match="confidence"):
            make_block(confidence=-0.01)

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="boş"):
            ContextBlock(content="", provenance=Provenance.USER)

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValueError, match="boş"):
            ContextBlock(content="   ", provenance=Provenance.USER)

    def test_invalid_provenance_type_raises(self):
        with pytest.raises(TypeError):
            ContextBlock(content="x", provenance="USER")  # type: ignore[arg-type]

    def test_default_block_id_is_minus_one(self):
        block = make_block()
        assert block.block_id == -1

    def test_timestamp_is_utc(self):
        block = make_block()
        assert block.timestamp.tzinfo is not None

    def test_derived_from_default_empty(self):
        block = make_block()
        assert block.derived_from == []

    def test_metadata_default_empty(self):
        block = make_block()
        assert block.metadata == {}

    def test_all_provenance_values_valid(self):
        for prov in Provenance:
            block = make_block(provenance=prov)
            assert block.provenance == prov


class TestContextBlockIsLowConfidence:
    def test_low_confidence_below_threshold(self):
        block = make_block(confidence=0.3)
        assert block.is_low_confidence(threshold=0.5) is True

    def test_not_low_confidence_above_threshold(self):
        block = make_block(confidence=0.8)
        assert block.is_low_confidence(threshold=0.5) is False

    def test_equal_to_threshold_not_low(self):
        block = make_block(confidence=0.5)
        assert block.is_low_confidence(threshold=0.5) is False

    def test_custom_threshold(self):
        block = make_block(confidence=0.7)
        assert block.is_low_confidence(threshold=0.8) is True


class TestContextBlockSerialization:
    def test_to_dict_contains_all_keys(self):
        block = make_block()
        d = block.to_dict()
        assert set(d.keys()) == {
            "block_id", "content", "provenance", "confidence",
            "timestamp", "derived_from", "metadata",
        }

    def test_to_dict_provenance_is_string(self):
        block = make_block(provenance=Provenance.EXTERNAL_TOOL)
        assert block.to_dict()["provenance"] == "EXTERNAL_TOOL"

    def test_from_dict_roundtrip(self):
        original = ContextBlock(
            content="Test",
            provenance=Provenance.SELF_GENERATED,
            confidence=0.6,
            derived_from=[0, 1],
            metadata={"tool": "search"},
        )
        original.block_id = 42
        d = original.to_dict()
        restored = ContextBlock.from_dict(d)
        assert restored.content == original.content
        assert restored.provenance == original.provenance
        assert restored.confidence == pytest.approx(original.confidence)
        assert restored.derived_from == original.derived_from
        assert restored.metadata == original.metadata
        assert restored.block_id == original.block_id

    def test_from_dict_invalid_provenance_raises(self):
        d = make_block().to_dict()
        d["provenance"] = "INVALID_PROV"
        with pytest.raises(ValueError):
            ContextBlock.from_dict(d)


# ===========================================================================
# ContextManager testleri
# ===========================================================================


class TestContextManagerAddGet:
    def test_add_returns_int_id(self):
        mgr = ContextManager()
        bid = mgr.add(make_block())
        assert isinstance(bid, int)

    def test_add_increments_id(self):
        mgr = ContextManager()
        id0 = mgr.add(make_block())
        id1 = mgr.add(make_block())
        assert id1 == id0 + 1

    def test_get_block_returns_correct_block(self):
        mgr = ContextManager()
        bid = mgr.add(make_block(content="Merhaba"))
        assert mgr.get_block(bid).content == "Merhaba"

    def test_get_block_missing_raises_key_error(self):
        mgr = ContextManager()
        with pytest.raises(KeyError):
            mgr.get_block(999)

    def test_add_non_block_raises_type_error(self):
        mgr = ContextManager()
        with pytest.raises(TypeError):
            mgr.add("bu bir string")  # type: ignore[arg-type]

    def test_len_reflects_added_blocks(self):
        mgr = ContextManager()
        assert len(mgr) == 0
        mgr.add(make_block())
        assert len(mgr) == 1
        mgr.add(make_block())
        assert len(mgr) == 2

    def test_bool_false_when_empty(self):
        mgr = ContextManager()
        assert not mgr

    def test_bool_true_when_not_empty(self):
        mgr = ContextManager()
        mgr.add(make_block())
        assert mgr


class TestContextManagerRemoveClear:
    def test_remove_existing_block(self):
        mgr = ContextManager()
        bid = mgr.add(make_block())
        mgr.remove(bid)
        assert len(mgr) == 0

    def test_remove_nonexistent_raises_key_error(self):
        mgr = ContextManager()
        with pytest.raises(KeyError):
            mgr.remove(999)

    def test_clear_empties_manager(self):
        mgr = ContextManager()
        mgr.add(make_block())
        mgr.add(make_block())
        mgr.clear()
        assert len(mgr) == 0

    def test_blocks_property_returns_copy(self):
        mgr = ContextManager()
        mgr.add(make_block())
        copy = mgr.blocks
        copy.clear()
        assert len(mgr) == 1  # orijinal etkilenmemeli


class TestContextManagerProvenanceFilter:
    def test_get_by_provenance_returns_correct_subset(self):
        mgr = ContextManager()
        mgr.add(make_block(provenance=Provenance.USER))
        mgr.add(make_block(provenance=Provenance.EXTERNAL_TOOL))
        mgr.add(make_block(provenance=Provenance.USER))

        user_blocks = mgr.get_by_provenance(Provenance.USER)
        assert len(user_blocks) == 2
        assert all(b.provenance == Provenance.USER for b in user_blocks)

    def test_get_by_provenance_empty_result(self):
        mgr = ContextManager()
        mgr.add(make_block(provenance=Provenance.USER))
        result = mgr.get_by_provenance(Provenance.KNOWLEDGE_BASE)
        assert result == []

    def test_get_by_provenance_invalid_type_raises(self):
        mgr = ContextManager()
        with pytest.raises(TypeError):
            mgr.get_by_provenance("USER")  # type: ignore[arg-type]

    def test_all_provenances_filterable(self):
        mgr = ContextManager()
        for prov in Provenance:
            mgr.add(make_block(provenance=prov))
        for prov in Provenance:
            result = mgr.get_by_provenance(prov)
            assert len(result) == 1
            assert result[0].provenance == prov


class TestContextManagerPersistence:
    def test_save_and_load_roundtrip(self):
        mgr = ContextManager()
        mgr.add(make_block(content="Blok 1", provenance=Provenance.USER))
        mgr.add(
            ContextBlock(
                content="Blok 2",
                provenance=Provenance.DERIVED_INFERENCE,
                confidence=0.4,
                derived_from=[0],
                metadata={"source": "test"},
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name

        mgr.save_to_json(path)

        mgr2 = ContextManager()
        mgr2.load_from_json(path)

        assert len(mgr2) == 2
        assert mgr2.blocks[0].content == "Blok 1"
        assert mgr2.blocks[1].provenance == Provenance.DERIVED_INFERENCE
        assert mgr2.blocks[1].confidence == pytest.approx(0.4)
        assert mgr2.blocks[1].derived_from == [0]
        assert mgr2.blocks[1].metadata == {"source": "test"}

    def test_load_missing_file_raises(self):
        mgr = ContextManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_from_json("/nonexistent/path/file.json")

    def test_save_creates_parent_directories(self):
        mgr = ContextManager()
        mgr.add(make_block())
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir" / "ctx.json"
            mgr.save_to_json(nested)
            assert nested.exists()

    def test_json_format_is_valid(self):
        mgr = ContextManager()
        mgr.add(make_block())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        mgr.save_to_json(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "version" in data
        assert "blocks" in data
        assert isinstance(data["blocks"], list)
