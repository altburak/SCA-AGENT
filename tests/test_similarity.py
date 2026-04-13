"""
Tests for sca.similarity — SemanticSimilarity
"""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


def _make_sim_with_mock():
    import sca.similarity as sim_mod
    from sca.similarity import SemanticSimilarity

    sim = SemanticSimilarity()
    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        result = []
        for t in texts:
            vec = np.zeros(4)
            for i, ch in enumerate(t[:4]):
                vec[i] = ord(ch) / 256.0
            norm = np.linalg.norm(vec) or 1.0
            result.append(vec / norm)
        return np.array(result)

    mock_model.encode.side_effect = fake_encode
    sim._model = mock_model
    return sim, mock_model


class TestSemanticSimilarityInit(unittest.TestCase):
    def test_default_model_name(self):
        from sca.similarity import DEFAULT_MODEL_NAME, SemanticSimilarity
        sim = SemanticSimilarity()
        self.assertEqual(sim.model_name, DEFAULT_MODEL_NAME)

    def test_custom_model_name(self):
        from sca.similarity import SemanticSimilarity
        sim = SemanticSimilarity("custom/model")
        self.assertEqual(sim.model_name, "custom/model")

    def test_lazy_model_none_at_init(self):
        from sca.similarity import SemanticSimilarity
        sim = SemanticSimilarity()
        self.assertIsNone(sim._model)


class TestSemanticSimilarityLazyLoad(unittest.TestCase):
    def test_model_loads_on_encode(self):
        import sca.similarity as sim_mod
        from sca.similarity import SemanticSimilarity
        sim = SemanticSimilarity()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_st_class = MagicMock(return_value=mock_model)
        original = sim_mod.SentenceTransformer
        sim_mod.SentenceTransformer = mock_st_class
        try:
            sim.encode(["hello", "world"])
            mock_st_class.assert_called_once()
        finally:
            sim_mod.SentenceTransformer = original

    def test_model_loaded_only_once(self):
        import sca.similarity as sim_mod
        from sca.similarity import SemanticSimilarity
        sim = SemanticSimilarity()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_st_class = MagicMock(return_value=mock_model)
        original = sim_mod.SentenceTransformer
        sim_mod.SentenceTransformer = mock_st_class
        try:
            sim.encode(["hello"])
            sim.encode(["world"])
            self.assertEqual(mock_st_class.call_count, 1)
        finally:
            sim_mod.SentenceTransformer = original

    def test_import_error_raises_when_no_module(self):
        import sca.similarity as sim_mod
        from sca.similarity import SemanticSimilarity
        sim = SemanticSimilarity()
        sim._model = None
        original = sim_mod.SentenceTransformer
        sim_mod.SentenceTransformer = None
        with patch("sca.similarity._get_sentence_transformer", side_effect=ImportError("no module")):
            with self.assertRaises(ImportError):
                sim._load_model()
        sim_mod.SentenceTransformer = original


class TestCosineSimlarity(unittest.TestCase):
    def test_identical_texts_high_similarity(self):
        sim, _ = _make_sim_with_mock()
        score = sim.cosine_similarity("Hello world", "Hello world")
        self.assertGreaterEqual(score, 0.99)

    def test_score_in_range(self):
        sim, _ = _make_sim_with_mock()
        score = sim.cosine_similarity("Python is great", "I love programming")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_zero_vector_returns_zero(self):
        from sca.similarity import SemanticSimilarity
        sim = SemanticSimilarity()
        mock_m = MagicMock()
        mock_m.encode.return_value = np.array([[0.0, 0.0], [1.0, 0.0]])
        sim._model = mock_m
        score = sim.cosine_similarity("text a", "text b")
        self.assertEqual(score, 0.0)

    def test_score_clipped_to_one(self):
        sim, mock_model = _make_sim_with_mock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0 + 1e-10, 0.0]])
        score = sim.cosine_similarity("a", "b")
        self.assertLessEqual(score, 1.0)

    def test_score_clipped_to_zero(self):
        sim, mock_model = _make_sim_with_mock()
        mock_model.encode.return_value = np.array([[-1.0, 0.0], [1.0, 0.0]])
        score = sim.cosine_similarity("a", "b")
        self.assertGreaterEqual(score, 0.0)


class TestBatchCosineSimilarity(unittest.TestCase):
    def test_batch_returns_correct_count(self):
        sim, mock_model = _make_sim_with_mock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.7, 0.3]])
        scores = sim.batch_cosine_similarity("base", ["a", "b", "c"])
        self.assertEqual(len(scores), 3)

    def test_batch_scores_in_range(self):
        sim, mock_model = _make_sim_with_mock()
        vecs = np.random.rand(4, 8)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        mock_model.encode.return_value = vecs / norms
        scores = sim.batch_cosine_similarity("base", ["a", "b", "c"])
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_batch_empty_texts(self):
        sim, _ = _make_sim_with_mock()
        scores = sim.batch_cosine_similarity("base", [])
        self.assertEqual(scores, [])

if __name__ == "__main__":
    unittest.main()