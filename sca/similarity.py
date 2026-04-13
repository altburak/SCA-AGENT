"""
CSM — Semantik Benzerlik
=========================
Sentence-transformers tabanlı semantik benzerlik hesaplama.
"""
from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level placeholder — allows patching in tests
SentenceTransformer = None  # type: ignore[assignment]


def _get_sentence_transformer():
    """Lazily import SentenceTransformer, respecting module-level patches."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers kurulu değil. "
                "`pip install sentence-transformers` komutunu çalıştırın."
            ) from exc
    return SentenceTransformer


class SemanticSimilarity:
    """İki metin arasında semantik benzerlik hesaplar.

    sentence-transformers kütüphanesi kullanılır. Model ilk çağrıda yüklenir
    (lazy loading).

    Args:
        model_name: Kullanılacak sentence-transformer modeli.

    Example:
        >>> sim = SemanticSimilarity()
        >>> score = sim.cosine_similarity("The cat sat on the mat.", "A cat was resting on a rug.")
        >>> assert score > 0.7
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Optional[object] = None
        logger.debug("SemanticSimilarity oluşturuldu: model=%s (lazy load).", model_name)

    def _load_model(self) -> None:
        """Modeli ilk kullanımda yükler."""
        if self._model is None:
            ST = _get_sentence_transformer()
            logger.info("Sentence-transformer modeli yükleniyor: %s", self.model_name)
            self._model = ST(self.model_name)
            logger.info("Model yüklendi.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Metin listesini embedding vektörlerine dönüştürür.

        Args:
            texts: Encode edilecek metinler.

        Returns:
            (n_texts, embedding_dim) şeklinde numpy array.
        """
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)  # type: ignore[union-attr]
        logger.debug("Encode edildi: %d metin.", len(texts))
        return embeddings

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """İki metin arasındaki kosinüs benzerliğini döndürür.

        Args:
            text_a: Birinci metin.
            text_b: İkinci metin.

        Returns:
            [0, 1] aralığında benzerlik skoru.
        """
        embeddings = self.encode([text_a, text_b])
        vec_a, vec_b = embeddings[0], embeddings[1]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            logger.warning("Sıfır vektör tespit edildi, benzerlik=0.0 döndürülüyor.")
            return 0.0
        score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        score = float(np.clip(score, 0.0, 1.0))
        logger.debug("Kosinüs benzerliği: %.4f", score)
        return score

    def batch_cosine_similarity(self, base_text: str, texts: list[str]) -> list[float]:
        """Bir baz metin ile metin listesi arasındaki benzerlikleri hesaplar.

        Args:
            base_text: Baz metin.
            texts: Karşılaştırılacak metinler.

        Returns:
            Her metin için [0, 1] aralığında benzerlik skorları.
        """
        if not texts:
            return []
        all_texts = [base_text] + texts
        embeddings = self.encode(all_texts)
        base_vec = embeddings[0]
        base_norm = np.linalg.norm(base_vec)

        scores: list[float] = []
        for i in range(1, len(all_texts)):
            vec = embeddings[i]
            norm = np.linalg.norm(vec)
            if base_norm == 0 or norm == 0:
                scores.append(0.0)
            else:
                raw = float(np.dot(base_vec, vec) / (base_norm * norm))
                scores.append(float(np.clip(raw, 0.0, 1.0)))

        logger.debug("Batch benzerlik hesaplandı: %d metin.", len(texts))
        return scores