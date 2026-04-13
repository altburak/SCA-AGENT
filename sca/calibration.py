"""
CSM — Kalibrasyon Yardımcıları
================================
ConfidenceCalibrator: tahmin edilen skorları gerçek doğruluk oranlarına eşler.
Başlangıçta identity fonksiyon (hiçbir kalibrasyon yok).
AOGL modülünde (Modül 3) CalibrationLearner bu calibrator'ı besler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from sca.confidence import CompositeConfidenceScorer
    from sca.grounding import GroundingLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
MIN_CALIBRATION_SAMPLES: int = 5  # Anlamlı kalibrasyon için minimum örnek
DEFAULT_MIN_SAMPLES_PER_CATEGORY: int = 20
CALIBRATION_BIN_COUNT: int = 5  # Rapor için bin sayısı


# ---------------------------------------------------------------------------
# ConfidenceCalibrator
# ---------------------------------------------------------------------------
class ConfidenceCalibrator:
    """Isotonic regression tabanlı confidence kalibratörü.

    Gerçek doğruluk oranları ile tahmin edilen confidence skorlarını
    eşleştiren bir kalibrasyon modeli tutar.

    Başlangıçta identity fonksiyon (kalibrasyon verisi olmadan ham skoru döndürür).
    ``calibrate()`` ile eğitildikten sonra ``apply()`` kalibre edilmiş skor döndürür.

    Example:
        >>> cal = ConfidenceCalibrator()
        >>> cal.calibrate([0.3, 0.6, 0.9], [0.2, 0.7, 0.85])
        >>> cal.apply(0.6)  # kalibre edilmiş skor
    """

    def __init__(self) -> None:
        self._calibrator: Optional[object] = None
        self._is_fitted: bool = False
        logger.debug("ConfidenceCalibrator başlatıldı (identity fonksiyon).")

    def calibrate(self, predicted: list[float], actual: list[float]) -> None:
        """Isotonic regression ile kalibrasyon modelini eğitir.

        Args:
            predicted: Tahmin edilen confidence skorları [0, 1].
            actual: Gerçek doğruluk oranları [0, 1].

        Raises:
            ValueError: Listeler farklı uzunluktaysa veya yetersiz örnekse.
            ImportError: scikit-learn kurulu değilse.
        """
        if len(predicted) != len(actual):
            raise ValueError(
                f"predicted ve actual listeleri aynı uzunlukta olmalı: "
                f"{len(predicted)} != {len(actual)}"
            )
        if len(predicted) < MIN_CALIBRATION_SAMPLES:
            logger.warning(
                "Kalibrasyon için yetersiz örnek (%d < %d). Identity fonksiyon kullanılacak.",
                len(predicted),
                MIN_CALIBRATION_SAMPLES,
            )
            return

        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise ImportError(
                "scikit-learn kurulu değil. `pip install scikit-learn` komutunu çalıştırın."
            ) from exc

        predicted_arr = np.clip(predicted, 0.0, 1.0)
        actual_arr = np.clip(actual, 0.0, 1.0)

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(predicted_arr, actual_arr)  # type: ignore[union-attr]
        self._is_fitted = True
        logger.info("Kalibrasyon tamamlandı: %d örnek kullanıldı.", len(predicted))

    def apply(self, raw_score: float) -> float:
        """Ham skoru kalibre edilmiş skora dönüştürür.

        Kalibrasyon verisi yoksa identity fonksiyon (ham skoru döndürür).

        Args:
            raw_score: Ham confidence skoru [0, 1].

        Returns:
            Kalibre edilmiş skor [0, 1].
        """
        if not self._is_fitted or self._calibrator is None:
            logger.debug("Kalibrasyon modeli yok, ham skor döndürülüyor: %.4f", raw_score)
            return float(np.clip(raw_score, 0.0, 1.0))

        result = self._calibrator.predict([raw_score])[0]  # type: ignore[union-attr]
        calibrated = float(np.clip(result, 0.0, 1.0))
        logger.debug("Kalibrasyon uygulandı: %.4f → %.4f", raw_score, calibrated)
        return calibrated

    def apply_batch(self, raw_scores: list[float]) -> list[float]:
        """Toplu kalibrasyon uygular.

        Args:
            raw_scores: Ham confidence skoru listesi.

        Returns:
            Kalibre edilmiş skor listesi.
        """
        if not self._is_fitted or self._calibrator is None:
            return [float(np.clip(s, 0.0, 1.0)) for s in raw_scores]

        arr = np.clip(raw_scores, 0.0, 1.0)
        results = self._calibrator.predict(arr)  # type: ignore[union-attr]
        return [float(np.clip(r, 0.0, 1.0)) for r in results]

    @property
    def is_fitted(self) -> bool:
        """Kalibrasyon modelinin eğitilip eğitilmediğini döndürür."""
        return self._is_fitted

    def reset(self) -> None:
        """Kalibrasyon modelini sıfırlar (identity fonksiyona döner)."""
        self._calibrator = None
        self._is_fitted = False
        logger.info("Kalibrasyon modeli sıfırlandı.")


# ---------------------------------------------------------------------------
# CalibrationLearner
# ---------------------------------------------------------------------------
class CalibrationLearner:
    """GroundingLog verilerinden kalibrasyon öğrenen bileşen.

    Her kategori için ayrı bir ConfidenceCalibrator eğitir ve
    CompositeConfidenceScorer'a uygular.

    Args:
        grounding_log: Kalibrasyon verisi kaynağı (GroundingLog).
        min_samples_per_category: Bir kategorinin öğrenilmesi için
            gereken minimum örnek sayısı (default: 20).

    Example:
        >>> learner = CalibrationLearner(grounding_log=log, min_samples_per_category=10)
        >>> calibrators = learner.learn_from_log()
        >>> learner.apply_to_csm(csm_scorer)
        >>> print(learner.report())
    """

    def __init__(
        self,
        grounding_log: "GroundingLog",
        min_samples_per_category: int = DEFAULT_MIN_SAMPLES_PER_CATEGORY,
    ) -> None:
        self.grounding_log = grounding_log
        self.min_samples_per_category = min_samples_per_category
        self._calibrators: dict[str, ConfidenceCalibrator] = {}
        self._calibration_data: dict[str, list[tuple[float, float]]] = {}
        logger.debug(
            "CalibrationLearner başlatıldı: min_samples=%d",
            min_samples_per_category,
        )

    def learn_from_log(self) -> dict[str, ConfidenceCalibrator]:
        """GroundingLog verilerinden kalibrasyon modelleri öğrenir.

        Her kategori için ayrı ConfidenceCalibrator eğitir.
        Yeterli örnek olmayan kategoriler atlanır.

        Returns:
            {category: ConfidenceCalibrator} dict'i (sadece başarıyla
            eğitilen kategoriler).
        """
        raw_data = self.grounding_log.get_calibration_data()
        self._calibration_data = raw_data
        self._calibrators = {}

        for category, pairs in raw_data.items():
            if len(pairs) < self.min_samples_per_category:
                logger.info(
                    "Kategori '%s' atlandı: %d örnek < %d minimum.",
                    category,
                    len(pairs),
                    self.min_samples_per_category,
                )
                continue

            predicted = [p[0] for p in pairs]
            actual = [p[1] for p in pairs]

            cal = ConfidenceCalibrator()
            try:
                cal.calibrate(predicted, actual)
                if cal.is_fitted:
                    self._calibrators[category] = cal
                    logger.info(
                        "Kategori '%s' için kalibrasyon öğrenildi: %d örnek.",
                        category,
                        len(pairs),
                    )
            except Exception as exc:
                logger.warning(
                    "Kategori '%s' kalibrasyon hatası: %s", category, exc
                )

        return dict(self._calibrators)

    def apply_to_csm(self, csm_scorer: "CompositeConfidenceScorer") -> None:
        """Öğrenilen calibrator'ları CSM scorer'ına bağlar.

        CompositeConfidenceScorer'ın _category_calibrators sözlüğünü günceller.
        Bu sözlük, score() metodunda category parametresi verildiğinde kullanılır.

        Args:
            csm_scorer: Güncellenmesi gereken CompositeConfidenceScorer.
        """
        if not hasattr(csm_scorer, "_category_calibrators"):
            csm_scorer._category_calibrators = {}
        csm_scorer._category_calibrators.update(self._calibrators)
        logger.info(
            "%d kategori calibrator'ı CSM'e uygulandı: %s",
            len(self._calibrators),
            list(self._calibrators.keys()),
        )

    def report(self) -> str:
        """İnsan okunabilir kalibrasyon raporu üretir.

        Returns:
            Kategori başına kalibrasyon istatistiklerini içeren string.
            learn_from_log() çağrılmadan önce çağrılırsa kısa mesaj döner.
        """
        if not self._calibration_data:
            return "Kalibrasyon verisi yok. Önce learn_from_log() çağırın."

        lines = ["=== Kalibrasyon Raporu ===\n"]

        for category, pairs in sorted(self._calibration_data.items()):
            n_total = len(pairs)
            lines.append(f"Category: {category}")
            lines.append(f"  Samples: {n_total}")

            is_fitted = (
                category in self._calibrators
                and self._calibrators[category].is_fitted
            )
            if not is_fitted:
                lines.append(
                    f"  Status: YETERSİZ ÖRNEK (min {self.min_samples_per_category})\n"
                )
                continue

            lines.append("  Raw confidence -> Actual accuracy:")

            bin_edges = np.linspace(0.0, 1.0, CALIBRATION_BIN_COUNT + 1)
            for i in range(CALIBRATION_BIN_COUNT):
                low = float(bin_edges[i])
                high = float(bin_edges[i + 1])
                in_bin = [
                    actual
                    for conf, actual in pairs
                    if low <= conf < high
                    or (i == CALIBRATION_BIN_COUNT - 1 and conf == 1.0)
                ]
                if in_bin:
                    avg_actual = sum(in_bin) / len(in_bin)
                    lines.append(
                        f"    {low:.1f}-{high:.1f}: {avg_actual:.2f} (n={len(in_bin)})"
                    )
                else:
                    lines.append(f"    {low:.1f}-{high:.1f}: — (n=0)")

            lines.append("")

        return "\n".join(lines)