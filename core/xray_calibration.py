"""X-ray modellerinin Platt scaling kalibrasyonu.

Eğitim sırasında BCE loss minimize edilmiş bir modelin sigmoid çıktısı
"gerçek olasılık" değildir — overconfident veya underconfident olabilir.
Bu modül validation set üzerinde fit edilmiş Platt parametreleriyle
(scripts/kaggle_calibrate.py çıktısı) raw probs → kalibre olasılığa
dönüştürür.

KULLANIM:
    cal = ChestCalibrator.load("models/chest_calibration.json")
    calibrated_probs = cal.apply(raw_probs, CHEST_LABELS)
    threshold = cal.threshold_for("Pneumothorax")
    is_low_conf = cal.is_low_confidence("Pneumothorax")

KALİBRASYON DOSYASI YOKSA:
    cal = ChestCalibrator.load_or_default("models/chest_calibration.json")
    → identity calibrator döner, raw probs aynen geçer (graceful degradation).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger(__name__)

# logit() için 0/1 sınır clamp epsilonu
_EPS = 1e-7


def _logit(p: float) -> float:
    p = max(_EPS, min(1.0 - _EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    # math.exp overflow korumalı sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class LabelParams:
    platt_a: float
    platt_b: float
    threshold: float
    auroc: float
    ece_after: float
    n_positives: int


@dataclass(frozen=True)
class _Identity:
    """Kalibrasyon dosyası yoksa kullanılan no-op calibrator."""
    is_identity: bool = True

    def apply_one(self, raw_prob: float, label: str) -> float:
        return float(raw_prob)

    def apply(self, raw_probs: Sequence[float], labels: Sequence[str]) -> list[float]:
        return [float(p) for p in raw_probs]

    def threshold_for(self, label: str, default: float = 0.35) -> float:
        return default

    def is_low_confidence(self, label: str) -> bool:
        return False


class ChestCalibrator:
    """Multi-label chest classifier için per-label Platt scaling uygulayıcısı."""

    is_identity: bool = False

    def __init__(
        self,
        per_label: dict[str, LabelParams],
        low_confidence: frozenset[str],
        meta: dict,
    ) -> None:
        self._per_label = per_label
        self._low_confidence = low_confidence
        self._meta = meta

    # ── Loaders ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "ChestCalibrator":
        """Kalibrasyon dosyasını yükle; yoksa veya bozuksa FileNotFoundError."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Calibration file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls._from_dict(data)

    @classmethod
    def load_or_default(cls, path: str | Path) -> "ChestCalibrator | _Identity":
        """Dosya yoksa identity döner — pipeline graceful devam eder."""
        try:
            cal = cls.load(path)
            LOGGER.info(
                "chest calibration loaded | n_labels=%d low_conf=%s",
                len(cal._per_label), sorted(cal._low_confidence),
            )
            return cal
        except FileNotFoundError:
            LOGGER.info("chest calibration file yok — identity calibrator kullanılıyor")
            return _Identity()
        except Exception as exc:
            LOGGER.warning("chest calibration yükleme hatası, identity: %s", exc)
            return _Identity()

    @classmethod
    def _from_dict(cls, data: dict) -> "ChestCalibrator":
        labels_raw = data.get("labels", {})
        per_label: dict[str, LabelParams] = {}
        threshold_override = data.get("low_confidence_threshold_override", {})

        for name, entry in labels_raw.items():
            threshold = float(
                threshold_override.get(
                    name, entry.get("optimal_threshold_calibrated", 0.35),
                )
            )
            per_label[name] = LabelParams(
                platt_a=float(entry["platt_a"]),
                platt_b=float(entry["platt_b"]),
                threshold=threshold,
                auroc=float(entry.get("auroc", 0.5)),
                ece_after=float(entry.get("ece_after", 0.0)),
                n_positives=int(entry.get("n_positives", 0)),
            )

        low_conf = frozenset(data.get("low_confidence_labels", []))
        return cls(per_label=per_label, low_confidence=low_conf, meta=dict(data))

    # ── Public API ───────────────────────────────────────────────────────

    def apply_one(self, raw_prob: float, label: str) -> float:
        """Tek bir raw sigmoid çıktıyı kalibre et."""
        params = self._per_label.get(label)
        if params is None:
            return float(raw_prob)
        z = _logit(float(raw_prob))
        return _sigmoid(params.platt_a * z + params.platt_b)

    def apply(self, raw_probs: Sequence[float], labels: Sequence[str]) -> list[float]:
        """Label listesine paralel sigmoid çıktılarını kalibre et."""
        return [self.apply_one(p, lbl) for p, lbl in zip(raw_probs, labels)]

    def threshold_for(self, label: str, default: float = 0.35) -> float:
        """Bu label için F1-optimal kalibre threshold; yoksa default."""
        params = self._per_label.get(label)
        return params.threshold if params else default

    def is_low_confidence(self, label: str) -> bool:
        """AUROC<0.65 olan acil bulgular için True — UI uyarısı eklenir."""
        return label in self._low_confidence

    def label_meta(self, label: str) -> LabelParams | None:
        return self._per_label.get(label)


# ═══════════════════════════════════════════════════════════════════════════
# BONE — binary (abnormal var/yok)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _BoneIdentity:
    """Bone kalibrasyon dosyası yoksa kullanılan no-op."""
    is_identity: bool = True

    def apply(self, raw_prob: float) -> float:
        return float(raw_prob)

    @property
    def threshold(self) -> float:
        return 0.5

    @property
    def auroc(self) -> float:
        return 0.5

    @property
    def is_low_confidence(self) -> bool:
        return True  # kalibrasyon yokken her zaman düşük güven olarak işaretle


class BoneCalibrator:
    """Bone binary classifier için Platt scaling uygulayıcısı."""

    is_identity: bool = False

    def __init__(
        self,
        platt_a: float,
        platt_b: float,
        threshold: float,
        auroc: float,
        ece_after: float,
        meta: dict,
    ) -> None:
        self._a = platt_a
        self._b = platt_b
        self._threshold = threshold
        self._auroc = auroc
        self._ece_after = ece_after
        self._meta = meta

    @classmethod
    def load(cls, path: str | Path) -> "BoneCalibrator":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Bone calibration file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls._from_dict(data)

    @classmethod
    def load_or_default(cls, path: str | Path) -> "BoneCalibrator | _BoneIdentity":
        try:
            cal = cls.load(path)
            LOGGER.info(
                "bone calibration loaded | auroc=%.3f threshold=%.2f ece_after=%.3f",
                cal._auroc, cal._threshold, cal._ece_after,
            )
            return cal
        except FileNotFoundError:
            LOGGER.info("bone calibration file yok — identity calibrator kullanılıyor")
            return _BoneIdentity()
        except Exception as exc:
            LOGGER.warning("bone calibration yükleme hatası, identity: %s", exc)
            return _BoneIdentity()

    @classmethod
    def _from_dict(cls, data: dict) -> "BoneCalibrator":
        entry = data.get("abnormal", {})
        if not entry:
            raise ValueError("Bone calibration missing 'abnormal' block")
        return cls(
            platt_a=float(entry["platt_a"]),
            platt_b=float(entry["platt_b"]),
            threshold=float(entry.get("optimal_threshold_calibrated", 0.5)),
            auroc=float(entry.get("auroc", 0.5)),
            ece_after=float(entry.get("ece_after", 0.0)),
            meta=dict(data),
        )

    # ── Public API ───────────────────────────────────────────────────────

    def apply(self, raw_prob: float) -> float:
        """Ham sigmoid çıktıyı Platt parametreleriyle kalibre et."""
        z = _logit(float(raw_prob))
        return _sigmoid(self._a * z + self._b)

    @property
    def threshold(self) -> float:
        """F1-optimal kalibre threshold (validation set'ten)."""
        return self._threshold

    @property
    def auroc(self) -> float:
        return self._auroc

    @property
    def is_low_confidence(self) -> bool:
        """AUROC<0.65 ise model güvensiz — UI'da uyarı eklenir."""
        return self._auroc < 0.65
