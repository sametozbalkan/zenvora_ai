"""X-ray modelleri için Out-of-Distribution (OOD) detection.

NEDEN:
  - Bone (MURA): yalnızca üst ekstremite içerir. Alt ekstremite (bacak, ayak)
    veya farklı modalite (BT/MR/USG/EKG) gelirse model güvensiz çıktı verir.
  - Chest (NIH ChestX-ray14): yetişkin akciğer grafileri içerir. BT axial
    kesit, MR, USG veya tamamen alakasız görüntü gelirse model saçma çıktı verir.
  - Her iki dağılım için "in-distribution" referans bankası kurulur, gelen
    görüntünün bu dağılımdan uzaklığı (Mahalanobis²) ölçülür.

YÖNTEM:
  DenseNet121'in penultimate feature'ı (1024-d, global avgpool sonrası)
  üzerinde Mahalanobis distance. Referans görüntüler:
    Bone : data/mura_calibration/  (MURA train/valid'den normal örnekler)
    Chest: data/nih_calibration/   (NIH "No Finding" etiketli normal örnekler)

KULLANIM:
  detector = BoneOODDetector(transform, device)
  detector.build_calibration(model, calibration_dir)
  score, is_ood = detector.score(model, pil_image)

KALİBRASYON YOKSA:
  build_calibration() başarısız olursa score() (None, False) döner —
  pipeline graceful şekilde devam eder, hiçbir blok atmıyor.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Final

import torch
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

# Numerik stabilite için kovaryans diyagonaline eklenen küçük shrinkage
_COV_SHRINKAGE: Final[float] = 1e-3
# Görüntü file extension whitelist
_VALID_EXTS: Final[set[str]] = {".png", ".jpg", ".jpeg"}


class _MahalanobisOODBase:
    """OOD detector ortak mantığı.

    Alt sınıflar yalnızca `image_mode` (PIL convert hedefi) ile farklılaşır:
    - BoneOODDetector → "RGB" (MURA modeli 3 kanal bekler)
    - ChestOODDetector → "L"  (NIH modeli 1 kanal grayscale bekler)
    """

    _NAME: str = "ood"  # log mesajlarında etiket
    _IMAGE_MODE: str = "RGB"

    def __init__(self, transform, device: torch.device) -> None:
        self._transform = transform
        self._device = device

        # Kalibrasyon state (lazy)
        self._mean: torch.Tensor | None = None
        self._cov_inv: torch.Tensor | None = None
        self._threshold: float | None = None  # 95th percentile of in-dist distances
        self._n_calib: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self._mean is not None and self._cov_inv is not None

    def build_calibration(
        self,
        model: torch.nn.Module,
        calibration_dir: str | Path,
        *,
        min_samples: int = 10,
    ) -> bool:
        """Kalibrasyon klasöründen feature bank kur.

        Returns True iff yeterli örnek + başarılı fit.
        """
        calib_path = Path(calibration_dir)
        if not calib_path.exists():
            LOGGER.info("%s OOD calibration skipped: %s yok", self._NAME, calib_path)
            return False

        image_files = [
            p for p in calib_path.iterdir()
            if p.is_file() and p.suffix.lower() in _VALID_EXTS
        ]
        if len(image_files) < min_samples:
            LOGGER.warning(
                "%s OOD calibration skipped: yalnızca %d örnek bulundu (min %d gerekli)",
                self._NAME, len(image_files), min_samples,
            )
            return False

        features: list[torch.Tensor] = []
        for path in image_files:
            try:
                with open(path, "rb") as fh:
                    img = Image.open(io.BytesIO(fh.read()))
                img = ImageOps.exif_transpose(img).convert(self._IMAGE_MODE)
                feat = self._extract_features(model, img)
                features.append(feat)
            except Exception as exc:
                LOGGER.warning("%s OOD calib örneği atlandı %s: %s", self._NAME, path.name, exc)

        if len(features) < min_samples:
            LOGGER.warning("%s OOD: kullanılabilir örnek sayısı yetersiz", self._NAME)
            return False

        stack = torch.stack(features)                       # (N, D)
        self._mean = stack.mean(dim=0)                      # (D,)
        centered = stack - self._mean                       # (N, D)

        n = float(centered.shape[0])
        cov = (centered.T @ centered) / max(n - 1.0, 1.0)
        cov.diagonal().add_(_COV_SHRINKAGE)
        try:
            self._cov_inv = torch.linalg.inv(cov)
        except RuntimeError:
            self._cov_inv = torch.linalg.pinv(cov)

        with torch.no_grad():
            in_scores = torch.stack([
                self._mahalanobis_sq(f) for f in features
            ])
        self._threshold = float(torch.quantile(in_scores, 0.95).item())
        self._n_calib = len(features)

        LOGGER.info(
            "%s OOD calibration ready | n=%d threshold(95p)=%.2f",
            self._NAME, self._n_calib, self._threshold,
        )
        return True

    def score(
        self, model: torch.nn.Module, pil_image: Image.Image,
    ) -> tuple[float | None, bool]:
        """Gelen görüntü için (mahalanobis^2, is_ood) döner.

        Kalibre değilse (None, False) — caller graceful devam etmeli.
        """
        if not self.is_calibrated:
            return (None, False)
        img = pil_image.convert(self._IMAGE_MODE)
        feat = self._extract_features(model, img)
        dist_sq = float(self._mahalanobis_sq(feat).item())
        is_ood = dist_sq > (self._threshold or float("inf"))
        return (dist_sq, is_ood)

    # ── Internals ──────────────────────────────────────────────────────────

    def _mahalanobis_sq(self, feat: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._cov_inv is not None
        delta = feat - self._mean
        return delta @ self._cov_inv @ delta

    def _extract_features(
        self, model: torch.nn.Module, pil_image: Image.Image,
    ) -> torch.Tensor:
        """DenseNet121'in classifier öncesi 1024-d feature'ı (global avg pool sonrası)."""
        x = self._transform(pil_image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            feats = model.features(x)
            feats = torch.nn.functional.relu(feats, inplace=False)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1)
        return feats.squeeze(0).detach().cpu()


class BoneOODDetector(_MahalanobisOODBase):
    """MURA tabanlı bone OOD detector — RGB convert (3 kanal ImageNet pipeline)."""
    _NAME = "bone"
    _IMAGE_MODE = "RGB"


class ChestOODDetector(_MahalanobisOODBase):
    """NIH tabanlı chest OOD detector — L convert (1 kanal grayscale)."""
    _NAME = "chest"
    _IMAGE_MODE = "L"
