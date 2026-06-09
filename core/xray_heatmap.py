"""Grad-CAM tabanlı X-ray heatmap üretimi.

Modelin sınıflandırma çıktısı ("Pneumothorax %78") "neden" sorusuna cevap
vermez. Grad-CAM, DenseNet'in son conv katmanından gelen aktivasyon
haritalarını sınıfa-özgü gradyanlarla ağırlıklandırarak modelin
görüntünün hangi bölgesine baktığını gösterir.

KULLANIM:
    generator = HeatmapGenerator(model, target_layers=[model.features.denseblock4])
    png_bytes = generator.heatmap_for_class(
        input_tensor, pil_image_rgb, target_class_idx,
    )
    # png_bytes → base64 encode et → frontend overlay olarak gösterir

İÇERİK:
    Üretilen PNG, **orijinal görüntü + jet renkli heatmap overlay** birleşimi.
    Frontend ek işlem yapmadan <img src="data:image/png;base64,..." /> ile gösterir.

NEDEN HER FINDING İÇİN AYRI HEATMAP:
    Multi-label modelde her sınıf farklı bölgeye bakar — Cardiomegaly
    kalp gölgesini, Pneumothorax akciğer kenarını, Pneumonia parankim
    yoğunlaşmasını işaretler. Tek heatmap yetersiz.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

LOGGER = logging.getLogger(__name__)

# Heatmap'in görüntü ile birleşim oranı — 0.5 = yarı saydam overlay
_HEATMAP_ALPHA: float = 0.5


@dataclass(frozen=True)
class HeatmapResult:
    """PNG bytes + base64 — caller hangisi lazımsa onu kullanır."""
    png_bytes: bytes
    base64_str: str


class HeatmapGenerator:
    """Stateless Grad-CAM üreteci — thread-safe.

    GradCAM forward/backward hook'ları model üzerine ekler; aynı GradCAM
    nesnesini paralel çağırmak (örn. concurrent FastAPI request'leri)
    hook conflict yaratabilir. Bu sınıf her `heatmap_for_class` çağrısında
    `with GradCAM(...)` bloğunda yeni bir cam yaratır; hook'lar __exit__
    ile temizlenir, istek-izole.

    Instance state'i sadece (model, target_layers) — okuma-only, paylaşımı
    güvenli.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Sequence[torch.nn.Module],
    ) -> None:
        self._model = model
        self._target_layers = list(target_layers)

    def heatmap_for_class(
        self,
        input_tensor: torch.Tensor,
        display_image: Image.Image,
        target_class_idx: int,
        *,
        image_size: int = 224,
    ) -> HeatmapResult:
        """Tek bir sınıf için heatmap üret. Her çağrı yeni GradCAM oluşturur.

        Args:
            input_tensor: Modele giren ön işlenmiş tensor. (1, C, H, W)
            display_image: Overlay'in üzerine binecek görüntü (PIL).
            target_class_idx: Hangi sınıf için heatmap?
            image_size: Çıktı görüntü boyutu (kare).
        """
        targets = [ClassifierOutputTarget(target_class_idx)]

        # Per-request GradCAM — hook'lar bu blok dışında garanti temizlenir
        with GradCAM(model=self._model, target_layers=self._target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]  # (H, W) float32 [0,1]

        # Display image'ı normalize et — show_cam_on_image float [0,1] RGB ister
        rgb = display_image.convert("RGB").resize((image_size, image_size))
        rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0  # (H, W, 3)

        # Overlay üret — use_rgb=True (matplotlib jet -> RGB)
        overlay = show_cam_on_image(
            rgb_arr, grayscale_cam, use_rgb=True, image_weight=1.0 - _HEATMAP_ALPHA,
        )  # (H, W, 3) uint8

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return HeatmapResult(png_bytes=png_bytes, base64_str=b64)
