from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Final

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

from core.xray_calibration import BoneCalibrator, ChestCalibrator
from core.xray_heatmap import HeatmapGenerator
from core.xray_ood import BoneOODDetector, ChestOODDetector

LOGGER = logging.getLogger(__name__)

CHEST_LABELS: Final[list[str]] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# Genel fallback eÅŸik â€” etiket-Ã¶zel eÅŸik yoksa kullanÄ±lÄ±r
CHEST_THRESHOLD: Final[float] = 0.35
BONE_ABNORMAL_THRESHOLD: Final[float] = 0.5

# Etiket-Ã¶zel eÅŸikler â€” NIH ChestX-ray14'te sÄ±nÄ±f dengesizliÄŸi nedeniyle
# tek eÅŸik tÃ¼m patolojiler iÃ§in iyi Ã§alÄ±ÅŸmaz. DÃ¼ÅŸÃ¼k base-rate hastalÄ±klarda
# (Pneumonia, Mass) daha dÃ¼ÅŸÃ¼k eÅŸik recall'u korur; yÃ¼ksek base-rate'lerde
# (Infiltration) daha yÃ¼ksek eÅŸik false-positive'i azaltÄ±r.
# Bu deÄŸerler literatÃ¼r ortalamasÄ± â€” validation set ile finetune Ã¶nerilir.
CHEST_LABEL_THRESHOLDS: Final[dict[str, float]] = {
    "Atelectasis":        0.35,
    "Cardiomegaly":       0.30,
    "Effusion":           0.35,
    "Infiltration":       0.45,
    "Mass":               0.25,
    "Nodule":             0.30,
    "Pneumonia":          0.25,
    "Pneumothorax":       0.25,
    "Consolidation":      0.35,
    "Edema":              0.30,
    "Emphysema":          0.35,
    "Fibrosis":           0.35,
    "Pleural_Thickening": 0.40,
    "Hernia":             0.35,
}


def _clean_state_dict(raw_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(raw_obj, dict) and "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
        raw_obj = raw_obj["state_dict"]

    if not isinstance(raw_obj, dict):
        raise ValueError("Checkpoint is not a valid state_dict")

    if not all(isinstance(v, torch.Tensor) for v in raw_obj.values()):
        raise ValueError("Checkpoint dict contains non-tensor values")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in raw_obj.items():
        key = key[7:] if key.startswith("module.") else key
        cleaned[key] = value
    return cleaned


def _build_densenet(num_classes: int, in_channels: int) -> torch.nn.Module:
    model = models.densenet121(weights=None)
    if in_channels != 3:
        old_conv = model.features.conv0
        model.features.conv0 = torch.nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    return model


def _infer_shape_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    conv_weight = state_dict.get("features.conv0.weight")
    classifier_weight = state_dict.get("classifier.weight")

    if conv_weight is None or classifier_weight is None:
        raise ValueError("Checkpoint is missing required DenseNet keys")
    if conv_weight.ndim != 4 or classifier_weight.ndim != 2:
        raise ValueError("Checkpoint tensor shapes are invalid")

    in_channels = int(conv_weight.shape[1])
    num_classes = int(classifier_weight.shape[0])
    return in_channels, num_classes


def _is_dicom(contents: bytes) -> bool:
    return len(contents) > 132 and contents[128:132] == b"DICM"


def _load_image(contents: bytes) -> Image.Image:
    if _is_dicom(contents):
        try:
            import numpy as np
            import pydicom
        except ImportError as exc:
            raise RuntimeError("DICOM support requires pydicom and numpy installed") from exc

        ds = pydicom.dcmread(io.BytesIO(contents))
        arr = ds.pixel_array.astype("float32")
        arr -= arr.min()
        max_val = float(arr.max())
        if max_val > 0:
            arr /= max_val
        arr = (arr * 255.0).clip(0, 255).astype("uint8")
        return Image.fromarray(arr).convert("L")

    # Telefondan Ã§ekilen yan rÃ¶ntgenler iÃ§in EXIF orientation dÃ¼zeltmesi
    img = Image.open(io.BytesIO(contents))
    img = ImageOps.exif_transpose(img)
    return img.convert("L")


def _crop_by_rel_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left = max(0, min(width - 1, round(width * box[0])))
    upper = max(0, min(height - 1, round(height * box[1])))
    right = max(left + 1, min(width, round(width * box[2])))
    lower = max(upper + 1, min(height, round(height * box[3])))
    return image.crop((left, upper, right, lower))


def _bone_crop_candidates(image: Image.Image) -> list[tuple[str, Image.Image]]:
    """Generate ROI crops so small local fractures are not diluted by a full view."""
    rel_boxes: list[tuple[str, tuple[float, float, float, float]]] = [
        ("full", (0.00, 0.00, 1.00, 1.00)),
        ("center_80", (0.10, 0.10, 0.90, 0.90)),
        ("top_70", (0.00, 0.00, 1.00, 0.70)),
        ("bottom_70", (0.00, 0.30, 1.00, 1.00)),
        ("left_70", (0.00, 0.00, 0.70, 1.00)),
        ("right_70", (0.30, 0.00, 1.00, 1.00)),
        ("top_left_62", (0.00, 0.00, 0.62, 0.62)),
        ("top_right_62", (0.38, 0.00, 1.00, 0.62)),
        ("bottom_left_62", (0.00, 0.38, 0.62, 1.00)),
        ("bottom_right_62", (0.38, 0.38, 1.00, 1.00)),
    ]
    return [(name, _crop_by_rel_box(image, box)) for name, box in rel_boxes]


class XRayPredictor:
    def __init__(
        self,
        models_dir: str | Path = "models",
        *,
        eager: bool | None = None,
    ) -> None:
        """X-ray predictor.

        eager:
          - True  â†’ __init__'te her iki modeli yÃ¼kle (varsayÄ±lan).
          - False â†’ modeller ilk predict_* Ã§aÄŸrÄ±sÄ±nda lazy yÃ¼klenir.
          - None  â†’ ZENVORA_XRAY_LAZY=1 ise lazy, aksi halde eager.
        """
        if eager is None:
            eager = os.getenv("ZENVORA_XRAY_LAZY", "0") != "1"

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models_dir = Path(models_dir)

        # Lazy state â€” _ensure_chest / _ensure_bone tarafÄ±ndan doldurulur
        self._chest_model: torch.nn.Module | None = None
        self._bone_model: torch.nn.Module | None = None
        self._bone_num_classes: int = 0
        # Grad-CAM heatmap Ã¼reteÃ§leri â€” model yÃ¼klendikten sonra kurulur
        self._chest_heatmap: HeatmapGenerator | None = None
        self._bone_heatmap: HeatmapGenerator | None = None
        # Heatmap Ã¼retim toggle â€” testlerde / CPU yavaÅŸlÄ±ÄŸÄ±nda kapatÄ±labilir
        self._heatmap_enabled = os.getenv("ZENVORA_DISABLE_HEATMAP", "0") != "1"
        # OOD detector'lar lazy â€” _ensure_chest/_ensure_bone sonrasÄ± build edilir.
        # NOT: env adlarÄ±ndaki "calibration" tarihsel â€” aslÄ±nda OOD reference set'i.
        self._bone_ood: BoneOODDetector | None = None
        self._bone_ood_dir = Path(os.getenv(
            "ZENVORA_MURA_CALIBRATION_DIR", "data/mura_calibration",
        ))
        self._chest_ood: ChestOODDetector | None = None
        self._chest_ood_dir = Path(os.getenv(
            "ZENVORA_NIH_CALIBRATION_DIR", "data/nih_calibration",
        ))
        # Chest Platt calibration â€” JSON yoksa identity calibrator
        self._chest_calibration = ChestCalibrator.load_or_default(
            self._models_dir / "chest_calibration.json"
        )
        # Bone Platt calibration â€” JSON yoksa identity (raw probs)
        self._bone_calibration = BoneCalibrator.load_or_default(
            self._models_dir / "bone_calibration.json"
        )

        # torchxrayvision normalize(img, 255): [0,1] â†’ [-1024, 1024]
        # EÄŸitimle aynÄ± daÄŸÄ±lÄ±mÄ± korumak iÃ§in: (x - 0.5) * 2048
        self._chest_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2048.0),
        ])
        # MURA pipeline'Ä± aspect-ratio koruyan Resize(256) + CenterCrop(224)
        # kullanÄ±r. Ã–nceki Resize((224, 224)) gÃ¶rseli sÄ±kÄ±ÅŸtÄ±rarak deforme ediyordu.
        self._bone_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if eager:
            self._ensure_chest()
            self._ensure_bone()
            LOGGER.info("xray predictor ready (eager) | device=%s", self._device)
        else:
            LOGGER.info("xray predictor ready (lazy)  | device=%s", self._device)

    # â”€â”€ Lazy model loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _ensure_chest(self) -> torch.nn.Module:
        if self._chest_model is not None:
            return self._chest_model
        chest_path = self._models_dir / "nih_finetuned.pt"
        if not chest_path.exists():
            raise FileNotFoundError(f"Missing chest model: {chest_path}")
        state = _clean_state_dict(torch.load(chest_path, map_location="cpu", weights_only=True))
        in_channels, num_classes = _infer_shape_from_state_dict(state)
        if num_classes != len(CHEST_LABELS):
            raise ValueError(
                f"Chest model class count mismatch: got {num_classes}, expected {len(CHEST_LABELS)}"
            )
        model = _build_densenet(num_classes=num_classes, in_channels=in_channels)
        model.load_state_dict(state, strict=True)
        model.to(self._device).eval()
        self._chest_model = model

        # Grad-CAM target: son dense block (DenseNet121 features.denseblock4)
        if self._heatmap_enabled:
            try:
                self._chest_heatmap = HeatmapGenerator(
                    model, target_layers=[model.features.denseblock4]
                )
            except Exception as exc:
                LOGGER.warning("chest heatmap generator init failed: %s", exc)
                self._chest_heatmap = None

        # OOD detector kalibrasyonu â€” ZENVORA_NIH_CALIBRATION_DIR varsa aktifleÅŸir
        try:
            self._chest_ood = ChestOODDetector(self._chest_transform, self._device)
            self._chest_ood.build_calibration(model, self._chest_ood_dir)
        except Exception as exc:
            LOGGER.warning("chest OOD detector init failed: %s", exc)
            self._chest_ood = None

        LOGGER.info("xray chest model loaded")
        return model

    def _ensure_bone(self) -> torch.nn.Module:
        if self._bone_model is not None:
            return self._bone_model
        bone_path = self._models_dir / "mura_finetuned.pt"
        if not bone_path.exists():
            raise FileNotFoundError(f"Missing bone model: {bone_path}")
        state = _clean_state_dict(torch.load(bone_path, map_location="cpu", weights_only=True))
        in_channels, num_classes = _infer_shape_from_state_dict(state)
        self._bone_num_classes = num_classes
        model = _build_densenet(num_classes=num_classes, in_channels=in_channels)
        model.load_state_dict(state, strict=True)
        model.to(self._device).eval()
        self._bone_model = model
        LOGGER.info("xray bone model loaded")

        # Grad-CAM target: aynÄ± ÅŸekilde son dense block
        if self._heatmap_enabled:
            try:
                self._bone_heatmap = HeatmapGenerator(
                    model, target_layers=[model.features.denseblock4]
                )
            except Exception as exc:
                LOGGER.warning("bone heatmap generator init failed: %s", exc)
                self._bone_heatmap = None

        # OOD detector kalibrasyonu â€” bone model yÃ¼klendikten sonra denenir.
        # data/mura_calibration/ klasÃ¶rÃ¼nde â‰¥10 Ã¶rnek varsa aktifleÅŸir.
        try:
            self._bone_ood = BoneOODDetector(self._bone_transform, self._device)
            self._bone_ood.build_calibration(model, self._bone_ood_dir)
        except Exception as exc:
            LOGGER.warning("bone OOD detector init failed: %s", exc)
            self._bone_ood = None

        return model

    def predict_chest(self, contents: bytes) -> dict:
        started = time.perf_counter()
        model = self._ensure_chest()

        # GÃ¶rÃ¼ntÃ¼yÃ¼ iki kez yÃ¼klemeyelim â€” bir kez yÃ¼kle, tensor ve display iÃ§in kullan
        pil_image = _load_image(contents)  # zaten grayscale L dÃ¶ndÃ¼rÃ¼r
        x = self._chest_transform(pil_image).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            logits = model(x)
            raw_probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

        # Platt scaling â€” JSON kalibrasyon dosyasÄ± varsa raw â†’ calibrated
        cal = self._chest_calibration
        calibrated = cal.apply(raw_probs, CHEST_LABELS)
        raw_by_label = dict(zip(CHEST_LABELS, (float(p) for p in raw_probs)))
        cal_by_label = dict(zip(CHEST_LABELS, (float(p) for p in calibrated)))

        # Threshold: kalibratÃ¶rden F1-optimal deÄŸer; yoksa literatÃ¼r fallback
        def _threshold(label: str) -> float:
            return cal.threshold_for(label, CHEST_LABEL_THRESHOLDS.get(label, CHEST_THRESHOLD))

        top_findings: list[dict] = []
        for label, prob in sorted(cal_by_label.items(), key=lambda item: item[1], reverse=True):
            if prob < _threshold(label):
                continue
            finding: dict = {"label": label, "probability": prob}
            # DÃ¼ÅŸÃ¼k AUROC bulgularÄ± iÃ§in "model emin deÄŸil" iÅŸareti â€” UI uyarÄ± ekler
            if cal.is_low_confidence(label):
                finding["low_confidence"] = True
            top_findings.append(finding)
            if len(top_findings) >= 5:
                break

        # OOD skoru â€” kalibrasyon yoksa None dÃ¶ner (graceful)
        ood_score: float | None = None
        ood_is_out: bool = False
        if self._chest_ood is not None:
            try:
                ood_score, ood_is_out = self._chest_ood.score(model, pil_image)
            except Exception as exc:
                LOGGER.warning("chest OOD scoring failed: %s", exc)

        # Grad-CAM heatmap'leri â€” sadece top findings iÃ§in (max 5 PNG)
        if self._chest_heatmap is not None and top_findings:
            for finding in top_findings:
                label = finding["label"]
                try:
                    class_idx = CHEST_LABELS.index(label)
                    result = self._chest_heatmap.heatmap_for_class(
                        input_tensor=x, display_image=pil_image, target_class_idx=class_idx,
                    )
                    finding["heatmap_base64"] = result.base64_str
                except Exception as exc:
                    LOGGER.warning("chest heatmap failed | label=%s err=%s", label, exc)

        normal_probability = max(0.0, min(1.0, 1.0 - max(cal_by_label.values())))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "top_findings": top_findings,
            "normal_probability": normal_probability,
            "all_probabilities": cal_by_label,           # backward-compat alanÄ± kalibre deÄŸerlerle
            "raw_probabilities": raw_by_label,           # debugging iÃ§in ham Ã§Ä±ktÄ±lar
            "calibrated": not getattr(cal, "is_identity", False),
            "ood_score": ood_score,
            "ood_is_out_of_distribution": ood_is_out,
            "heatmap_available": self._chest_heatmap is not None,
            "inference_ms": elapsed_ms,
        }

    def _score_bone_image(self, model: torch.nn.Module, pil_image: Image.Image) -> tuple[float, torch.Tensor]:
        x = self._bone_transform(pil_image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = model(x).reshape(-1).detach().cpu()

        if self._bone_num_classes == 1:
            raw_abnormal = float(torch.sigmoid(logits[0]).item())
        elif self._bone_num_classes == 2:
            probs = torch.softmax(logits, dim=0)
            raw_abnormal = float(probs[1].item())
        else:
            raise RuntimeError(f"Unsupported bone model output size: {self._bone_num_classes}")
        return raw_abnormal, x

    def predict_bone(self, contents: bytes) -> dict:
        started = time.perf_counter()
        model = self._ensure_bone()

        # Full image can dilute small local fractures; score broad ROI crops too.
        pil_image = _load_image(contents).convert("RGB")
        bcal = self._bone_calibration

        crop_scores: list[dict[str, float | str]] = []
        best_crop_name = "full"
        best_image = pil_image
        raw_abnormal = -1.0
        abnormal_probability = -1.0
        x: torch.Tensor | None = None

        for crop_name, crop_image in _bone_crop_candidates(pil_image):
            crop_raw, crop_x = self._score_bone_image(model, crop_image)
            crop_prob = bcal.apply(crop_raw)
            crop_scores.append({
                "crop": crop_name,
                "raw_abnormal_probability": crop_raw,
                "abnormal_probability": crop_prob,
            })
            if crop_prob > abnormal_probability:
                raw_abnormal = crop_raw
                abnormal_probability = crop_prob
                best_crop_name = crop_name
                best_image = crop_image
                x = crop_x

        if x is None:
            raise RuntimeError("Bone crop ensemble produced no candidates")

        normal_probability = 1.0 - abnormal_probability

        # Threshold: kalibratÃ¶rden F1-optimal; yoksa sabit fallback
        base_threshold = bcal.threshold if not getattr(bcal, "is_identity", False) else BONE_ABNORMAL_THRESHOLD
        threshold = base_threshold if best_crop_name == "full" else min(0.95, base_threshold + 0.10)
        result = "abnormal" if abnormal_probability >= threshold else "normal"

        # Confidence band â€” kalibre olasÄ±lÄ±k Ã¼zerinden (threshold'a uzaklÄ±k)
        center_distance = abs(abnormal_probability - threshold)
        if center_distance >= 0.3:
            confidence = "high"
        elif center_distance >= 0.15:
            confidence = "medium"
        else:
            confidence = "low"
        # AUROC<0.65 olan kalibrasyonlarda zorla low â€” kullanÄ±cÄ±ya "model emin deÄŸil" mesajÄ± git
        if getattr(bcal, "is_low_confidence", False):
            confidence = "low"

        # OOD skoru â€” kalibrasyon yoksa None dÃ¶ner (graceful)
        ood_score: float | None = None
        ood_is_out: bool = False
        if self._bone_ood is not None:
            try:
                ood_score, ood_is_out = self._bone_ood.score(model, pil_image)
            except Exception as exc:
                LOGGER.warning("bone OOD scoring failed: %s", exc)

        # OOD ise model Ã§Ä±ktÄ±sÄ± gÃ¼venilir deÄŸil â€” confidence'Ä± low'a Ã§ek
        if ood_is_out:
            confidence = "low"

        # Grad-CAM heatmap â€” bone binary, "abnormal" sÄ±nÄ±fÄ± iÃ§in tek heatmap
        heatmap_b64: str | None = None
        if self._bone_heatmap is not None and result == "abnormal":
            try:
                # 1-class: target_idx=0; 2-class: pozitif sÄ±nÄ±f=1
                target_idx = 1 if self._bone_num_classes == 2 else 0
                hm = self._bone_heatmap.heatmap_for_class(
                    input_tensor=x, display_image=best_image, target_class_idx=target_idx,
                )
                heatmap_b64 = hm.base64_str
            except Exception as exc:
                LOGGER.warning("bone heatmap failed: %s", exc)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "result": result,
            "confidence": confidence,
            "abnormal_probability": abnormal_probability,       # kalibre edilmiÅŸ
            "raw_abnormal_probability": raw_abnormal,           # debug iÃ§in ham
            "normal_probability": normal_probability,
            "threshold_used": threshold,                        # hangi eÅŸik uygulandÄ±
            "calibrated": not getattr(bcal, "is_identity", False),
            "roi_ensemble": True,
            "roi_best_crop": best_crop_name,
            "roi_crop_scores": crop_scores,
            "roi_base_threshold": base_threshold,
            "ood_score": ood_score,
            "ood_is_out_of_distribution": ood_is_out,
            "heatmap_base64": heatmap_b64,
            "heatmap_available": self._bone_heatmap is not None,
            "inference_ms": elapsed_ms,
        }
