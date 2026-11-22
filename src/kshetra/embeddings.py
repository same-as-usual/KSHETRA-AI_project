"""Image-embedding change scores for before/after satellite composites (optional, deep-learning path).

``embedding_difference`` is framework-independent: it takes any feature extractor callable.
``torchvision_extractor`` builds a MobileNetV3-Small feature extractor, but needs PyTorch,
torchvision and a locally supplied weights file. Those are not bundled or downloaded, and the
torch path has not been executed in the sandbox where this module was written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

Extractor = Callable[[np.ndarray], np.ndarray]


def embedding_difference(before_rgb: np.ndarray, after_rgb: np.ndarray, extractor: Extractor) -> dict[str, float]:
    """Compare two image embeddings. Larger distance means more change in learned image features.

    The score is a feature-space distance, not a classification of land cover or poverty.
    """
    b = np.asarray(extractor(before_rgb), dtype=float).ravel()
    a = np.asarray(extractor(after_rgb), dtype=float).ravel()
    if b.shape != a.shape or b.size == 0:
        raise ValueError("extractor must return non-empty embeddings of identical length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("embeddings contain non-finite values")
    nb, na = np.linalg.norm(b), np.linalg.norm(a)
    cosine = float(b @ a / (nb * na)) if nb > 0 and na > 0 else float("nan")
    return {
        "embedding_dim": int(b.size),
        "l2_distance": float(np.linalg.norm(a - b)),
        "cosine_similarity": cosine,
        "cosine_distance": float(1.0 - cosine) if np.isfinite(cosine) else float("nan"),
    }


def torchvision_extractor(weights_path: Path | str, image_size: int = 224) -> Extractor:
    """Return a callable mapping an HxWx3 uint8 RGB array to a 576-dim MobileNetV3-Small embedding.

    Requires ``torch`` and ``torchvision`` and a weights file you have downloaded and verified.
    """
    try:
        import torch  # type: ignore[import-not-found]
        import torchvision  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchvision are required for the CNN embedding path") from exc
    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError(f"weights file not found: {weights}")

    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.classifier = torch.nn.Identity()  # keep pooled 576-dim features
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def extract(rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(rgb[:, :, :3])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            return model(x).squeeze(0).numpy()

    return extract
