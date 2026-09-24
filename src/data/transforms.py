"""Reproducible, medically conservative MURA XR_WRIST preprocessing.

All images are decoded as grayscale and replicated to three channels for an
ImageNet-pretrained ResNet-18.  Normalisation uses fixed ImageNet constants;
there is no dataset-statistics fitting step that could touch validation/test
images.  Evaluation transforms are deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageEnhance, ImageOps
    except ModuleNotFoundError as exc:  # pragma: no cover - installation dependent.
        raise RuntimeError("Pillow is required for XR_WRIST preprocessing; install pillow>=10.") from exc
    return Image, ImageEnhance, ImageOps


@dataclass
class MURAXRayTransform:
    """Aspect-preserving radiograph transform with optional safe augmentation.

    Rotation, brightness, and contrast ranges are intentionally modest.  The
    transform does not flip laterality, crop anatomy, inject noise, or perform
    random augmentation when ``train`` is false.
    """

    train: bool
    image_size: int = 224
    seed: int | None = None
    rotation_degrees: float = 7.0
    brightness_jitter: float = 0.08
    contrast_jitter: float = 0.10
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16 pixels")
        if self.rotation_degrees < 0 or self.brightness_jitter < 0 or self.contrast_jitter < 0:
            raise ValueError("augmentation ranges must be non-negative")
        if len(self.mean) != 3 or len(self.std) != 3 or any(value <= 0 for value in self.std):
            raise ValueError("mean and std must each contain three channels; std values must be positive")
        self._rng = random.Random(self.seed)

    def set_seed(self, seed: int | None) -> None:
        """Reset augmentation sequence for reproducible single-worker runs."""

        self.seed = seed
        self._rng = random.Random(seed)

    @staticmethod
    def _pad_to_square(image: Any, image_ops: Any) -> Any:
        width, height = image.size
        if width == height:
            return image
        size = max(width, height)
        left = (size - width) // 2
        top = (size - height) // 2
        right = size - width - left
        bottom = size - height - top
        return image_ops.expand(image, border=(left, top, right, bottom), fill=0)

    def _augment(self, image: Any, image_module: Any, image_enhance: Any) -> Any:
        if not self.train:
            return image
        if self.rotation_degrees:
            angle = self._rng.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(
                angle,
                resample=image_module.Resampling.BILINEAR,
                expand=False,
                fillcolor=0,
            )
        if self.brightness_jitter:
            factor = self._rng.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            image = image_enhance.Brightness(image).enhance(factor)
        if self.contrast_jitter:
            factor = self._rng.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            image = image_enhance.Contrast(image).enhance(factor)
        return image

    def _to_three_channel_tensor_or_image(self, grayscale_image: Any, image_module: Any) -> Any:
        rgb = image_module.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))
        try:
            import torch
        except ModuleNotFoundError:
            # Useful for visual inspection and CSV-only smoke tests.  Training
            # requires torch and will receive a tensor once it is installed.
            return rgb

        width, height = rgb.size
        # torch.tensor(bytearray(...)) avoids a NumPy dependency and creates an
        # owned buffer, unlike frombuffer on immutable image bytes.
        tensor = torch.tensor(bytearray(rgb.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(height, width, 3).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return (tensor - mean) / std

    def __call__(self, image: Any) -> Any:
        image_module, image_enhance, image_ops = _require_pillow()
        if not isinstance(image, image_module.Image):
            raise TypeError("MURAXRayTransform expects a PIL.Image.Image")
        grayscale = image.convert("L")
        squared = self._pad_to_square(grayscale, image_ops)
        augmented = self._augment(squared, image_module, image_enhance)
        resized = augmented.resize((self.image_size, self.image_size), resample=image_module.Resampling.BILINEAR)
        return self._to_three_channel_tensor_or_image(resized, image_module)


def build_transforms(
    train: bool,
    *,
    image_size: int = 224,
    seed: int | None = None,
) -> MURAXRayTransform:
    """Build a train or deterministic evaluation preprocessing transform."""

    return MURAXRayTransform(train=train, image_size=image_size, seed=seed)


def build_train_transform(*, image_size: int = 224, seed: int | None = None) -> MURAXRayTransform:
    """Convenience constructor for safe training-time augmentation."""

    return build_transforms(True, image_size=image_size, seed=seed)


def build_eval_transform(*, image_size: int = 224) -> MURAXRayTransform:
    """Convenience constructor for deterministic validation/test preprocessing."""

    return build_transforms(False, image_size=image_size)


def preprocessing_spec(image_size: int = 224) -> dict[str, object]:
    """Machine-readable documentation for reports/config snapshots."""

    return {
        "input_resolution": [image_size, image_size],
        "grayscale_handling": "decode L then replicate to RGB (3 channels)",
        "resize": "aspect-preserving black pad to square, then bilinear resize",
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "fitted_on": "none"},
        "train_augmentation": {
            "rotation_degrees": 7.0,
            "brightness_jitter": 0.08,
            "contrast_jitter": 0.10,
            "excluded": ["horizontal_flip", "random_crop", "noise_injection"],
        },
        "evaluation_augmentation": "none",
    }

