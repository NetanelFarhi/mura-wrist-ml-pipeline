"""Regression checks for the masked Phase 2C study-level classifier."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.study_mean_classifier import StudyMeanEmbeddingClassifier


class _TinyBackbone(nn.Module):
    """Fast ResNet-shaped stand-in that exposes ``fc`` for unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Identity()
        self.bn1 = nn.Identity()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.feature = nn.Linear(3 * 2 * 2, 4)
        self.fc = nn.Linear(4, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.feature(images.reshape(images.shape[0], -1)))


class StudyMeanEmbeddingClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = StudyMeanEmbeddingClassifier(_TinyBackbone())
        self.model.configure_trainable_layers(train_layer4=False)
        self.model.eval()

    def test_padded_pixels_have_zero_influence(self) -> None:
        real = torch.randn(1, 2, 3, 2, 2)
        padded_a = torch.zeros(1, 1, 3, 2, 2)
        padded_b = torch.full((1, 1, 3, 2, 2), 999.0)
        mask = torch.tensor([[True, True, False]])
        with torch.inference_mode():
            score_a = self.model(torch.cat([real, padded_a], dim=1), mask)
            score_b = self.model(torch.cat([real, padded_b], dim=1), mask)
        self.assertTrue(torch.allclose(score_a, score_b, atol=1e-6))

    def test_view_order_does_not_change_mean_prediction(self) -> None:
        images = torch.randn(1, 3, 3, 2, 2)
        mask = torch.tensor([[True, True, True]])
        permutation = torch.tensor([2, 0, 1])
        with torch.inference_mode():
            original = self.model(images, mask)
            reordered = self.model(images[:, permutation], mask[:, permutation])
        self.assertTrue(torch.allclose(original, reordered, atol=1e-6))

    def test_single_view_and_variable_view_batches_work(self) -> None:
        images = torch.randn(2, 3, 3, 2, 2)
        mask = torch.tensor([[True, False, False], [True, True, True]])
        with torch.inference_mode():
            logits = self.model(images, mask)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_empty_study_is_rejected(self) -> None:
        images = torch.randn(1, 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            self.model(images, torch.tensor([[False, False]]))


if __name__ == "__main__":
    unittest.main()
