import numpy as np
import pytest

from evaluation.metrics import dice_score, mask_volume_ml


def test_dice_score_perfect_match():
    prediction = np.array([1, 1, 0, 0])
    ground_truth = np.array([1, 1, 0, 0])

    score = dice_score(prediction, ground_truth)

    assert score == 1.0


def test_dice_score_no_overlap():
    prediction = np.array([1, 1, 0, 0])
    ground_truth = np.array([0, 0, 1, 1])

    score = dice_score(prediction, ground_truth)

    assert score == 0.0


def test_dice_score_partial_overlap():
    prediction = np.array([1, 1, 0, 0])
    ground_truth = np.array([1, 0, 1, 0])

    score = dice_score(prediction, ground_truth)

    assert score == pytest.approx(0.5)


def test_dice_score_empty_masks():
    prediction = np.zeros(4)
    ground_truth = np.zeros(4)

    score = dice_score(prediction, ground_truth)

    assert score == 1.0


def test_mask_volume_ml():
    mask = np.ones((10, 10, 10))

    spacing = (1.0, 1.0, 1.0)

    volume = mask_volume_ml(mask, spacing)

    assert volume == pytest.approx(1.0)