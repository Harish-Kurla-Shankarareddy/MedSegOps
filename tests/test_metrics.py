import numpy as np

from evaluation.metrics import (
    dice_score,
    iou_score,
    evaluate_segmentation
)


def test_perfect_dice_score():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [1, 1],
        [0, 0]
    ])

    score = dice_score(
        prediction,
        ground_truth
    )

    assert score == 1.0


def test_no_overlap_dice_score():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [0, 0],
        [1, 1]
    ])

    score = dice_score(
        prediction,
        ground_truth
    )

    assert score == 0.0


def test_partial_overlap_dice_score():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [1, 0],
        [1, 0]
    ])

    score = dice_score(
        prediction,
        ground_truth
    )

    assert score == 0.5


def test_perfect_iou_score():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [1, 1],
        [0, 0]
    ])

    score = iou_score(
        prediction,
        ground_truth
    )

    assert score == 1.0


def test_no_overlap_iou_score():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [0, 0],
        [1, 1]
    ])

    score = iou_score(
        prediction,
        ground_truth
    )

    assert score == 0.0


def test_partial_overlap_iou_score():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [1, 0],
        [1, 0]
    ])

    score = iou_score(
        prediction,
        ground_truth
    )

    assert score == 1 / 3


def test_evaluate_segmentation():

    prediction = np.array([
        [1, 1],
        [0, 0]
    ])

    ground_truth = np.array([
        [1, 1],
        [0, 0]
    ])

    results = evaluate_segmentation(
        prediction,
        ground_truth
    )

    assert "dice" in results
    assert "iou" in results

    assert results["dice"] == 1.0
    assert results["iou"] == 1.0