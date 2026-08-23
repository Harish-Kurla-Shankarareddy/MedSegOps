from pathlib import Path

import pytest

from app.pipeline.inference import SpleenSegmenter
from evaluation.metrics import dice_score, load_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data/raw/spleen/Task09_Spleen/imagesTr/spleen_10.nii.gz"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/spleen/Task09_Spleen/labelsTr/spleen_10.nii.gz"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/test_model"


@pytest.mark.model
def test_spleen_segmentation_quality():
    """Verify that the pretrained model achieves acceptable Dice."""

    segmenter = SpleenSegmenter()

    segmenter.predict(
        IMAGE_PATH,
        OUTPUT_DIR,
    )

    prediction_path = OUTPUT_DIR / "spleen_10_seg.nii.gz"

    assert prediction_path.exists(), (
        f"Prediction was not created: {prediction_path}"
    )

    prediction, _ = load_mask(prediction_path)
    ground_truth, _ = load_mask(LABEL_PATH)

    assert prediction.shape == ground_truth.shape

    dice = dice_score(prediction, ground_truth)

    print(f"\nDice score: {dice:.4f}")

    # Conservative quality threshold for this regression test.
    assert dice > 0.90, (
        f"Segmentation quality too low. Dice = {dice:.4f}"
    )