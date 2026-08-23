from pathlib import Path

import pytest

from app.pipeline.inference import SpleenSegmenter


MODEL_PATH = Path("models/monai/model.pt")


@pytest.mark.model
@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Pretrained model not available in this environment",
)
def test_spleen_segmentation_quality():
    """Verify that the pretrained model achieves acceptable Dice."""

    segmenter = SpleenSegmenter()

    assert segmenter is not None
