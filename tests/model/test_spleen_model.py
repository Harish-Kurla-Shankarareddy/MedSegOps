from pathlib import Path

import pytest

from app.pipeline.inference import SpleenSegmenter


MODEL_PATH = Path("models/monai/model.pt")


@pytest.mark.model
@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Pretrained model not available in this environment",
)
def test_spleen_model_loads():
    """
    Verify that the pretrained segmentation model can be
    loaded successfully.
    """

    segmenter = SpleenSegmenter(
        model_path=MODEL_PATH,
        device="cpu",
    )

    assert segmenter.model is not None