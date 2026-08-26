from pathlib import Path

import nibabel as nib
import pytest

from app.io.dicom import (
    DicomSeriesError,
    convert_dicom_series_to_nifti,
)


@pytest.mark.integration
def test_dicom_series_conversion(
    synthetic_dicom_series: Path,
    tmp_path: Path,
):
    """
    Verify that a DICOM CT series can be converted
    into a 3D NIfTI volume.
    """

    output_path = (
        tmp_path
        / "converted.nii.gz"
    )

    try:

        result = (
            convert_dicom_series_to_nifti(
                synthetic_dicom_series,
                output_path,
            )
        )

    except DicomSeriesError as error:

        pytest.fail(
            f"DICOM conversion failed: {error}"
        )

    assert output_path.exists()

    assert result["modality"] == "CT"

    assert result["file_count"] == 8

    assert result["dimensions"] == {
        "x": 64,
        "y": 64,
        "z": 8,
    }

    image = nib.load(
        str(output_path)
    )

    assert image.shape == (
        64,
        64,
        8,
    )

    spacing = (
        image.header.get_zooms()[:3]
    )

    assert spacing[0] == pytest.approx(
        1.0,
        abs=1e-4,
    )

    assert spacing[1] == pytest.approx(
        1.0,
        abs=1e-4,
    )

    assert spacing[2] == pytest.approx(
        2.5,
        abs=1e-4,
    )