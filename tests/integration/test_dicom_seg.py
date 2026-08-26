from pathlib import Path

import highdicom as hd
import nibabel as nib
import numpy as np
import pytest

from app.io.dicom import (
    convert_dicom_series_to_nifti,
)

from app.io.dicom_seg import (
    create_dicom_seg,
)


@pytest.mark.integration
def test_dicom_seg_creation_and_validation(
    synthetic_dicom_series: Path,
    tmp_path: Path,
):
    """
    Verify the complete DICOM SEG pipeline without
    depending on generated files from the local machine.

    Flow:

        synthetic DICOM
            ↓
        DICOM → NIfTI
            ↓
        synthetic binary segmentation mask
            ↓
        DICOM SEG
            ↓
        read-back validation
    """

    # ========================================================
    # Convert DICOM to NIfTI
    # ========================================================

    converted_nifti = (
        tmp_path
        / "converted.nii.gz"
    )

    result = (
        convert_dicom_series_to_nifti(
            synthetic_dicom_series,
            converted_nifti,
        )
    )

    assert result["modality"] == "CT"

    assert result["file_count"] == 8

    # ========================================================
    # Create deterministic synthetic mask
    # ========================================================

    source_nii = nib.load(
        str(converted_nifti)
    )

    source_data = (
        source_nii.get_fdata()
    )

    assert source_data.shape == (
        64,
        64,
        8,
    )

    mask = np.zeros(
        source_data.shape,
        dtype=np.uint8,
    )

    # A deterministic central 3D block.
    mask[
        20:40,
        20:40,
        2:6,
    ] = 1

    expected_voxels = int(
        np.count_nonzero(mask)
    )

    assert expected_voxels > 0

    segmentation_path = (
        tmp_path
        / "synthetic_segmentation.nii.gz"
    )

    segmentation_nii = nib.Nifti1Image(
        mask,
        source_nii.affine,
        header=source_nii.header.copy(),
    )

    nib.save(
        segmentation_nii,
        str(segmentation_path),
    )

    # ========================================================
    # Create DICOM SEG
    # ========================================================

    output_path = (
        tmp_path
        / "spleen_segmentation.dcm"
    )

    result = create_dicom_seg(
        dicom_directory=(
            synthetic_dicom_series
        ),
        segmentation_path=(
            segmentation_path
        ),
        output_path=output_path,
    )

    assert output_path.exists()

    assert result[
        "source_file_count"
    ] == 8

    assert result[
        "segment_label"
    ] == "Spleen"

    assert result[
        "segment_number"
    ] == 1

    assert result[
        "segmentation_type"
    ] == "BINARY"

    assert result[
        "spleen_voxels"
    ] == expected_voxels

    # ========================================================
    # Read DICOM SEG back
    # ========================================================

    seg = hd.seg.segread(
        str(output_path)
    )

    assert seg.Modality == "SEG"

    assert (
        seg.SegmentationType
        == "BINARY"
    )

    assert (
        seg.number_of_segments
        == 1
    )

    description = (
        seg.get_segment_description(
            1
        )
    )

    assert (
        description.segment_label
        == "Spleen"
    )

    # ========================================================
    # Source-image references
    # ========================================================

    source_uids = (
        seg.get_source_image_uids()
    )

    assert len(
        source_uids
    ) == 8

    # ========================================================
    # Decode SEG pixels
    # ========================================================

    pixels = (
        seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[
                item[2]
                for item in source_uids
            ]
        )
    )

    assert pixels.shape == (
        8,
        64,
        64,
        1,
    )

    unique_values = np.unique(
        pixels
    )

    assert np.array_equal(
        unique_values,
        np.array([0, 1]),
    )

    assert (
        int(
            np.count_nonzero(
                pixels
            )
        )
        == expected_voxels
    )