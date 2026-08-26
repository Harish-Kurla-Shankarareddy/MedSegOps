from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import pytest


@pytest.fixture
def synthetic_dicom_series(
    tmp_path: Path,
) -> Path:
    """
    Create one deterministic synthetic CT DICOM series.

    All slices share the same StudyInstanceUID and
    SeriesInstanceUID so GDCM discovers exactly one
    CT series containing all slices.
    """

    dicom_directory = (
        tmp_path / "dicom_series"
    )

    dicom_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Synthetic 3D CT volume
    # --------------------------------------------------------

    image = sitk.Image(
        64,
        64,
        8,
        sitk.sitkInt16,
    )

    image.SetSpacing(
        (
            1.0,
            1.0,
            2.5,
        )
    )

    image.SetOrigin(
        (
            0.0,
            0.0,
            0.0,
        )
    )

    image.SetDirection(
        (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )

    # Deterministic voxel values.
    for z in range(8):
        for y in range(64):
            for x in range(64):
                value = (
                    x
                    + y
                    + z * 10
                )

                image.SetPixel(
                    x,
                    y,
                    z,
                    int(value),
                )

    # --------------------------------------------------------
    # Fixed DICOM UIDs
    # --------------------------------------------------------

    study_uid = (
        "1.2.826.0.1.3680043.10.999."
        "111111111"
    )

    series_uid = (
        "1.2.826.0.1.3680043.10.999."
        "111111111.1"
    )

    frame_uid = (
        "1.2.826.0.1.3680043.10.999."
        "111111111.2"
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    direction = image.GetDirection()

    row_direction = (
        direction[0],
        direction[3],
        direction[6],
    )

    column_direction = (
        direction[1],
        direction[4],
        direction[7],
    )

    orientation = "\\".join(
        str(value)
        for value in (
            row_direction
            + column_direction
        )
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    spacing = image.GetSpacing()
    size = image.GetSize()

    writer = sitk.ImageFileWriter()

    # CRITICAL:
    # Keep one SeriesInstanceUID for all slices.
    #
    # Also explicitly disable keeping any accidental
    # original UID from the temporary image.
    writer.KeepOriginalImageUIDOn()

    # --------------------------------------------------------
    # Write all slices
    # --------------------------------------------------------

    for index in range(
        image.GetDepth()
    ):

        slice_image = image[
            :,
            :,
            index,
        ]

        # --------------------------------------------
        # Modality / SOP
        # --------------------------------------------

        slice_image.SetMetaData(
            "0008|0060",
            "CT",
        )

        slice_image.SetMetaData(
            "0008|0008",
            "DERIVED\\SECONDARY",
        )

        slice_image.SetMetaData(
            "0008|0016",
            "1.2.840.10008.5.1.4.1.1.2",
        )

        # --------------------------------------------
        # Study / Series
        # --------------------------------------------

        slice_image.SetMetaData(
            "0020|000d",
            study_uid,
        )

        slice_image.SetMetaData(
            "0020|000e",
            series_uid,
        )

        slice_image.SetMetaData(
            "0020|0052",
            frame_uid,
        )

        # --------------------------------------------
        # Patient
        # --------------------------------------------

        slice_image.SetMetaData(
            "0010|0010",
            "MEDSEGOPS^TEST",
        )

        slice_image.SetMetaData(
            "0010|0020",
            "TEST001",
        )

        # --------------------------------------------
        # Description
        # --------------------------------------------

        slice_image.SetMetaData(
            "0008|1030",
            "MedSegOps CI Test",
        )

        slice_image.SetMetaData(
            "0008|103e",
            "Synthetic CT Series",
        )

        # --------------------------------------------
        # Series/instance numbers
        # --------------------------------------------

        slice_image.SetMetaData(
            "0020|0011",
            "1",
        )

        slice_image.SetMetaData(
            "0020|0013",
            str(index + 1),
        )

        # --------------------------------------------
        # Orientation
        # --------------------------------------------

        slice_image.SetMetaData(
            "0020|0037",
            orientation,
        )

        # --------------------------------------------
        # Position
        # --------------------------------------------

        physical_point = (
            image.TransformIndexToPhysicalPoint(
                (
                    0,
                    0,
                    index,
                )
            )
        )

        slice_image.SetMetaData(
            "0020|0032",
            "\\".join(
                str(value)
                for value in physical_point
            ),
        )

        # --------------------------------------------
        # Pixel spacing
        # --------------------------------------------

        slice_image.SetMetaData(
            "0028|0030",
            f"{spacing[1]}\\{spacing[0]}",
        )

        slice_image.SetMetaData(
            "0018|0050",
            str(spacing[2]),
        )

        # --------------------------------------------
        # Dimensions
        # --------------------------------------------

        slice_image.SetMetaData(
            "0028|0010",
            str(size[1]),
        )

        slice_image.SetMetaData(
            "0028|0011",
            str(size[0]),
        )

        # --------------------------------------------
        # Pixel representation
        # --------------------------------------------

        slice_image.SetMetaData(
            "0028|0002",
            "1",
        )

        slice_image.SetMetaData(
            "0028|0004",
            "MONOCHROME2",
        )

        slice_image.SetMetaData(
            "0028|0100",
            "16",
        )

        slice_image.SetMetaData(
            "0028|0101",
            "16",
        )

        slice_image.SetMetaData(
            "0028|0102",
            "15",
        )

        slice_image.SetMetaData(
            "0028|0103",
            "1",
        )

        # --------------------------------------------
        # Rescale
        # --------------------------------------------

        slice_image.SetMetaData(
            "0028|1052",
            "0",
        )

        slice_image.SetMetaData(
            "0028|1053",
            "1",
        )

        slice_image.SetMetaData(
            "0028|1054",
            "HU",
        )

        # --------------------------------------------
        # CRITICAL:
        # Explicitly set the SeriesInstanceUID again
        # immediately before writing.
        # --------------------------------------------

        slice_image.SetMetaData(
            "0020|000e",
            series_uid,
        )

        output_file = (
            dicom_directory
            / f"slice_{index:04d}.dcm"
        )

        writer.SetFileName(
            str(output_file)
        )

        writer.Execute(
            slice_image
        )

    return dicom_directory