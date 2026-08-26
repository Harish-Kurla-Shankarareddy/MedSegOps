from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# INPUT
# ============================================================

INPUT_NIFTI = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spleen"
    / "Task09_Spleen"
    / "imagesTr"
    / "spleen_10.nii.gz"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DICOM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dicom_test"
)


OUTPUT_DICOM.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD ORIGINAL NIFTI
# ============================================================

print("=" * 70)
print("MedSegOps - Create Test DICOM Series")
print("=" * 70)

print()
print(
    "Loading NIfTI:"
)
print(
    INPUT_NIFTI
)

image = sitk.ReadImage(
    str(INPUT_NIFTI)
)


print()
print(
    "Original image size:",
    image.GetSize(),
)

print(
    "Original pixel type:",
    image.GetPixelIDTypeAsString(),
)

print(
    "Original spacing:",
    image.GetSpacing(),
)

print(
    "Original direction:",
    image.GetDirection(),
)


# ============================================================
# SAVE ORIGINAL DIRECTION
# ============================================================

direction = image.GetDirection()

# SimpleITK stores a 3x3 direction matrix:
#
# [ d00 d01 d02 ]
# [ d10 d11 d12 ]
# [ d20 d21 d22 ]
#
# DICOM ImageOrientationPatient contains:
#
# first row direction:
#   d00, d10, d20
#
# first column direction:
#   d01, d11, d21

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


image_orientation = (
    "\\".join(
        str(value)
        for value in (
            row_direction
            + column_direction
        )
    )
)


print()
print(
    "DICOM ImageOrientationPatient:"
)
print(
    image_orientation
)


# ============================================================
# PIXEL TYPE
# ============================================================

#
# CT DICOM commonly stores signed integer pixels.
#
# For this synthetic test dataset we use Int16.
#
# The transformation is intentionally kept simple:
#
# stored_value = original_value
# slope        = 1
# intercept    = 0
#
# This preserves the original intensity scale as long
# as the source values fit inside Int16.
#

if image.GetPixelID() != sitk.sitkInt16:

    print()
    print(
        "Converting source image to Int16..."
    )

    image = sitk.Cast(
        image,
        sitk.sitkInt16,
    )


print(
    "DICOM pixel type:",
    image.GetPixelIDTypeAsString(),
)


# ============================================================
# DICOM IDENTIFIERS
# ============================================================

study_uid = (
    "1.2.826.0.1.3680043.10.999."
    "123456789"
)

series_uid = (
    "1.2.826.0.1.3680043.10.999."
    "123456789.1"
)

frame_of_reference_uid = (
    "1.2.826.0.1.3680043.10.999."
    "123456789.2"
)


# ============================================================
# WRITER
# ============================================================

writer = sitk.ImageFileWriter()

writer.KeepOriginalImageUIDOn()


depth = image.GetDepth()

spacing = image.GetSpacing()

size = image.GetSize()


print()
print(
    "Creating DICOM series..."
)

print(
    "Number of slices:",
    depth,
)


# ============================================================
# WRITE SLICES
# ============================================================

for index in range(depth):

    slice_image = image[
        :,
        :,
        index,
    ]


    # ========================================================
    # GENERAL DICOM INFORMATION
    # ========================================================

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


    # ========================================================
    # STUDY / SERIES
    # ========================================================

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
        frame_of_reference_uid,
    )


    # ========================================================
    # SYNTHETIC PATIENT
    # ========================================================

    slice_image.SetMetaData(
        "0010|0010",
        "MEDSEGOPS^TEST",
    )

    slice_image.SetMetaData(
        "0010|0020",
        "TEST001",
    )


    # ========================================================
    # STUDY / SERIES DESCRIPTION
    # ========================================================

    slice_image.SetMetaData(
        "0008|1030",
        "MedSegOps Test Study",
    )

    slice_image.SetMetaData(
        "0008|103e",
        "MedSegOps Test CT",
    )


    # ========================================================
    # SERIES / INSTANCE NUMBERS
    # ========================================================

    slice_image.SetMetaData(
        "0020|0011",
        "1",
    )

    slice_image.SetMetaData(
        "0020|0013",
        str(index + 1),
    )


    # ========================================================
    # PRESERVE ORIGINAL ORIENTATION
    # ========================================================

    slice_image.SetMetaData(
        "0020|0037",
        image_orientation,
    )


    # ========================================================
    # PRESERVE PHYSICAL POSITION
    # ========================================================

    physical_point = (
        image.TransformIndexToPhysicalPoint(
            (
                0,
                0,
                index,
            )
        )
    )


    image_position = (
        "\\".join(
            str(value)
            for value in physical_point
        )
    )


    slice_image.SetMetaData(
        "0020|0032",
        image_position,
    )


    # ========================================================
    # SPACING
    # ========================================================

    slice_image.SetMetaData(
        "0028|0030",
        f"{spacing[1]}\\{spacing[0]}",
    )

    slice_image.SetMetaData(
        "0018|0050",
        str(
            spacing[2]
        ),
    )


    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    slice_image.SetMetaData(
        "0028|0010",
        str(
            size[1]
        ),
    )

    slice_image.SetMetaData(
        "0028|0011",
        str(
            size[0]
        ),
    )


    # ========================================================
    # PIXEL FORMAT
    # ========================================================

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


    # ========================================================
    # RESCALE
    # ========================================================

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


    # ========================================================
    # WRITE
    # ========================================================

    filename = (
        OUTPUT_DICOM
        / f"slice_{index:04d}.dcm"
    )

    writer.SetFileName(
        str(filename)
    )

    writer.Execute(
        slice_image
    )


    if (
        index == 0
        or (index + 1) % 10 == 0
        or index == depth - 1
    ):

        print(
            f"Wrote slice "
            f"{index + 1}/{depth}"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("DICOM TEST SERIES CREATED")
print("=" * 70)

print(
    "Directory:",
    OUTPUT_DICOM,
)

print(
    "Slices:",
    depth,
)

print(
    "Size:",
    image.GetSize(),
)

print(
    "Spacing:",
    image.GetSpacing(),
)

print(
    "Direction:",
    image.GetDirection(),
)