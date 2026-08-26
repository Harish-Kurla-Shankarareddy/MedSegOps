from pathlib import Path

from app.io.dicom_seg import (
    DicomSegError,
    create_dicom_seg,
)


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# INPUTS
# ============================================================

DICOM_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dicom_test"
)

SEGMENTATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dicom_test"
    / "segmentation"
    / "converted_seg.nii.gz"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dicom_test"
    / "dicom_seg"
    / "spleen_segmentation.dcm"
)


# ============================================================
# RUN
# ============================================================

print("=" * 70)
print(
    "MedSegOps - DICOM SEG Export Test"
)
print("=" * 70)

print()
print(
    "DICOM source:",
    DICOM_DIRECTORY,
)

print(
    "Segmentation:",
    SEGMENTATION_PATH,
)

print(
    "DICOM SEG output:",
    OUTPUT_PATH,
)


try:

    result = create_dicom_seg(
        dicom_directory=DICOM_DIRECTORY,
        segmentation_path=SEGMENTATION_PATH,
        output_path=OUTPUT_PATH,
    )

except DicomSegError as error:

    print()
    print(
        "DICOM SEG ERROR:"
    )

    print(
        error
    )

    raise SystemExit(1)


print()
print("=" * 70)
print(
    "DICOM SEG CREATION SUCCESS"
)
print("=" * 70)

print(
    "Output:",
    result["output_path"],
)

print(
    "Source DICOM files:",
    result["source_file_count"],
)

print(
    "Mask shape:",
    result["mask_shape"],
)

print(
    "Spleen voxels:",
    result["spleen_voxels"],
)

print(
    "Segment:",
    result["segment_label"],
)

print(
    "Segment number:",
    result["segment_number"],
)

print(
    "Type:",
    result["segmentation_type"],
)