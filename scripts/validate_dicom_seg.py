from pathlib import Path

import highdicom as hd
import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


SEG_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dicom_test"
    / "dicom_seg"
    / "spleen_segmentation.dcm"
)


if not SEG_PATH.exists():

    raise FileNotFoundError(
        f"DICOM SEG not found:\n{SEG_PATH}"
    )


print("=" * 70)
print(
    "MedSegOps - DICOM SEG Validation"
)
print("=" * 70)

print()
print(
    "Reading:",
    SEG_PATH
)


seg = hd.seg.segread(
    str(SEG_PATH)
)


print()
print(
    "SOP Class:",
    seg.SOPClassUID
)

print(
    "Modality:",
    seg.Modality
)

print(
    "Segmentation Type:",
    seg.SegmentationType
)

print(
    "Number of Segments:",
    seg.number_of_segments
)

print(
    "Segment Numbers:",
    list(
        seg.segment_numbers
    )
)


# ============================================================
# SEGMENT DESCRIPTION
# ============================================================

description = (
    seg.get_segment_description(
        1
    )
)


print()
print(
    "Segment Label:",
    description.segment_label
)

print(
    "Algorithm Type:",
    description.algorithm_type
)

print(
    "Segmented Property Type:",
    description.segmented_property_type
)


# ============================================================
# SOURCE IMAGES
# ============================================================

source_uids = (
    seg.get_source_image_uids()
)


print()
print(
    "Referenced source images:",
    len(source_uids)
)


# ============================================================
# PIXELS
# ============================================================

pixels = (
    seg.get_pixels_by_source_instance(
        source_sop_instance_uids=[
            item[2]
            for item in source_uids
        ]
    )
)


print(
    "Decoded SEG pixel shape:",
    pixels.shape
)

print(
    "Unique pixel values:",
    np.unique(
        pixels
    )
)

print(
    "Spleen voxels in SEG:",
    int(
        np.count_nonzero(
            pixels
        )
    )
)


# ============================================================
# CHECK
# ============================================================

if seg.Modality != "SEG":

    raise RuntimeError(
        "Output is not a DICOM SEG object."
    )


if seg.SegmentationType != "BINARY":

    raise RuntimeError(
        "Expected BINARY segmentation."
    )


if seg.number_of_segments != 1:

    raise RuntimeError(
        "Expected exactly one segment."
    )


unique_values = np.unique(
    pixels
)

if not np.all(
    np.isin(
        unique_values,
        [0, 1],
    )
):

    raise RuntimeError(
        "SEG contains unexpected pixel values."
    )


print()
print("=" * 70)
print(
    "DICOM SEG VALIDATION SUCCESS"
)
print("=" * 70)