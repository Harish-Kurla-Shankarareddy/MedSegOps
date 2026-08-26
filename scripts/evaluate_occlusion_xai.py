from pathlib import Path

import nibabel as nib

from app.xai.alignment import (
    evaluate_explanation_alignment,
)


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ==================================================
# FILES
# ==================================================

OCCLUSION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "occlusion"
    / "spleen_occlusion_original_space.nii.gz"
)

SEGMENTATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "segmentation_for_xai.nii.gz"
)


# ==================================================
# CHECK
# ==================================================

if not OCCLUSION_PATH.exists():

    raise FileNotFoundError(
        f"Occlusion heatmap not found:\n"
        f"{OCCLUSION_PATH}"
    )


if not SEGMENTATION_PATH.exists():

    raise FileNotFoundError(
        f"Segmentation not found:\n"
        f"{SEGMENTATION_PATH}"
    )


# ==================================================
# LOAD
# ==================================================

print(
    "Loading occlusion heatmap..."
)

occlusion = nib.load(
    str(OCCLUSION_PATH)
).get_fdata()


print(
    "Occlusion shape:",
    occlusion.shape
)


print(
    "Loading segmentation..."
)

segmentation = nib.load(
    str(SEGMENTATION_PATH)
).get_fdata()


print(
    "Segmentation shape:",
    segmentation.shape
)


# ==================================================
# CHECK SHAPE
# ==================================================

if occlusion.shape != segmentation.shape:

    raise ValueError(
        "Shape mismatch:\n"
        f"Occlusion: {occlusion.shape}\n"
        f"Segmentation: {segmentation.shape}"
    )


# ==================================================
# EVALUATE
# ==================================================

print()
print("=" * 70)
print("MedSegOps - Occlusion XAI Alignment Evaluation")
print("=" * 70)


result = evaluate_explanation_alignment(
    heatmap=occlusion,
    segmentation_mask=segmentation,
    percentile=90.0,
)


# ==================================================
# PRINT
# ==================================================

print()

print(
    f"Threshold percentile: "
    f"{result['threshold_percentile']}"
)

print(
    f"Threshold value: "
    f"{result['cam_threshold']:.6f}"
)

print(
    f"High-activation voxels: "
    f"{result['high_activation_voxels']}"
)

print(
    f"Segmentation voxels: "
    f"{result['segmentation_voxels']}"
)

print(
    f"Intersection voxels: "
    f"{result['intersection_voxels']}"
)

print(
    f"Explanation Precision: "
    f"{result['explanation_precision']:.4f}"
)

print(
    f"Explanation Coverage: "
    f"{result['explanation_coverage']:.4f}"
)

print(
    f"Explanation IoU: "
    f"{result['explanation_iou']:.4f}"
)

print()
print("=" * 70)
print("OCCLUSION XAI EVALUATION COMPLETE")
print("=" * 70)