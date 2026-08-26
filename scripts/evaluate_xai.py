from pathlib import Path

import nibabel as nib
import numpy as np

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
# INPUT FILES
# ==================================================

GRADCAM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "spleen_gradcam_original_space.nii.gz"
)

SEGMENTATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "segmentation_for_xai.nii.gz"
)


# ==================================================
# Check Grad-CAM
# ==================================================

if not GRADCAM_PATH.exists():
    raise FileNotFoundError(
        f"Grad-CAM file not found:\n{GRADCAM_PATH}"
    )


# ==================================================
# We need a predicted segmentation mask
# ==================================================
#
# For the first alignment experiment, use the
# existing segmentation produced by your model.
#
# We will create a copy in the next step if needed.
# ==================================================

if not SEGMENTATION_PATH.exists():
    raise FileNotFoundError(
        "Segmentation file for XAI was not found:\n"
        f"{SEGMENTATION_PATH}\n\n"
        "Create this file from your existing "
        "spleen segmentation output first."
    )


# ==================================================
# LOAD FILES
# ==================================================

print("Loading Grad-CAM...")

cam_nii = nib.load(
    str(GRADCAM_PATH)
)

cam = cam_nii.get_fdata()


print(
    "Grad-CAM shape:",
    cam.shape
)


print("Loading segmentation...")

seg_nii = nib.load(
    str(SEGMENTATION_PATH)
)

segmentation = seg_nii.get_fdata()


print(
    "Segmentation shape:",
    segmentation.shape
)


# ==================================================
# ALIGNMENT CHECK
# ==================================================

if cam.shape != segmentation.shape:

    raise ValueError(
        "Grad-CAM and segmentation have "
        f"different shapes: "
        f"{cam.shape} vs "
        f"{segmentation.shape}"
    )


# ==================================================
# EVALUATE
# ==================================================

print()
print("=" * 60)
print("MedSegOps - XAI Alignment Evaluation")
print("=" * 60)

results = evaluate_explanation_alignment(
    heatmap=cam,
    segmentation_mask=segmentation,
    percentile=90.0,
)


# ==================================================
# PRINT RESULTS
# ==================================================

print()

print(
    f"CAM threshold percentile: "
    f"{results['threshold_percentile']}"
)

print(
    f"CAM threshold value: "
    f"{results['cam_threshold']:.6f}"
)

print(
    f"High-activation voxels: "
    f"{results['high_activation_voxels']}"
)

print(
    f"Segmentation voxels: "
    f"{results['segmentation_voxels']}"
)

print(
    f"Intersection voxels: "
    f"{results['intersection_voxels']}"
)

print(
    f"Explanation Precision: "
    f"{results['explanation_precision']:.4f}"
)

print(
    f"Explanation Coverage: "
    f"{results['explanation_coverage']:.4f}"
)

print(
    f"Explanation IoU: "
    f"{results['explanation_iou']:.4f}"
)


print()
print("=" * 60)
print("XAI EVALUATION COMPLETE")
print("=" * 60)