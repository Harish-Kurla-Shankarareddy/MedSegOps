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

CAM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "spleen_gradcam_decoder_original_space.nii.gz"
)

SEGMENTATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "segmentation_for_xai.nii.gz"
)


# ==================================================
# LOAD DATA
# ==================================================

if not CAM_PATH.exists():
    raise FileNotFoundError(
        f"Grad-CAM not found:\n{CAM_PATH}"
    )


if not SEGMENTATION_PATH.exists():
    raise FileNotFoundError(
        f"Segmentation not found:\n"
        f"{SEGMENTATION_PATH}"
    )


print("Loading decoder Grad-CAM...")

cam = nib.load(
    str(CAM_PATH)
).get_fdata()


print(
    "CAM shape:",
    cam.shape
)


print("Loading segmentation...")

segmentation = nib.load(
    str(SEGMENTATION_PATH)
).get_fdata()


print(
    "Segmentation shape:",
    segmentation.shape
)


# ==================================================
# CHECK
# ==================================================

if cam.shape != segmentation.shape:
    raise ValueError(
        f"Shape mismatch: "
        f"{cam.shape} vs "
        f"{segmentation.shape}"
    )


# ==================================================
# THRESHOLD SWEEP
# ==================================================

thresholds = [
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
]


print()
print("=" * 80)
print("MedSegOps - Grad-CAM Threshold Sensitivity")
print("=" * 80)

print()

print(
    f"{'Threshold':>12}"
    f"{'Precision':>15}"
    f"{'Coverage':>15}"
    f"{'IoU':>15}"
)

print("-" * 80)


results = []


for threshold in thresholds:

    result = evaluate_explanation_alignment(
        heatmap=cam,
        segmentation_mask=segmentation,
        percentile=threshold,
    )

    results.append(result)

    print(
        f"{threshold:>11.0f}%"
        f"{result['explanation_precision']:>15.4f}"
        f"{result['explanation_coverage']:>15.4f}"
        f"{result['explanation_iou']:>15.4f}"
    )


print()
print("=" * 80)
print("Threshold sweep complete.")
print("=" * 80)


# ==================================================
# FIND BEST IOU
# ==================================================

best = max(
    results,
    key=lambda item: item["explanation_iou"],
)


print()

print(
    "Best IoU threshold:",
    best["threshold_percentile"],
)

print(
    "Best explanation IoU:",
    best["explanation_iou"],
)

print(
    "Precision at best threshold:",
    best["explanation_precision"],
)

print(
    "Coverage at best threshold:",
    best["explanation_coverage"],
)