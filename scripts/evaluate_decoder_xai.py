from pathlib import Path

import nibabel as nib

from app.xai.alignment import (
    evaluate_explanation_alignment,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


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


if not CAM_PATH.exists():

    raise FileNotFoundError(
        f"Decoder Grad-CAM not found:\n{CAM_PATH}"
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
    "Decoder CAM shape:",
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


if cam.shape != segmentation.shape:

    raise ValueError(
        f"Shape mismatch: "
        f"{cam.shape} vs {segmentation.shape}"
    )


print()
print("=" * 60)
print("Decoder Grad-CAM Alignment Evaluation")
print("=" * 60)


results = evaluate_explanation_alignment(
    heatmap=cam,
    segmentation_mask=segmentation,
    percentile=90.0,
)


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
print("DECODER XAI EVALUATION COMPLETE")
print("=" * 60)