from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nibabel.processing import resample_from_to


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


CT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spleen"
    / "Task09_Spleen"
    / "imagesTr"
    / "spleen_10.nii.gz"
)


CAM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "spleen_gradcam.nii.gz"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "gradcam_comparison.png"
)


RESAMPLED_CAM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "spleen_gradcam_original_space.nii.gz"
)


# ==================================================
# CHECK FILES
# ==================================================

if not CT_PATH.exists():

    raise FileNotFoundError(
        f"CT not found: {CT_PATH}"
    )


if not CAM_PATH.exists():

    raise FileNotFoundError(
        f"Grad-CAM not found: {CAM_PATH}"
    )


# ==================================================
# LOAD IMAGES
# ==================================================

print("Loading original CT...")

ct_nii = nib.load(
    str(CT_PATH)
)


print("Loading Grad-CAM...")

cam_nii = nib.load(
    str(CAM_PATH)
)


print(
    "Original CT shape:",
    ct_nii.shape
)


print(
    "Grad-CAM shape:",
    cam_nii.shape
)


# ==================================================
# RESAMPLE CAM TO ORIGINAL CT
# ==================================================

print(
    "Resampling Grad-CAM back to original CT space..."
)


target = (
    ct_nii.shape,
    ct_nii.affine,
)


resampled_cam_nii = (
    resample_from_to(
        cam_nii,
        target,
        order=1,
    )
)


nib.save(
    resampled_cam_nii,
    str(RESAMPLED_CAM_PATH),
)


print(
    "Resampled Grad-CAM shape:",
    resampled_cam_nii.shape
)


print(
    "Saved:",
    RESAMPLED_CAM_PATH
)


# ==================================================
# GET DATA
# ==================================================

ct = ct_nii.get_fdata()

cam = resampled_cam_nii.get_fdata()


print(
    "CT shape:",
    ct.shape
)


print(
    "CAM shape:",
    cam.shape
)


# ==================================================
# NORMALIZE CAM
# ==================================================

cam = np.nan_to_num(
    cam,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


cam_min = cam.min()

cam_max = cam.max()


if cam_max > cam_min:

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min
    )

else:

    cam = np.zeros_like(
        cam
    )


# ==================================================
# FIND MOST ACTIVE SLICE
# ==================================================

slice_scores = cam.mean(
    axis=(0, 1)
)


slice_index = int(
    np.argmax(
        slice_scores
    )
)


print(
    "Most active original CT slice:",
    slice_index
)


# ==================================================
# EXTRACT SLICES
# ==================================================

ct_slice = (
    ct[
        :,
        :,
        slice_index
    ]
)


cam_slice = (
    cam[
        :,
        :,
        slice_index
    ]
)


# ==================================================
# CT DISPLAY NORMALIZATION
# ==================================================

lower = np.percentile(
    ct_slice,
    1,
)


upper = np.percentile(
    ct_slice,
    99,
)


if upper <= lower:

    upper = lower + 1


ct_slice = np.clip(
    ct_slice,
    lower,
    upper,
)


# ==================================================
# VISUALIZATION
# ==================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)


# ----------------------------------------------
# Original CT
# ----------------------------------------------

axes[0].imshow(
    np.rot90(
        ct_slice
    ),
    cmap="gray",
)


axes[0].set_title(
    f"Original CT\nSlice {slice_index}"
)


axes[0].axis("off")


# ----------------------------------------------
# Grad-CAM overlay
# ----------------------------------------------

axes[1].imshow(
    np.rot90(
        ct_slice
    ),
    cmap="gray",
)


axes[1].imshow(
    np.rot90(
        cam_slice
    ),
    cmap="jet",
    alpha=0.5,
    vmin=0,
    vmax=1,
)


axes[1].set_title(
    "CT + 3D Grad-CAM"
)


axes[1].axis("off")


# ----------------------------------------------
# Heatmap alone
# ----------------------------------------------

axes[2].imshow(
    np.rot90(
        cam_slice
    ),
    cmap="jet",
    vmin=0,
    vmax=1,
)


axes[2].set_title(
    "Grad-CAM Heatmap"
)


axes[2].axis("off")


plt.tight_layout()


plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)


plt.close()


print(
    "Comparison saved:",
    OUTPUT_PATH
)