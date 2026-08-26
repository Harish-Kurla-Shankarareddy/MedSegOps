from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nibabel.processing import resample_from_to


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

CT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spleen"
    / "Task09_Spleen"
    / "imagesTr"
    / "spleen_10.nii.gz"
)

DECODER_CAM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "spleen_gradcam_decoder.nii.gz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
)

RESAMPLED_CAM_PATH = (
    OUTPUT_DIR
    / "spleen_gradcam_decoder_original_space.nii.gz"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "decoder_gradcam_comparison.png"
)


# ==================================================
# CHECK FILES
# ==================================================

if not CT_PATH.exists():

    raise FileNotFoundError(
        f"CT not found:\n{CT_PATH}"
    )


if not DECODER_CAM_PATH.exists():

    raise FileNotFoundError(
        f"Decoder Grad-CAM not found:\n"
        f"{DECODER_CAM_PATH}"
    )


# ==================================================
# LOAD ORIGINAL CT
# ==================================================

print("Loading original CT...")

ct_nii = nib.load(
    str(CT_PATH)
)

print(
    "Original CT shape:",
    ct_nii.shape
)


# ==================================================
# LOAD DECODER GRAD-CAM
# ==================================================

print(
    "Loading decoder Grad-CAM..."
)

cam_nii = nib.load(
    str(DECODER_CAM_PATH)
)

print(
    "Decoder Grad-CAM shape:",
    cam_nii.shape
)


# ==================================================
# RESAMPLE TO ORIGINAL CT SPACE
# ==================================================

print(
    "Resampling decoder Grad-CAM "
    "back to original CT space..."
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
    "Resampled decoder Grad-CAM shape:",
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
# CLEAN CAM
# ==================================================

cam = np.nan_to_num(
    cam,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


# Normalize CAM

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
# EXTRACT SLICE
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
# NORMALIZE CT FOR DISPLAY
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

    upper = lower + 1.0


ct_display = np.clip(
    ct_slice,
    lower,
    upper,
)


# ==================================================
# CREATE COMPARISON
# ==================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)


# --------------------------------------------------
# Original CT
# --------------------------------------------------

axes[0].imshow(
    np.rot90(ct_display),
    cmap="gray",
)

axes[0].set_title(
    f"Original CT\nSlice {slice_index}"
)

axes[0].axis("off")


# --------------------------------------------------
# Decoder Grad-CAM overlay
# --------------------------------------------------

axes[1].imshow(
    np.rot90(ct_display),
    cmap="gray",
)

axes[1].imshow(
    np.rot90(cam_slice),
    cmap="jet",
    alpha=0.5,
    vmin=0,
    vmax=1,
)

axes[1].set_title(
    "CT + Decoder Grad-CAM"
)

axes[1].axis("off")


# --------------------------------------------------
# Heatmap alone
# --------------------------------------------------

axes[2].imshow(
    np.rot90(cam_slice),
    cmap="jet",
    vmin=0,
    vmax=1,
)

axes[2].set_title(
    "Decoder Grad-CAM Heatmap"
)

axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    COMPARISON_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    "Comparison saved:",
    COMPARISON_PATH
)