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

OCCLUSION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "occlusion"
    / "spleen_occlusion_preprocessed.nii.gz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "occlusion"
)

ORIGINAL_SPACE_PATH = (
    OUTPUT_DIR
    / "spleen_occlusion_original_space.nii.gz"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "occlusion_comparison.png"
)


# ==================================================
# CHECK FILES
# ==================================================

if not CT_PATH.exists():

    raise FileNotFoundError(
        f"CT not found:\n{CT_PATH}"
    )


if not OCCLUSION_PATH.exists():

    raise FileNotFoundError(
        f"Occlusion heatmap not found:\n"
        f"{OCCLUSION_PATH}"
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
# LOAD OCCLUSION HEATMAP
# ==================================================

print(
    "Loading occlusion heatmap..."
)

occlusion_nii = nib.load(
    str(OCCLUSION_PATH)
)

print(
    "Occlusion shape:",
    occlusion_nii.shape
)


# ==================================================
# RESAMPLE TO ORIGINAL CT SPACE
# ==================================================

print(
    "Resampling occlusion map "
    "back to original CT space..."
)

target = (
    ct_nii.shape,
    ct_nii.affine,
)

resampled = resample_from_to(
    occlusion_nii,
    target,
    order=1,
)


nib.save(
    resampled,
    str(ORIGINAL_SPACE_PATH),
)


print(
    "Resampled occlusion shape:",
    resampled.shape
)

print(
    "Saved:",
    ORIGINAL_SPACE_PATH
)


# ==================================================
# LOAD ARRAYS
# ==================================================

ct = ct_nii.get_fdata()

occlusion = (
    resampled.get_fdata()
)


print(
    "CT shape:",
    ct.shape
)

print(
    "Occlusion shape:",
    occlusion.shape
)


# ==================================================
# CLEAN HEATMAP
# ==================================================

occlusion = np.nan_to_num(
    occlusion,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


minimum = occlusion.min()
maximum = occlusion.max()


if maximum > minimum:

    occlusion = (
        occlusion - minimum
    ) / (
        maximum - minimum
    )

else:

    occlusion = np.zeros_like(
        occlusion
    )


# ==================================================
# FIND MOST ACTIVE SLICE
# ==================================================

slice_scores = (
    occlusion.mean(
        axis=(0, 1)
    )
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

occlusion_slice = (
    occlusion[
        :,
        :,
        slice_index
    ]
)


# ==================================================
# NORMALIZE CT
# ==================================================

lower = np.percentile(
    ct_slice,
    1
)

upper = np.percentile(
    ct_slice,
    99
)

if upper <= lower:

    upper = lower + 1.0


ct_display = np.clip(
    ct_slice,
    lower,
    upper
)


# ==================================================
# CREATE COMPARISON
# ==================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# --------------------------------------------------
# Original CT
# --------------------------------------------------

axes[0].imshow(
    np.rot90(
        ct_display
    ),
    cmap="gray"
)

axes[0].set_title(
    f"Original CT\nSlice {slice_index}"
)

axes[0].axis("off")


# --------------------------------------------------
# CT + Occlusion
# --------------------------------------------------

axes[1].imshow(
    np.rot90(
        ct_display
    ),
    cmap="gray"
)

axes[1].imshow(
    np.rot90(
        occlusion_slice
    ),
    cmap="jet",
    alpha=0.5,
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "CT + Occlusion Sensitivity"
)

axes[1].axis("off")


# --------------------------------------------------
# Heatmap
# --------------------------------------------------

axes[2].imshow(
    np.rot90(
        occlusion_slice
    ),
    cmap="jet",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Occlusion Heatmap"
)

axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    COMPARISON_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "Comparison saved:",
    COMPARISON_PATH
)