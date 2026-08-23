from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

CT_PATH = Path(
    "data/uploads/bf58e687-df32-4dc9-bdb0-c44939b6bd9c.nii.gz"
)

MASK_PATH = Path(
    "outputs/bf58e687-df32-4dc9-bdb0-c44939b6bd9c_seg.nii.gz"
)

OUTPUT_PATH = Path(
    "outputs/segmentation_best_slice.png"
)


# --------------------------------------------------
# Load NIfTI files
# --------------------------------------------------

ct_nii = nib.load(CT_PATH)
mask_nii = nib.load(MASK_PATH)

ct = ct_nii.get_fdata()
mask = mask_nii.get_fdata()

print(f"CT shape: {ct.shape}")
print(f"Mask shape: {mask.shape}")


# --------------------------------------------------
# Check shapes
# --------------------------------------------------

if ct.shape != mask.shape:
    raise ValueError(
        f"CT and mask shapes do not match: "
        f"{ct.shape} vs {mask.shape}"
    )


# --------------------------------------------------
# Find slice with largest segmentation area
# --------------------------------------------------

# Count segmented pixels in every axial slice
slice_areas = np.sum(mask > 0, axis=(0, 1))

best_slice = int(np.argmax(slice_areas))
largest_area = int(slice_areas[best_slice])

print(f"Best slice index: {best_slice}")
print(f"Largest mask area: {largest_area} pixels")


# --------------------------------------------------
# Extract best slice
# --------------------------------------------------

ct_slice = ct[:, :, best_slice]
mask_slice = mask[:, :, best_slice]


# --------------------------------------------------
# CT windowing for visualization
# --------------------------------------------------

window_min = -100
window_max = 300

ct_display = np.clip(
    ct_slice,
    window_min,
    window_max,
)


# --------------------------------------------------
# Create visualization
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))


# Original CT
axes[0].imshow(
    np.rot90(ct_display),
    cmap="gray",
)

axes[0].set_title(
    f"Original CT\nSlice {best_slice}"
)

axes[0].axis("off")


# Predicted segmentation
axes[1].imshow(
    np.rot90(mask_slice),
    cmap="gray",
)

axes[1].set_title(
    f"Predicted Spleen Segmentation\nSlice {best_slice}"
)

axes[1].axis("off")


# CT + segmentation overlay
axes[2].imshow(
    np.rot90(ct_display),
    cmap="gray",
)

axes[2].imshow(
    np.rot90(mask_slice),
    cmap="autumn",
    alpha=0.5,
)

axes[2].set_title(
    f"CT + Segmentation Overlay\nSlice {best_slice}"
)

axes[2].axis("off")


# --------------------------------------------------
# Save result
# --------------------------------------------------

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(f"\nVisualization saved to:")
print(OUTPUT_PATH.resolve())