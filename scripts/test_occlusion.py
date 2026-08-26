from pathlib import Path

from app.pipeline.inference import SpleenSegmenter
from app.xai.occlusion import SpleenOcclusion


# ==================================================
# PROJECT
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ==================================================
# INPUT
# ==================================================

INPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spleen"
    / "Task09_Spleen"
    / "imagesTr"
    / "spleen_10.nii.gz"
)


# ==================================================
# OUTPUT
# ==================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
    / "occlusion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# CHECK INPUT
# ==================================================

if not INPUT_IMAGE.exists():

    raise FileNotFoundError(
        f"Input image not found:\n"
        f"{INPUT_IMAGE}"
    )


# ==================================================
# LOAD MODEL
# ==================================================

print("=" * 60)
print("MedSegOps - 3D Occlusion Sensitivity")
print("=" * 60)

print()
print("Loading segmentation model...")

segmenter = SpleenSegmenter()

print(
    "Segmentation model loaded."
)


# ==================================================
# CREATE OCCLUSION EXPLAINER
# ==================================================

occlusion = SpleenOcclusion(
    segmenter=segmenter,
    roi_size=(96, 96, 96),
    block_size=(16, 16, 16),
)


# ==================================================
# GENERATE
# ==================================================

print()
print(
    "Generating 3D occlusion sensitivity..."
)

result = occlusion.generate(
    INPUT_IMAGE
)


# ==================================================
# RESULTS
# ==================================================

heatmap = result["heatmap"]

print()
print("=" * 60)
print("Occlusion Results")
print("=" * 60)

print(
    "Full heatmap shape:",
    heatmap.shape
)

print(
    "Heatmap minimum:",
    heatmap.min()
)

print(
    "Heatmap maximum:",
    heatmap.max()
)

print(
    "Baseline target score:",
    result["target_score"]
)

print(
    "Spleen center:",
    result["center"]
)

print(
    "ROI start:",
    result["roi_start"]
)

print(
    "ROI size:",
    result["roi_size"]
)

print(
    "Block size:",
    result["block_size"]
)

print(
    "Preprocessed shape:",
    result["preprocessed_shape"]
)


# ==================================================
# SAVE FULL PREPROCESSED HEATMAP
# ==================================================

heatmap_path = (
    OUTPUT_DIR
    / "spleen_occlusion_preprocessed.nii.gz"
)

occlusion.save_heatmap(
    heatmap=heatmap,
    output_path=heatmap_path,
    affine=result["preprocessed_affine"],
)


# ==================================================
# SAVE FULL PREPROCESSED SLICES
# ==================================================

slice_directory = (
    OUTPUT_DIR
    / "preprocessed_slices"
)

occlusion.save_slice_heatmaps(
    heatmap=heatmap,
    output_directory=slice_directory,
)


print()
print("=" * 60)
print("OCCLUSION COMPLETE")
print("=" * 60)

print(
    "Full heatmap:",
    heatmap_path
)

print(
    "Slices:",
    slice_directory
)