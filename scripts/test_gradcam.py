from pathlib import Path

from app.pipeline.inference import SpleenSegmenter
from app.xai.gradcam import SpleenGradCAM


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

INPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spleen"
    / "Task09_Spleen"
    / "imagesTr"
    / "spleen_10.nii.gz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# TARGET LAYER
# ==================================================

TARGET_LAYER_NAME = (
    "model.1.submodule.2.1.conv.unit0.conv"
)


# ==================================================
# LOAD MODEL
# ==================================================

print("=" * 60)
print("MedSegOps - Decoder Grad-CAM Test")
print("=" * 60)

print()
print("Loading segmentation model...")

segmenter = SpleenSegmenter()

print("Segmentation model loaded.")


# ==================================================
# FIND TARGET LAYER
# ==================================================

target_layer = None

for name, module in segmenter.model.named_modules():

    if name == TARGET_LAYER_NAME:
        target_layer = module
        break


if target_layer is None:

    raise RuntimeError(
        f"Target layer not found: "
        f"{TARGET_LAYER_NAME}"
    )


print()
print("Selected Grad-CAM layer:")
print(TARGET_LAYER_NAME)
print(target_layer)


# ==================================================
# CREATE GRAD-CAM
# ==================================================

gradcam = SpleenGradCAM(
    segmenter,
    target_layer=target_layer,
)


# ==================================================
# GENERATE
# ==================================================

print()
print("Generating decoder Grad-CAM...")

result = gradcam.generate(
    INPUT_IMAGE
)


# ==================================================
# RESULTS
# ==================================================

heatmap = result["heatmap"]

print()
print("=" * 60)
print("Decoder Grad-CAM Results")
print("=" * 60)

print(
    "Heatmap shape:",
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
    "Target score:",
    result["target_score"]
)

print(
    "Spleen center:",
    result["center"]
)

print(
    "Patch size:",
    result["patch_size"]
)


# ==================================================
# SAVE
# ==================================================

heatmap_path = (
    OUTPUT_DIR
    / "spleen_gradcam_decoder.nii.gz"
)

gradcam.save_heatmap(
    heatmap=heatmap,
    output_path=heatmap_path,
    affine=result["preprocessed_affine"],
)


slice_directory = (
    OUTPUT_DIR
    / "decoder_slices"
)

gradcam.save_slice_heatmaps(
    heatmap,
    slice_directory,
)


gradcam.close()


print()
print("=" * 60)
print("DECODER GRAD-CAM COMPLETE")
print("=" * 60)

print(
    "Heatmap:",
    heatmap_path
)

print(
    "Slices:",
    slice_directory
)