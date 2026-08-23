from pathlib import Path
from uuid import uuid4
import shutil

import nibabel as nib
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline.inference import SpleenSegmenter


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="MedSegOps",
    description="AI-powered medical image segmentation"
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# --------------------------------------------------
# Load segmentation model
# --------------------------------------------------

print("Loading spleen segmentation model...")

segmenter = SpleenSegmenter(
    model_path=BASE_DIR / "models" / "monai" / "model.pt"
)

print("Model loaded successfully.")


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert CT slice to 0-255 grayscale image.
    """

    image = image.astype(np.float32)

    lower = np.percentile(image, 1)
    upper = np.percentile(image, 99)

    if upper <= lower:
        upper = lower + 1.0

    image = np.clip(image, lower, upper)

    image = (image - lower) / (upper - lower)

    image = image * 255.0

    return image.astype(np.uint8)


# --------------------------------------------------
# Segmentation statistics
# --------------------------------------------------

def calculate_segmentation_statistics(
    mask_path: Path
):
    """
    Calculate statistics from the segmentation mask.
    """

    print("Calculating segmentation statistics...")

    mask_nifti = nib.load(
        str(mask_path)
    )

    mask_data = mask_nifti.get_fdata()

    mask_data = np.squeeze(mask_data)

    if mask_data.ndim != 3:
        raise ValueError(
            f"Expected 3D segmentation mask, got {mask_data.shape}"
        )

    # Convert segmentation to binary mask
    binary_mask = mask_data > 0

    # Number of spleen voxels
    voxel_count = int(
        np.count_nonzero(binary_mask)
    )

    # Get voxel spacing
    voxel_spacing = (
        mask_nifti.header.get_zooms()[:3]
    )

    spacing_x = float(voxel_spacing[0])
    spacing_y = float(voxel_spacing[1])
    spacing_z = float(voxel_spacing[2])

    # Volume of one voxel in mm³
    voxel_volume_mm3 = (
        spacing_x
        * spacing_y
        * spacing_z
    )

    # Total spleen volume
    volume_mm3 = (
        voxel_count
        * voxel_volume_mm3
    )

    # 1 mL = 1000 mm³
    volume_ml = volume_mm3 / 1000.0

    # 1 cm³ = 1000 mm³
    volume_cm3 = volume_mm3 / 1000.0

    statistics = {
        "voxel_count": voxel_count,

        "voxel_spacing_mm": {
            "x": round(spacing_x, 4),
            "y": round(spacing_y, 4),
            "z": round(spacing_z, 4),
        },

        "voxel_volume_mm3": round(
            voxel_volume_mm3,
            4
        ),

        "volume_mm3": round(
            volume_mm3,
            2
        ),

        "volume_ml": round(
            volume_ml,
            2
        ),

        "volume_cm3": round(
            volume_cm3,
            2
        ),

        "dimensions": {
            "x": int(mask_data.shape[0]),
            "y": int(mask_data.shape[1]),
            "z": int(mask_data.shape[2]),
        },
    }

    print("Segmentation statistics:")
    print(
        f"Voxel count: {statistics['voxel_count']}"
    )

    print(
        f"Voxel spacing: "
        f"{statistics['voxel_spacing_mm']}"
    )

    print(
        f"Voxel volume: "
        f"{statistics['voxel_volume_mm3']} mm³"
    )

    print(
        f"Spleen volume: "
        f"{statistics['volume_ml']} mL"
    )

    return statistics


# --------------------------------------------------
# Create slice visualizations
# --------------------------------------------------

def create_slice_visualizations(
    ct_path: Path,
    mask_path: Path,
    output_directory: Path,
):
    """
    Create CT, mask, and overlay PNG images
    for every slice.
    """

    print("Creating slice visualizations...")

    ct_nifti = nib.load(
        str(ct_path)
    )

    mask_nifti = nib.load(
        str(mask_path)
    )

    ct_data = ct_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    print(
        f"CT shape: {ct_data.shape}"
    )

    print(
        f"Mask shape: {mask_data.shape}"
    )

    ct_data = np.squeeze(ct_data)
    mask_data = np.squeeze(mask_data)

    if ct_data.ndim != 3:
        raise ValueError(
            f"Expected 3D CT volume, got shape {ct_data.shape}"
        )

    if mask_data.ndim != 3:
        raise ValueError(
            f"Expected 3D mask volume, got shape {mask_data.shape}"
        )

    # --------------------------------------------------
    # Create directories
    # --------------------------------------------------

    visualization_dir = (
        output_directory / "visualizations"
    )

    ct_dir = visualization_dir / "ct"
    mask_dir = visualization_dir / "mask"
    overlay_dir = visualization_dir / "overlay"

    ct_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    num_slices = ct_data.shape[2]

    print(
        f"Creating {num_slices} slice visualizations..."
    )

    # --------------------------------------------------
    # Process every slice
    # --------------------------------------------------

    for slice_index in range(num_slices):

        ct_slice = ct_data[
            :,
            :,
            slice_index
        ]

        mask_slice = mask_data[
            :,
            :,
            slice_index
        ]

        # ------------------------------
        # CT image
        # ------------------------------

        ct_image = normalize_image(
            ct_slice
        )

        ct_pil = Image.fromarray(
            ct_image
        )

        ct_path_png = (
            ct_dir
            / f"slice_{slice_index:03d}.png"
        )

        ct_pil.save(
            ct_path_png
        )

        # ------------------------------
        # Segmentation mask
        # ------------------------------

        binary_mask = (
            mask_slice > 0
        ).astype(np.uint8)

        mask_image = (
            binary_mask * 255
        )

        mask_pil = Image.fromarray(
            mask_image
        )

        mask_path_png = (
            mask_dir
            / f"slice_{slice_index:03d}.png"
        )

        mask_pil.save(
            mask_path_png
        )

        # ------------------------------
        # CT + segmentation overlay
        # ------------------------------

        overlay = np.stack(
            [
                ct_image,
                ct_image,
                ct_image,
            ],
            axis=-1,
        ).astype(np.float32)

        mask_pixels = (
            binary_mask > 0
        )

        # Red segmentation overlay
        overlay[mask_pixels, 0] = 255
        overlay[mask_pixels, 1] *= 0.35
        overlay[mask_pixels, 2] *= 0.35

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        overlay_pil = Image.fromarray(
            overlay
        )

        overlay_path_png = (
            overlay_dir
            / f"slice_{slice_index:03d}.png"
        )

        overlay_pil.save(
            overlay_path_png
        )

    print(
        "Slice visualizations created successfully."
    )

    return {
        "ct_url": (
            f"/outputs/"
            f"{output_directory.name}"
            f"/visualizations/ct"
        ),

        "mask_url": (
            f"/outputs/"
            f"{output_directory.name}"
            f"/visualizations/mask"
        ),

        "overlay_url": (
            f"/outputs/"
            f"{output_directory.name}"
            f"/visualizations/overlay"
        ),

        "num_slices": num_slices,
    }


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    index_file = (
        TEMPLATE_DIR / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail="index.html not found",
        )

    return index_file.read_text(
        encoding="utf-8"
    )


# --------------------------------------------------
# Run segmentation
# --------------------------------------------------

@app.post("/segment")
async def segment_ct(
    file: UploadFile = File(...)
):

    # --------------------------------------------------
    # Validate file
    # --------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected",
        )

    filename = (
        file.filename.lower()
    )

    if not (
        filename.endswith(".nii")
        or filename.endswith(".nii.gz")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only .nii and .nii.gz "
                "files are supported"
            ),
        )

    # --------------------------------------------------
    # Create job ID
    # --------------------------------------------------

    job_id = str(
        uuid4()
    )

    if filename.endswith(".nii.gz"):

        extension = ".nii.gz"

    else:

        extension = ".nii"

    upload_path = (
        UPLOAD_DIR
        / f"{job_id}{extension}"
    )

    job_output_dir = (
        OUTPUT_DIR / job_id
    )

    job_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------

    try:

        with upload_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        print(
            f"Uploaded file saved: "
            f"{upload_path}"
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save uploaded "
                f"file: {error}"
            ),
        )

    # --------------------------------------------------
    # Run segmentation
    # --------------------------------------------------

    try:

        print(
            "Starting segmentation..."
        )

        segmenter.predict(
            image_path=upload_path,
            output_dir=job_output_dir,
        )

        print(
            "Segmentation completed."
        )

    except Exception as error:

        print(
            f"Segmentation error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference failed: {error}"
            ),
        )

    # --------------------------------------------------
    # Find segmentation output
    # --------------------------------------------------

    segmentation_files = list(
        job_output_dir.glob(
            "*_seg.nii.gz"
        )
    )

    if not segmentation_files:

        segmentation_files = list(
            job_output_dir.glob(
                "*.nii.gz"
            )
        )

    if not segmentation_files:

        raise HTTPException(
            status_code=500,
            detail=(
                "Segmentation output file "
                "was not created"
            ),
        )

    segmentation_path = (
        segmentation_files[0]
    )

    print(
        f"Segmentation file: "
        f"{segmentation_path}"
    )

    # --------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------

    try:

        statistics = (
            calculate_segmentation_statistics(
                segmentation_path
            )
        )

        print(
            "Segmentation statistics "
            "calculated successfully."
        )

    except Exception as error:

        print(
            f"Statistics error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Statistics calculation "
                f"failed: {error}"
            ),
        )

    # --------------------------------------------------
    # Create visualizations
    # --------------------------------------------------

    try:

        visualization_result = (
            create_slice_visualizations(
                ct_path=upload_path,
                mask_path=segmentation_path,
                output_directory=job_output_dir,
            )
        )

        print(
            f"Created "
            f"{visualization_result['num_slices']} "
            f"slice visualizations."
        )

    except Exception as error:

        print(
            f"Visualization error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Visualization failed: "
                f"{error}"
            ),
        )

    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {
        "success": True,

        "job_id": job_id,

        "num_slices": (
            visualization_result[
                "num_slices"
            ]
        ),

        "ct_url": (
            visualization_result[
                "ct_url"
            ]
        ),

        "mask_url": (
            visualization_result[
                "mask_url"
            ]
        ),

        "overlay_url": (
            visualization_result[
                "overlay_url"
            ]
        ),

        "statistics": statistics,

        "download_url": (
            f"/download/{job_id}"
        ),

        "mask_download_url": (
            f"/download/{job_id}"
        ),
    }


# --------------------------------------------------
# Download segmentation
# --------------------------------------------------

@app.get(
    "/download/{job_id}"
)
async def download_segmentation(
    job_id: str
):

    job_directory = (
        OUTPUT_DIR / job_id
    )

    if not job_directory.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Segmentation result not found"
            ),
        )

    segmentation_files = list(
        job_directory.glob(
            "*_seg.nii.gz"
        )
    )

    if not segmentation_files:

        raise HTTPException(
            status_code=404,
            detail=(
                "Segmentation file not found"
            ),
        )

    segmentation_file = (
        segmentation_files[0]
    )

    return FileResponse(
        path=str(segmentation_file),
        media_type="application/gzip",
        filename=(
            "spleen_segmentation.nii.gz"
        ),
    )