from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from nibabel.processing import resample_from_to
from PIL import Image

from app.io.dicom import (
    DicomSeriesError,
    convert_dicom_series_to_nifti,
)

from app.io.dicom_seg import (
    DicomSegError,
    create_dicom_seg,
)

from app.pipeline.inference import (
    SpleenSegmenter,
)

from app.xai.alignment import (
    evaluate_explanation_alignment,
)

from app.xai.gradcam import (
    SpleenGradCAM,
)

from app.xai.occlusion import (
    SpleenOcclusion,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

UPLOAD_DIR = (
    BASE_DIR
    / "data"
    / "uploads"
)

DICOM_UPLOAD_DIR = (
    BASE_DIR
    / "data"
    / "dicom_uploads"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)

TEMPLATE_DIR = (
    BASE_DIR
    / "app"
    / "templates"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DICOM_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="MedSegOps",
    description=(
        "Explainable and production-oriented "
        "medical image segmentation pipeline"
    ),
    version="0.4.0",
)


app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading spleen segmentation model..."
)

segmenter = SpleenSegmenter(
    model_path=(
        BASE_DIR
        / "models"
        / "monai"
        / "model.pt"
    )
)

print(
    "Model loaded successfully."
)


# ============================================================
# CT VISUALIZATION
# ============================================================

def normalize_ct_slice(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert a CT slice to an 8-bit image.
    """

    image = image.astype(
        np.float32
    )

    lower = np.percentile(
        image,
        1,
    )

    upper = np.percentile(
        image,
        99,
    )

    if upper <= lower:

        upper = lower + 1.0

    image = np.clip(
        image,
        lower,
        upper,
    )

    image = (
        image - lower
    ) / (
        upper - lower
    )

    image *= 255.0

    return image.astype(
        np.uint8
    )


# ============================================================
# XAI OVERLAY
# ============================================================

def save_overlay_slice(
    ct_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    output_path: Path,
    title: str,
):
    """
    Save CT + XAI heatmap overlay.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    ct_display = normalize_ct_slice(
        ct_slice
    )

    heatmap_slice = np.nan_to_num(
        heatmap_slice,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = heatmap_slice.min()
    maximum = heatmap_slice.max()

    if maximum > minimum:

        heatmap_slice = (
            heatmap_slice - minimum
        ) / (
            maximum - minimum
        )

    else:

        heatmap_slice = np.zeros_like(
            heatmap_slice
        )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        np.rot90(
            ct_display
        ),
        cmap="gray",
    )

    plt.imshow(
        np.rot90(
            heatmap_slice
        ),
        cmap="jet",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )

    plt.title(
        title
    )

    plt.axis("off")

    plt.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# SAVE XAI SLICES
# ============================================================

def save_heatmap_slices(
    ct_path: Path,
    heatmap_path: Path,
    output_directory: Path,
    title_prefix: str,
):
    """
    Save one PNG per original CT slice.
    """

    ct_nii = nib.load(
        str(ct_path)
    )

    heatmap_nii = nib.load(
        str(heatmap_path)
    )

    ct_data = np.squeeze(
        ct_nii.get_fdata()
    )

    heatmap_data = np.squeeze(
        heatmap_nii.get_fdata()
    )

    if ct_data.shape != heatmap_data.shape:

        raise ValueError(
            "CT and heatmap shapes do not match: "
            f"{ct_data.shape} vs "
            f"{heatmap_data.shape}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for slice_index in range(
        ct_data.shape[2]
    ):

        output_path = (
            output_directory
            / f"slice_{slice_index:03d}.png"
        )

        save_overlay_slice(
            ct_data[
                :,
                :,
                slice_index,
            ],
            heatmap_data[
                :,
                :,
                slice_index,
            ],
            output_path,
            f"{title_prefix} - Slice {slice_index}",
        )

    return int(
        ct_data.shape[2]
    )


# ============================================================
# SEGMENTATION STATISTICS
# ============================================================

def calculate_segmentation_statistics(
    mask_path: Path,
):
    """
    Calculate spleen volume statistics.
    """

    mask_nii = nib.load(
        str(mask_path)
    )

    mask_data = np.squeeze(
        mask_nii.get_fdata()
    )

    if mask_data.ndim != 3:

        raise ValueError(
            f"Expected a 3D mask, "
            f"got {mask_data.shape}"
        )

    binary_mask = (
        mask_data > 0
    )

    voxel_count = int(
        np.count_nonzero(
            binary_mask
        )
    )

    spacing = (
        mask_nii.header.get_zooms()[:3]
    )

    spacing_x = float(
        spacing[0]
    )

    spacing_y = float(
        spacing[1]
    )

    spacing_z = float(
        spacing[2]
    )

    voxel_volume_mm3 = (
        spacing_x
        * spacing_y
        * spacing_z
    )

    volume_mm3 = (
        voxel_count
        * voxel_volume_mm3
    )

    volume_ml = (
        volume_mm3
        / 1000.0
    )

    return {
        "voxel_count": voxel_count,

        "voxel_spacing_mm": {
            "x": round(
                spacing_x,
                4,
            ),
            "y": round(
                spacing_y,
                4,
            ),
            "z": round(
                spacing_z,
                4,
            ),
        },

        "voxel_volume_mm3": round(
            voxel_volume_mm3,
            4,
        ),

        "volume_mm3": round(
            volume_mm3,
            2,
        ),

        "volume_ml": round(
            volume_ml,
            2,
        ),

        "volume_cm3": round(
            volume_ml,
            2,
        ),

        "dimensions": {
            "x": int(
                mask_data.shape[0]
            ),
            "y": int(
                mask_data.shape[1]
            ),
            "z": int(
                mask_data.shape[2]
            ),
        },
    }


# ============================================================
# SEGMENTATION VISUALIZATIONS
# ============================================================

def create_segmentation_visualizations(
    ct_path: Path,
    mask_path: Path,
    output_directory: Path,
):
    """
    Create CT, mask, and overlay PNG slices.
    """

    ct_nii = nib.load(
        str(ct_path)
    )

    mask_nii = nib.load(
        str(mask_path)
    )

    ct_data = np.squeeze(
        ct_nii.get_fdata()
    )

    mask_data = np.squeeze(
        mask_nii.get_fdata()
    )

    if ct_data.shape != mask_data.shape:

        raise ValueError(
            "CT and mask shapes do not match: "
            f"{ct_data.shape} vs "
            f"{mask_data.shape}"
        )

    base_dir = (
        output_directory
        / "visualizations"
    )

    ct_dir = (
        base_dir
        / "ct"
    )

    mask_dir = (
        base_dir
        / "mask"
    )

    overlay_dir = (
        base_dir
        / "overlay"
    )

    ct_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_slices = (
        ct_data.shape[2]
    )

    for slice_index in range(
        num_slices
    ):

        ct_slice = ct_data[
            :,
            :,
            slice_index,
        ]

        mask_slice = mask_data[
            :,
            :,
            slice_index,
        ]

        # ------------------------------------------
        # CT
        # ------------------------------------------

        ct_image = (
            normalize_ct_slice(
                ct_slice
            )
        )

        Image.fromarray(
            ct_image
        ).save(
            ct_dir
            / f"slice_{slice_index:03d}.png"
        )

        # ------------------------------------------
        # MASK
        # ------------------------------------------

        binary_mask = (
            mask_slice > 0
        )

        mask_image = (
            binary_mask
            .astype(np.uint8)
            * 255
        )

        Image.fromarray(
            mask_image
        ).save(
            mask_dir
            / f"slice_{slice_index:03d}.png"
        )

        # ------------------------------------------
        # OVERLAY
        # ------------------------------------------

        overlay = np.stack(
            [
                ct_image,
                ct_image,
                ct_image,
            ],
            axis=-1,
        ).astype(
            np.float32
        )

        overlay[
            binary_mask,
            0
        ] = 255

        overlay[
            binary_mask,
            1
        ] *= 0.35

        overlay[
            binary_mask,
            2
        ] *= 0.35

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(
            np.uint8
        )

        Image.fromarray(
            overlay
        ).save(
            overlay_dir
            / f"slice_{slice_index:03d}.png"
        )

    return {
        "num_slices": num_slices,

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
    }


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home():

    index_file = (
        TEMPLATE_DIR
        / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "app/templates/index.html "
                "was not found."
            ),
        )

    return HTMLResponse(
        content=index_file.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "MedSegOps",
    }


# ============================================================
# SEGMENT
# ============================================================

@app.post("/segment")
async def segment(
    file: UploadFile | None = File(
        default=None
    ),
    dicom_files: list[UploadFile] | None = File(
        default=None
    ),
):
    """
    Accept either:
        - one NIfTI file
        - one DICOM CT series
    """

    has_nifti = (
        file is not None
        and bool(file.filename)
    )

    has_dicom = bool(
        dicom_files
    )

    if has_nifti and has_dicom:

        raise HTTPException(
            status_code=400,
            detail=(
                "Choose either NIfTI or DICOM, "
                "not both."
            ),
        )

    if not has_nifti and not has_dicom:

        raise HTTPException(
            status_code=400,
            detail=(
                "No NIfTI file or DICOM "
                "series was uploaded."
            ),
        )

    job_id = str(
        uuid4()
    )

    job_output_dir = (
        OUTPUT_DIR
        / job_id
    )

    job_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path: Path

    input_type: str

    dicom_directory: Path | None = None

    try:

        # ==================================================
        # NIFTI INPUT
        # ==================================================

        if has_nifti:

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
                        "files are supported."
                    ),
                )

            extension = (
                ".nii.gz"
                if filename.endswith(
                    ".nii.gz"
                )
                else ".nii"
            )

            input_path = (
                UPLOAD_DIR
                / f"{job_id}{extension}"
            )

            with input_path.open(
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer,
                )

            input_type = "NIfTI"

            print(
                f"NIfTI uploaded: "
                f"{input_path}"
            )

        # ==================================================
        # DICOM INPUT
        # ==================================================

        else:

            dicom_directory = (
                DICOM_UPLOAD_DIR
                / job_id
            )

            dicom_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            saved_files = 0

            for dicom_file in (
                dicom_files or []
            ):

                if not dicom_file.filename:

                    continue

                safe_name = (
                    Path(
                        dicom_file.filename
                    ).name
                )

                if not safe_name:

                    continue

                target_path = (
                    dicom_directory
                    / safe_name
                )

                with target_path.open(
                    "wb"
                ) as buffer:

                    shutil.copyfileobj(
                        dicom_file.file,
                        buffer,
                    )

                saved_files += 1

            if saved_files == 0:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The DICOM series "
                        "contained no files."
                    ),
                )

            input_path = (
                job_output_dir
                / "dicom_converted.nii.gz"
            )

            print(
                f"Received {saved_files} "
                "DICOM files."
            )

            try:

                dicom_result = (
                    convert_dicom_series_to_nifti(
                        dicom_directory,
                        input_path,
                    )
                )

            except DicomSeriesError as error:

                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                )

            print(
                "DICOM converted successfully:"
            )

            print(
                dicom_result
            )

            input_type = "DICOM"

        # ==================================================
        # SEGMENTATION
        # ==================================================

        print(
            f"Starting segmentation for "
            f"{input_type}..."
        )

        segmenter.predict(
            image_path=input_path,
            output_dir=job_output_dir,
        )

        print(
            "Segmentation completed."
        )

        # ==================================================
        # FIND MASK
        # ==================================================

        segmentation_files = list(
            job_output_dir.glob(
                "*_seg.nii.gz"
            )
        )

        if not segmentation_files:

            raise RuntimeError(
                "Segmentation output "
                "was not created."
            )

        segmentation_path = (
            segmentation_files[0]
        )

        print(
            "Segmentation file:",
            segmentation_path,
        )

        # ==================================================
        # STATISTICS
        # ==================================================

        statistics = (
            calculate_segmentation_statistics(
                segmentation_path
            )
        )

        # ==================================================
        # VISUALIZATION
        # ==================================================

        visualization = (
            create_segmentation_visualizations(
                ct_path=input_path,
                mask_path=segmentation_path,
                output_directory=job_output_dir,
            )
        )

        # ==================================================
        # DICOM SEG
        # ==================================================

        dicom_seg_available = False

        dicom_seg_path = None

        if (
            input_type == "DICOM"
            and dicom_directory is not None
        ):

            print(
                "Creating DICOM SEG..."
            )

            dicom_seg_path = (
                job_output_dir
                / "dicom_seg"
                / "spleen_segmentation.dcm"
            )

            try:

                dicom_seg_result = (
                    create_dicom_seg(
                        dicom_directory=dicom_directory,
                        segmentation_path=segmentation_path,
                        output_path=dicom_seg_path,
                    )
                )

                dicom_seg_available = True

                print(
                    "DICOM SEG created successfully:"
                )

                print(
                    dicom_seg_result
                )

            except DicomSegError as error:

                print(
                    "DICOM SEG creation failed:"
                )

                print(
                    error
                )

                # Segmentation itself remains successful.
                dicom_seg_available = False

        # ==================================================
        # RESPONSE
        # ==================================================

        return {
            "success": True,

            "job_id": job_id,

            "input_type": input_type,

            "num_slices": visualization[
                "num_slices"
            ],

            "ct_url": visualization[
                "ct_url"
            ],

            "mask_url": visualization[
                "mask_url"
            ],

            "overlay_url": visualization[
                "overlay_url"
            ],

            "statistics": statistics,

            "download_url": (
                f"/download/{job_id}"
            ),

            "mask_download_url": (
                f"/download/{job_id}"
            ),

            "dicom_seg_available": (
                dicom_seg_available
            ),

            "dicom_seg_download_url": (
                f"/download-dicom-seg/{job_id}"
                if dicom_seg_available
                else None
            ),

            "xai_available": True,
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Segmentation error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Segmentation failed: "
                f"{error}"
            ),
        )

    finally:

        if file is not None:

            await file.close()

        for uploaded_dicom in (
            dicom_files or []
        ):

            await uploaded_dicom.close()


# ============================================================
# XAI
# ============================================================

@app.post(
    "/xai/{job_id}"
)
async def generate_xai(
    job_id: str,
):

    job_directory = (
        OUTPUT_DIR
        / job_id
    )

    if not job_directory.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Segmentation job not found."
            ),
        )

    # ------------------------------------------
    # Find input NIfTI
    # ------------------------------------------

    candidates = []

    for path in job_directory.iterdir():

        if (
            path.is_file()
            and (
                path.name.endswith(".nii")
                or path.name.endswith(".nii.gz")
            )
            and not path.name.endswith(
                "_seg.nii.gz"
            )
        ):

            candidates.append(
                path
            )

    # NIfTI uploads
    if not candidates:

        candidates = list(
            UPLOAD_DIR.glob(
                f"{job_id}.nii"
            )
        )

        candidates += list(
            UPLOAD_DIR.glob(
                f"{job_id}.nii.gz"
            )
        )

    if not candidates:

        raise HTTPException(
            status_code=404,
            detail=(
                "Original volume not found."
            ),
        )

    input_path = candidates[0]

    # ------------------------------------------
    # Segmentation
    # ------------------------------------------

    segmentation_files = list(
        job_directory.glob(
            "*_seg.nii.gz"
        )
    )

    if not segmentation_files:

        raise HTTPException(
            status_code=404,
            detail=(
                "Segmentation file not found."
            ),
        )

    segmentation_path = (
        segmentation_files[0]
    )

    # ------------------------------------------
    # XAI directory
    # ------------------------------------------

    xai_directory = (
        job_directory
        / "xai"
    )

    xai_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ==================================================
        # GRAD-CAM
        # ==================================================

        print(
            "Generating decoder Grad-CAM..."
        )

        target_layer_name = (
            "model.1.submodule.2.1.conv.unit0.conv"
        )

        target_layer = None

        for name, module in (
            segmenter.model.named_modules()
        ):

            if name == target_layer_name:

                target_layer = module
                break

        if target_layer is None:

            raise RuntimeError(
                "Grad-CAM target layer not found: "
                f"{target_layer_name}"
            )

        gradcam = SpleenGradCAM(
            segmenter,
            target_layer=target_layer,
        )

        gradcam_result = (
            gradcam.generate(
                input_path
            )
        )

        gradcam_processed_path = (
            xai_directory
            / "gradcam_preprocessed.nii.gz"
        )

        gradcam.save_heatmap(
            heatmap=gradcam_result[
                "heatmap"
            ],
            output_path=gradcam_processed_path,
            affine=gradcam_result[
                "preprocessed_affine"
            ],
        )

        gradcam.close()

        # ------------------------------------------
        # Resample Grad-CAM
        # ------------------------------------------

        ct_nii = nib.load(
            str(input_path)
        )

        gradcam_nii = nib.load(
            str(
                gradcam_processed_path
            )
        )

        gradcam_original_nii = (
            resample_from_to(
                gradcam_nii,
                (
                    ct_nii.shape,
                    ct_nii.affine,
                ),
                order=1,
            )
        )

        gradcam_original_path = (
            xai_directory
            / "gradcam_original_space.nii.gz"
        )

        nib.save(
            gradcam_original_nii,
            str(
                gradcam_original_path
            ),
        )

        gradcam_slice_dir = (
            xai_directory
            / "gradcam_slices"
        )

        gradcam_slice_count = (
            save_heatmap_slices(
                ct_path=input_path,
                heatmap_path=gradcam_original_path,
                output_directory=gradcam_slice_dir,
                title_prefix="Grad-CAM",
            )
        )

        # ==================================================
        # OCCLUSION
        # ==================================================

        print(
            "Generating occlusion sensitivity..."
        )

        occlusion = SpleenOcclusion(
            segmenter=segmenter,
            roi_size=(
                96,
                96,
                96,
            ),
            block_size=(
                16,
                16,
                16,
            ),
        )

        occlusion_result = (
            occlusion.generate(
                input_path
            )
        )

        occlusion_processed_path = (
            xai_directory
            / "occlusion_preprocessed.nii.gz"
        )

        occlusion.save_heatmap(
            heatmap=occlusion_result[
                "heatmap"
            ],
            output_path=(
                occlusion_processed_path
            ),
            affine=occlusion_result[
                "preprocessed_affine"
            ],
        )

        # ------------------------------------------
        # Resample occlusion
        # ------------------------------------------

        occlusion_nii = nib.load(
            str(
                occlusion_processed_path
            )
        )

        occlusion_original_nii = (
            resample_from_to(
                occlusion_nii,
                (
                    ct_nii.shape,
                    ct_nii.affine,
                ),
                order=1,
            )
        )

        occlusion_original_path = (
            xai_directory
            / "occlusion_original_space.nii.gz"
        )

        nib.save(
            occlusion_original_nii,
            str(
                occlusion_original_path
            ),
        )

        occlusion_slice_dir = (
            xai_directory
            / "occlusion_slices"
        )

        occlusion_slice_count = (
            save_heatmap_slices(
                ct_path=input_path,
                heatmap_path=occlusion_original_path,
                output_directory=occlusion_slice_dir,
                title_prefix="Occlusion Sensitivity",
            )
        )

        # ==================================================
        # ALIGNMENT
        # ==================================================

        segmentation_data = (
            nib.load(
                str(segmentation_path)
            ).get_fdata()
        )

        gradcam_data = (
            gradcam_original_nii
            .get_fdata()
        )

        occlusion_data = (
            occlusion_original_nii
            .get_fdata()
        )

        gradcam_alignment = (
            evaluate_explanation_alignment(
                heatmap=gradcam_data,
                segmentation_mask=segmentation_data,
                percentile=80.0,
            )
        )

        occlusion_alignment = (
            evaluate_explanation_alignment(
                heatmap=occlusion_data,
                segmentation_mask=segmentation_data,
                percentile=90.0,
            )
        )

        print(
            "XAI generation completed."
        )

        return {
            "success": True,

            "job_id": job_id,

            "gradcam": {
                "method": "Grad-CAM",

                "target_layer": (
                    target_layer_name
                ),

                "threshold_percentile": 80.0,

                "num_slices": (
                    gradcam_slice_count
                ),

                "slice_url": (
                    f"/outputs/"
                    f"{job_id}"
                    f"/xai/gradcam_slices"
                ),

                "heatmap_url": (
                    f"/outputs/"
                    f"{job_id}"
                    f"/xai/"
                    f"gradcam_original_space.nii.gz"
                ),

                "precision": (
                    gradcam_alignment[
                        "explanation_precision"
                    ]
                ),

                "coverage": (
                    gradcam_alignment[
                        "explanation_coverage"
                    ]
                ),

                "iou": (
                    gradcam_alignment[
                        "explanation_iou"
                    ]
                ),
            },

            "occlusion": {
                "method": (
                    "Occlusion Sensitivity"
                ),

                "threshold_percentile": 90.0,

                "block_size": [
                    16,
                    16,
                    16,
                ],

                "num_slices": (
                    occlusion_slice_count
                ),

                "slice_url": (
                    f"/outputs/"
                    f"{job_id}"
                    f"/xai/"
                    f"occlusion_slices"
                ),

                "heatmap_url": (
                    f"/outputs/"
                    f"{job_id}"
                    f"/xai/"
                    f"occlusion_original_space.nii.gz"
                ),

                "precision": (
                    occlusion_alignment[
                        "explanation_precision"
                    ]
                ),

                "coverage": (
                    occlusion_alignment[
                        "explanation_coverage"
                    ]
                ),

                "iou": (
                    occlusion_alignment[
                        "explanation_iou"
                    ]
                ),
            },
        }

    except Exception as error:

        print(
            "XAI error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"XAI generation failed: "
                f"{error}"
            ),
        )


# ============================================================
# DOWNLOAD NIFTI MASK
# ============================================================

@app.get(
    "/download/{job_id}"
)
async def download_segmentation(
    job_id: str,
):

    job_directory = (
        OUTPUT_DIR
        / job_id
    )

    if not job_directory.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Segmentation result not found."
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
                "Segmentation file not found."
            ),
        )

    segmentation_file = (
        segmentation_files[0]
    )

    return FileResponse(
        path=str(
            segmentation_file
        ),
        media_type="application/gzip",
        filename=(
            "spleen_segmentation.nii.gz"
        ),
    )


# ============================================================
# DOWNLOAD DICOM SEG
# ============================================================

@app.get(
    "/download-dicom-seg/{job_id}"
)
async def download_dicom_seg(
    job_id: str,
):

    seg_path = (
        OUTPUT_DIR
        / job_id
        / "dicom_seg"
        / "spleen_segmentation.dcm"
    )

    if not seg_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "DICOM SEG was not found "
                "for this job."
            ),
        )

    return FileResponse(
        path=str(seg_path),
        media_type="application/dicom",
        filename=(
            "spleen_segmentation.dcm"
        ),
    )