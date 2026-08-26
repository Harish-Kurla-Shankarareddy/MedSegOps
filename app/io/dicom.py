from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


class DicomSeriesError(RuntimeError):
    """Raised when a usable DICOM CT series cannot be found."""


def find_dicom_series(
    directory: str | Path,
) -> list[dict]:
    """
    Discover DICOM series inside a directory.

    Returns a list containing:
        series_uid
        file_count
        modality
        description
        files
        image
    """

    directory = Path(directory)

    if not directory.exists():
        raise DicomSeriesError(
            f"DICOM directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise DicomSeriesError(
            f"DICOM input must be a directory: {directory}"
        )

    series_ids = (
        sitk.ImageSeriesReader.GetGDCMSeriesIDs(
            str(directory)
        )
    )

    if not series_ids:
        raise DicomSeriesError(
            "No DICOM series were found in the uploaded directory."
        )

    results = []

    for series_uid in series_ids:

        files = (
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(directory),
                series_uid,
            )
        )

        if not files:
            continue

        reader = sitk.ImageSeriesReader()

        reader.SetFileNames(files)

        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()

        modality = ""

        description = ""

        try:
            modality = reader.GetMetaData(
                0,
                "0008|0060",
            )
        except Exception:
            pass

        try:
            description = reader.GetMetaData(
                0,
                "0008|103e",
            )
        except Exception:
            pass

        results.append(
            {
                "series_uid": series_uid,
                "file_count": len(files),
                "modality": modality,
                "description": description,
                "files": files,
                "image": image,
            }
        )

    return results


def select_ct_series(
    directory: str | Path,
) -> dict:
    """
    Select a CT DICOM series.

    If exactly one CT series exists, it is selected.

    If multiple CT series exist, the function currently
    refuses to guess and asks the caller to provide a
    cleaner single-series directory.
    """

    series = find_dicom_series(
        directory
    )

    ct_series = [
        item
        for item in series
        if item["modality"].upper() == "CT"
    ]

    if not ct_series:

        available = [
            {
                "uid": item["series_uid"],
                "modality": item["modality"],
                "description": item["description"],
                "files": item["file_count"],
            }
            for item in series
        ]

        raise DicomSeriesError(
            "No CT DICOM series was found. "
            f"Available series: {available}"
        )

    if len(ct_series) > 1:

        available = [
            {
                "uid": item["series_uid"],
                "description": item["description"],
                "files": item["file_count"],
            }
            for item in ct_series
        ]

        raise DicomSeriesError(
            "Multiple CT DICOM series were found. "
            "Please upload one DICOM series at a time. "
            f"Found: {available}"
        )

    return ct_series[0]


def convert_dicom_series_to_nifti(
    dicom_directory: str | Path,
    output_path: str | Path,
) -> dict:
    """
    Convert a single DICOM CT series into NIfTI.

    The output NIfTI is then compatible with the
    existing MedSegOps MONAI inference pipeline.
    """

    dicom_directory = Path(
        dicom_directory
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = select_ct_series(
        dicom_directory
    )

    image = selected["image"]

    if image.GetDimension() != 3:

        raise DicomSeriesError(
            "Expected a 3D DICOM CT volume, "
            f"but got dimension {image.GetDimension()}."
        )

    size = image.GetSize()
    spacing = image.GetSpacing()

    sitk.WriteImage(
        image,
        str(output_path),
        useCompression=True,
    )

    return {
        "series_uid": selected[
            "series_uid"
        ],
        "modality": selected[
            "modality"
        ],
        "description": selected[
            "description"
        ],
        "file_count": selected[
            "file_count"
        ],
        "dimensions": {
            "x": int(size[0]),
            "y": int(size[1]),
            "z": int(size[2]),
        },
        "spacing_mm": {
            "x": float(spacing[0]),
            "y": float(spacing[1]),
            "z": float(spacing[2]),
        },
        "nifti_path": str(
            output_path
        ),
    }