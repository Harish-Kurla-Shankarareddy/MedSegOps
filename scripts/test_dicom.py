from pathlib import Path

from app.io.dicom import (
    DicomSeriesError,
    convert_dicom_series_to_nifti,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# --------------------------------------------------
# Change this to your DICOM test directory
# --------------------------------------------------

DICOM_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dicom_test"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dicom_test"
    / "converted.nii.gz"
)


print("=" * 60)
print("MedSegOps - DICOM Series Test")
print("=" * 60)

print()
print(
    "DICOM directory:",
    DICOM_DIRECTORY
)

print(
    "Output NIfTI:",
    OUTPUT_PATH
)


try:

    result = (
        convert_dicom_series_to_nifti(
            DICOM_DIRECTORY,
            OUTPUT_PATH,
        )
    )

except DicomSeriesError as error:

    print()
    print("DICOM ERROR:")
    print(error)
    raise SystemExit(1)


print()
print("=" * 60)
print("DICOM CONVERSION SUCCESS")
print("=" * 60)

print(
    "Series UID:",
    result["series_uid"]
)

print(
    "Modality:",
    result["modality"]
)

print(
    "Description:",
    result["description"]
)

print(
    "Number of files:",
    result["file_count"]
)

print(
    "Dimensions:",
    result["dimensions"]
)

print(
    "Spacing:",
    result["spacing_mm"]
)

print(
    "NIfTI:",
    result["nifti_path"]
)