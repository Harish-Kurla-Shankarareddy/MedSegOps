from __future__ import annotations

import sys
import tempfile
from pathlib import Path


# ------------------------------------------------------------
# Make the repository root importable when this file is
# executed directly with:
#
#     python scripts/check_model_quality.py
#
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import nibabel as nib

from app.pipeline.inference import SpleenSegmenter
from evaluation.metrics import dice_score, iou_score


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET_ROOT = (
    PROJECT_ROOT
    / ".ci_data"
    / "Task09_Spleen"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "monai"
    / "model.pt"
)


VALIDATION_CASES = [
    "10",
    "12",
    "13",
    "14",
    "16",
]


# ------------------------------------------------------------
# Regression thresholds
# ------------------------------------------------------------

MIN_MEAN_DICE = 0.95
MIN_MEAN_IOU = 0.92

MIN_CASE_DICE = 0.93
MIN_CASE_IOU = 0.88


def main() -> int:
    """Run the model-quality regression gate."""

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        print(
            f"ERROR: model not found:\n"
            f"  {MODEL_PATH}"
        )
        return 1

    if not DATASET_ROOT.exists():
        print(
            f"ERROR: validation dataset not found:\n"
            f"  {DATASET_ROOT}"
        )
        return 1

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("=" * 60)
    print("MedSegOps - Model Quality Gate")
    print("=" * 60)
    print()
    print("Loading segmentation model...")

    segmenter = SpleenSegmenter(
        model_path=MODEL_PATH,
        device="cpu",
    )

    print("Model loaded successfully.")
    print()

    results: list[dict[str, float | str]] = []

    # --------------------------------------------------------
    # Evaluate validation cases
    # --------------------------------------------------------

    for case_id in VALIDATION_CASES:

        image_path = (
            DATASET_ROOT
            / "imagesTr"
            / f"spleen_{case_id}.nii.gz"
        )

        label_path = (
            DATASET_ROOT
            / "labelsTr"
            / f"spleen_{case_id}.nii.gz"
        )

        print(
            f"Evaluating spleen_{case_id}..."
        )

        if not image_path.exists():
            print(
                f"ERROR: missing image:\n"
                f"  {image_path}"
            )
            return 1

        if not label_path.exists():
            print(
                f"ERROR: missing label:\n"
                f"  {label_path}"
            )
            return 1

        with tempfile.TemporaryDirectory() as tmp_dir:

            segmenter.predict(
                image_path=image_path,
                output_dir=tmp_dir,
            )

            prediction_files = list(
                Path(tmp_dir).glob("*_seg.nii.gz")
            )

            if not prediction_files:
                print(
                    f"ERROR: no prediction generated "
                    f"for spleen_{case_id}"
                )
                return 1

            prediction = (
                nib.load(
                    str(prediction_files[0])
                )
                .get_fdata()
                > 0
            )

            ground_truth = (
                nib.load(
                    str(label_path)
                )
                .get_fdata()
                > 0
            )

            # ------------------------------------------------
            # Geometry check
            # ------------------------------------------------

            if prediction.shape != ground_truth.shape:
                print(
                    f"ERROR: shape mismatch for "
                    f"spleen_{case_id}"
                )
                print(
                    f"  prediction: {prediction.shape}"
                )
                print(
                    f"  ground truth: {ground_truth.shape}"
                )
                return 1

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            dice = dice_score(
                prediction,
                ground_truth,
            )

            iou = iou_score(
                prediction,
                ground_truth,
            )

            results.append(
                {
                    "case": case_id,
                    "dice": dice,
                    "iou": iou,
                }
            )

            print(
                f"  Dice: {dice:.4f}"
            )

            print(
                f"  IoU : {iou:.4f}"
            )

            print()

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    mean_dice = (
        sum(
            float(result["dice"])
            for result in results
        )
        / len(results)
    )

    mean_iou = (
        sum(
            float(result["iou"])
            for result in results
        )
        / len(results)
    )

    min_dice = min(
        float(result["dice"])
        for result in results
    )

    min_iou = min(
        float(result["iou"])
        for result in results
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL QUALITY RESULTS")
    print("=" * 60)

    for result in results:
        print(
            f"spleen_{result['case']:>2} "
            f"Dice={float(result['dice']):.4f} "
            f"IoU={float(result['iou']):.4f}"
        )

    print()

    print(
        f"Mean Dice: {mean_dice:.4f}"
    )

    print(
        f"Mean IoU : {mean_iou:.4f}"
    )

    print(
        f"Min Dice : {min_dice:.4f}"
    )

    print(
        f"Min IoU  : {min_iou:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    print("Required thresholds:")
    print(
        f"Mean Dice >= {MIN_MEAN_DICE:.4f}"
    )
    print(
        f"Mean IoU  >= {MIN_MEAN_IOU:.4f}"
    )
    print(
        f"Min Dice  >= {MIN_CASE_DICE:.4f}"
    )
    print(
        f"Min IoU   >= {MIN_CASE_IOU:.4f}"
    )

    # --------------------------------------------------------
    # Quality decision
    # --------------------------------------------------------

    failures: list[str] = []

    if mean_dice < MIN_MEAN_DICE:
        failures.append(
            f"Mean Dice {mean_dice:.4f} "
            f"< {MIN_MEAN_DICE:.4f}"
        )

    if mean_iou < MIN_MEAN_IOU:
        failures.append(
            f"Mean IoU {mean_iou:.4f} "
            f"< {MIN_MEAN_IOU:.4f}"
        )

    if min_dice < MIN_CASE_DICE:
        failures.append(
            f"Min Dice {min_dice:.4f} "
            f"< {MIN_CASE_DICE:.4f}"
        )

    if min_iou < MIN_CASE_IOU:
        failures.append(
            f"Min IoU {min_iou:.4f} "
            f"< {MIN_CASE_IOU:.4f}"
        )

    print()

    if failures:
        print("QUALITY GATE: FAIL")
        print()

        for failure in failures:
            print(f"  - {failure}")

        print()

        return 1

    print("QUALITY GATE: PASS")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())