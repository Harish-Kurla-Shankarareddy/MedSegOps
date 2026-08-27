from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import nibabel as nib

from app.pipeline.inference import SpleenSegmenter
from evaluation.metrics import dice_score, iou_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

# Regression thresholds based on the measured five-case baseline.
MIN_MEAN_DICE = 0.95
MIN_MEAN_IOU = 0.92
MIN_CASE_DICE = 0.93
MIN_CASE_IOU = 0.88


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        return 1

    segmenter = SpleenSegmenter(
        model_path=MODEL_PATH,
        device="cpu",
    )

    results: list[dict[str, float | str]] = []

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

        if not image_path.exists():
            print(f"ERROR: missing image: {image_path}")
            return 1

        if not label_path.exists():
            print(f"ERROR: missing label: {label_path}")
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
                    f"ERROR: no prediction for spleen_{case_id}"
                )
                return 1

            prediction = (
                nib.load(
                    str(prediction_files[0])
                ).get_fdata()
                > 0
            )

            ground_truth = (
                nib.load(
                    str(label_path)
                ).get_fdata()
                > 0
            )

            if prediction.shape != ground_truth.shape:
                print(
                    f"ERROR: shape mismatch for spleen_{case_id}: "
                    f"{prediction.shape} vs "
                    f"{ground_truth.shape}"
                )
                return 1

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

    mean_dice = sum(
        float(r["dice"]) for r in results
    ) / len(results)

    mean_iou = sum(
        float(r["iou"]) for r in results
    ) / len(results)

    min_dice = min(
        float(r["dice"]) for r in results
    )

    min_iou = min(
        float(r["iou"]) for r in results
    )

    print("=" * 60)
    print("MedSegOps - Model Quality Gate")
    print("=" * 60)

    for result in results:
        print(
            f"spleen_{result['case']:>2} "
            f"Dice={float(result['dice']):.4f} "
            f"IoU={float(result['iou']):.4f}"
        )

    print()
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")
    print(f"Min Dice : {min_dice:.4f}")
    print(f"Min IoU  : {min_iou:.4f}")

    print()
    print("Required thresholds:")
    print(f"Mean Dice >= {MIN_MEAN_DICE:.4f}")
    print(f"Mean IoU  >= {MIN_MEAN_IOU:.4f}")
    print(f"Min Dice  >= {MIN_CASE_DICE:.4f}")
    print(f"Min IoU   >= {MIN_CASE_IOU:.4f}")

    quality_passed = (
        mean_dice >= MIN_MEAN_DICE
        and mean_iou >= MIN_MEAN_IOU
        and min_dice >= MIN_CASE_DICE
        and min_iou >= MIN_CASE_IOU
    )

    print()

    if quality_passed:
        print("QUALITY GATE: PASS")
        return 0

    print("QUALITY GATE: FAIL")

    if mean_dice < MIN_MEAN_DICE:
        print("  - Mean Dice below threshold")

    if mean_iou < MIN_MEAN_IOU:
        print("  - Mean IoU below threshold")

    if min_dice < MIN_CASE_DICE:
        print("  - Minimum Dice below threshold")

    if min_iou < MIN_CASE_IOU:
        print("  - Minimum IoU below threshold")

    return 1


if __name__ == "__main__":
    sys.exit(main())