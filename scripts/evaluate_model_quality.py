from pathlib import Path
import tempfile

import nibabel as nib

from app.pipeline.inference import SpleenSegmenter
from evaluation.metrics import dice_score, iou_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spleen"
    / "Task09_Spleen"
)

VALIDATION_CASES = [
    "10",
    "12",
    "13",
    "14",
    "16",
]


def main():
    segmenter = SpleenSegmenter(
        model_path=PROJECT_ROOT / "models" / "monai" / "model.pt",
        device="cpu",
    )

    results = []

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
            raise FileNotFoundError(image_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            segmenter.predict(
                image_path=image_path,
                output_dir=tmp_dir,
            )

            prediction_files = list(
                Path(tmp_dir).glob("*_seg.nii.gz")
            )

            if not prediction_files:
                raise RuntimeError(
                    f"No prediction generated for spleen_{case_id}"
                )

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
                raise RuntimeError(
                    f"Shape mismatch for spleen_{case_id}: "
                    f"{prediction.shape} vs {ground_truth.shape}"
                )

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
        result["dice"]
        for result in results
    ) / len(results)

    mean_iou = sum(
        result["iou"]
        for result in results
    ) / len(results)

    min_dice = min(
        result["dice"]
        for result in results
    )

    min_iou = min(
        result["iou"]
        for result in results
    )

    print("=" * 60)
    print("MedSegOps - Model Quality Benchmark")
    print("=" * 60)

    for result in results:
        print(
            f"spleen_{result['case']:>2} "
            f"Dice={result['dice']:.4f} "
            f"IoU={result['iou']:.4f}"
        )

    print()
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")
    print(f"Min Dice : {min_dice:.4f}")
    print(f"Min IoU  : {min_iou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()