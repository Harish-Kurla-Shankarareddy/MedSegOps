from __future__ import annotations

import numpy as np


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert an array to a boolean binary mask.
    """
    return np.asarray(mask) > 0


def threshold_gradcam(
    heatmap: np.ndarray,
    percentile: float = 90.0,
) -> np.ndarray:
    """
    Convert a continuous Grad-CAM heatmap into a
    binary high-activation explanation mask.

    Parameters
    ----------
    heatmap:
        3D Grad-CAM heatmap normalized to [0, 1].

    percentile:
        Activation percentile used as the threshold.

    Returns
    -------
    np.ndarray
        Boolean 3D high-activation mask.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    if heatmap.ndim != 3:
        raise ValueError(
            f"Expected a 3D heatmap, got {heatmap.shape}"
        )

    active_values = heatmap[heatmap > 0]

    if active_values.size == 0:
        return np.zeros(
            heatmap.shape,
            dtype=bool,
        )

    threshold = np.percentile(
        active_values,
        percentile,
    )

    return heatmap >= threshold


def explanation_precision(
    explanation_mask: np.ndarray,
    segmentation_mask: np.ndarray,
) -> float:
    """
    Fraction of high-activation explanation voxels
    that fall inside the predicted segmentation.

    Precision =
        explanation ∩ segmentation
        ----------------------------
        explanation
    """

    explanation = _to_binary_mask(
        explanation_mask
    )

    segmentation = _to_binary_mask(
        segmentation_mask
    )

    explanation_size = int(
        np.count_nonzero(explanation)
    )

    if explanation_size == 0:
        return 0.0

    intersection = np.logical_and(
        explanation,
        segmentation,
    )

    return float(
        np.count_nonzero(intersection)
        / explanation_size
    )


def explanation_coverage(
    explanation_mask: np.ndarray,
    segmentation_mask: np.ndarray,
) -> float:
    """
    Fraction of the predicted segmentation covered
    by high-activation explanation voxels.

    Coverage =
        explanation ∩ segmentation
        ----------------------------
        segmentation
    """

    explanation = _to_binary_mask(
        explanation_mask
    )

    segmentation = _to_binary_mask(
        segmentation_mask
    )

    segmentation_size = int(
        np.count_nonzero(segmentation)
    )

    if segmentation_size == 0:
        return 0.0

    intersection = np.logical_and(
        explanation,
        segmentation,
    )

    return float(
        np.count_nonzero(intersection)
        / segmentation_size
    )


def explanation_iou(
    explanation_mask: np.ndarray,
    segmentation_mask: np.ndarray,
) -> float:
    """
    IoU between the high-activation explanation
    region and the predicted segmentation.
    """

    explanation = _to_binary_mask(
        explanation_mask
    )

    segmentation = _to_binary_mask(
        segmentation_mask
    )

    intersection = np.logical_and(
        explanation,
        segmentation,
    )

    union = np.logical_or(
        explanation,
        segmentation,
    )

    intersection_count = np.count_nonzero(
        intersection
    )

    union_count = np.count_nonzero(
        union
    )

    if union_count == 0:
        return 0.0

    return float(
        intersection_count
        / union_count
    )


def evaluate_explanation_alignment(
    heatmap: np.ndarray,
    segmentation_mask: np.ndarray,
    percentile: float = 90.0,
) -> dict:
    """
    Calculate a complete Grad-CAM alignment report.

    Parameters
    ----------
    heatmap:
        Original-space 3D Grad-CAM heatmap.

    segmentation_mask:
        Original-space predicted binary spleen mask.

    percentile:
        Threshold used to identify high-activation CAM voxels.

    Returns
    -------
    dict
        Alignment metrics and metadata.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    segmentation_mask = np.asarray(
        segmentation_mask
    )

    if heatmap.shape != segmentation_mask.shape:
        raise ValueError(
            "Heatmap and segmentation mask "
            f"must have identical shapes. "
            f"Got {heatmap.shape} and "
            f"{segmentation_mask.shape}."
        )

    high_activation = threshold_gradcam(
        heatmap,
        percentile=percentile,
    )

    segmentation = _to_binary_mask(
        segmentation_mask
    )

    return {
        "threshold_percentile": float(
            percentile
        ),
        "cam_threshold": float(
            np.percentile(
                heatmap[heatmap > 0],
                percentile,
            )
            if np.any(heatmap > 0)
            else 0.0
        ),
        "high_activation_voxels": int(
            np.count_nonzero(
                high_activation
            )
        ),
        "segmentation_voxels": int(
            np.count_nonzero(
                segmentation
            )
        ),
        "intersection_voxels": int(
            np.count_nonzero(
                np.logical_and(
                    high_activation,
                    segmentation,
                )
            )
        ),
        "explanation_precision": round(
            explanation_precision(
                high_activation,
                segmentation,
            ),
            4,
        ),
        "explanation_coverage": round(
            explanation_coverage(
                high_activation,
                segmentation,
            ),
            4,
        ),
        "explanation_iou": round(
            explanation_iou(
                high_activation,
                segmentation,
            ),
            4,
        ),
    }