import numpy as np
import nibabel as nib


def load_mask(mask_path):
    """
    Load a segmentation mask from a NIfTI file.

    Parameters
    ----------
    mask_path : str or pathlib.Path
        Path to the NIfTI mask file.

    Returns
    -------
    tuple
        mask : numpy.ndarray
            Segmentation mask as a NumPy array.

        voxel_spacing : tuple
            Voxel spacing in millimeters as:
            (x_spacing, y_spacing, z_spacing)
    """

    image = nib.load(str(mask_path))

    mask = image.get_fdata()

    voxel_spacing = image.header.get_zooms()[:3]

    return mask, voxel_spacing


def dice_score(prediction, ground_truth):
    """
    Calculate the Dice Similarity Coefficient.

    Parameters
    ----------
    prediction : numpy.ndarray
        Predicted segmentation mask.

    ground_truth : numpy.ndarray
        Ground truth segmentation mask.

    Returns
    -------
    float
        Dice score between 0 and 1.
    """

    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    total = prediction.sum() + ground_truth.sum()

    if total == 0:
        return 1.0

    dice = (2.0 * intersection) / total

    return float(dice)


def iou_score(prediction, ground_truth):
    """
    Calculate Intersection over Union (IoU).

    Parameters
    ----------
    prediction : numpy.ndarray
        Predicted segmentation mask.

    ground_truth : numpy.ndarray
        Ground truth segmentation mask.

    Returns
    -------
    float
        IoU score between 0 and 1.
    """

    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:
        return 1.0

    iou = intersection / union

    return float(iou)


def voxel_volume_mm3(voxel_spacing):
    """
    Calculate the physical volume of one voxel.

    Parameters
    ----------
    voxel_spacing : tuple or list
        Voxel spacing in millimeters.

    Example
    -------
    (0.9766, 0.9766, 5.0)

    Returns
    -------
    float
        Volume of one voxel in cubic millimeters.
    """

    spacing = np.asarray(
        voxel_spacing,
        dtype=float
    )

    if spacing.size != 3:
        raise ValueError(
            "voxel_spacing must contain exactly 3 values."
        )

    return float(np.prod(spacing))


def mask_volume_mm3(mask, voxel_spacing):
    """
    Calculate segmentation mask volume in cubic millimeters.
    """

    mask = np.asarray(mask)

    voxel_count = np.count_nonzero(mask)

    single_voxel_volume = voxel_volume_mm3(
        voxel_spacing
    )

    volume = voxel_count * single_voxel_volume

    return float(volume)


def mask_volume_ml(mask, voxel_spacing):
    """
    Calculate segmentation mask volume in milliliters.

    1 mL = 1000 mm³.
    """

    volume_mm3 = mask_volume_mm3(
        mask,
        voxel_spacing
    )

    volume_ml = volume_mm3 / 1000.0

    return float(volume_ml)


def evaluate_segmentation(prediction, ground_truth):
    """
    Calculate segmentation quality metrics.

    Returns
    -------
    dict
        Dictionary containing Dice and IoU scores.
    """

    return {
        "dice": dice_score(
            prediction,
            ground_truth
        ),
        "iou": iou_score(
            prediction,
            ground_truth
        ),
    }


def calculate_segmentation_statistics(
    mask,
    voxel_spacing,
    image_shape=None
):
    """
    Calculate quantitative segmentation statistics.

    Returns
    -------
    dict
        Dictionary containing segmentation statistics.
    """

    mask = np.asarray(mask)

    voxel_count = int(
        np.count_nonzero(mask)
    )

    single_voxel_volume = voxel_volume_mm3(
        voxel_spacing
    )

    volume_mm3 = mask_volume_mm3(
        mask,
        voxel_spacing
    )

    volume_ml = mask_volume_ml(
        mask,
        voxel_spacing
    )

    if image_shape is None:
        image_shape = mask.shape

    return {
        "voxel_count": voxel_count,
        "volume_mm3": round(volume_mm3, 2),
        "volume_ml": round(volume_ml, 2),
        "voxel_spacing": [
            float(value)
            for value in voxel_spacing
        ],
        "single_voxel_volume_mm3": round(
            single_voxel_volume,
            4
        ),
        "image_shape": [
            int(value)
            for value in image_shape
        ],
    }