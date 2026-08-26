from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


class SpleenOcclusion:
    """
    3D occlusion sensitivity for the MONAI spleen
    segmentation model.

    Occlusion is performed on a 96 x 96 x 96 ROI
    centered around the predicted spleen.

    The ROI heatmap is then inserted back into the
    complete MONAI-preprocessed volume so that it
    can be resampled into the original CT space.
    """

    def __init__(
        self,
        segmenter,
        roi_size=(96, 96, 96),
        block_size=(16, 16, 16),
    ):
        self.segmenter = segmenter
        self.model = segmenter.model
        self.device = segmenter.device

        self.roi_size = tuple(roi_size)
        self.block_size = tuple(block_size)

    # ==================================================
    # PREPROCESSING
    # ==================================================

    def _preprocess(
        self,
        image_path: str | Path,
    ):
        """
        Use the exact same preprocessing pipeline
        as the segmentation model.
        """

        data = self.segmenter.preprocessing(
            {
                "image": str(image_path)
            }
        )

        processed_image = data["image"]

        affine = np.asarray(
            processed_image.affine.detach().cpu()
        )

        image = processed_image.unsqueeze(0)

        image = image.to(
            self.device
        )

        return image, affine

    # ==================================================
    # DIRECT PATCH PREDICTION
    # ==================================================

    @torch.inference_mode()
    def _predict_patch(
        self,
        patch: torch.Tensor,
    ):
        """
        Run direct inference on a valid 96^3 patch.
        """

        logits = self.model(
            patch
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        spleen_probability = (
            probabilities[:, 1]
        )

        prediction = torch.argmax(
            probabilities,
            dim=1,
        )

        spleen_mask = (
            prediction == 1
        )

        if spleen_mask.any():

            score = (
                spleen_probability[
                    spleen_mask
                ].mean()
            )

        else:

            score = (
                spleen_probability.mean()
            )

        return float(
            score.detach().cpu()
        )

    # ==================================================
    # FIND SPLEEN CENTER
    # ==================================================

    def _find_spleen_center(
        self,
        prediction: torch.Tensor,
    ):
        """
        Find the center of the predicted spleen.
        """

        predicted_class = torch.argmax(
            prediction,
            dim=1,
        )[0]

        spleen_mask = (
            predicted_class == 1
        )

        coordinates = torch.nonzero(
            spleen_mask,
            as_tuple=False,
        )

        if coordinates.numel() == 0:

            shape = predicted_class.shape

            print(
                "No spleen prediction found. "
                "Using volume center."
            )

            return (
                shape[0] // 2,
                shape[1] // 2,
                shape[2] // 2,
            )

        mins = coordinates.min(
            dim=0
        ).values

        maxs = coordinates.max(
            dim=0
        ).values

        center = (
            mins + maxs
        ) // 2

        center_tuple = tuple(
            int(value)
            for value in (
                center
                .detach()
                .cpu()
                .tolist()
            )
        )

        print(
            "Predicted spleen center:",
            list(center_tuple),
        )

        return center_tuple

    # ==================================================
    # EXTRACT ROI
    # ==================================================

    def _extract_roi(
        self,
        image: torch.Tensor,
        center,
    ):
        """
        Extract a fixed-size 96^3 ROI.

        Returns:
            roi
            start coordinate in full volume
        """

        roi_d, roi_h, roi_w = self.roi_size

        _, _, depth, height, width = image.shape

        center_d, center_h, center_w = center

        start_d = center_d - roi_d // 2
        start_h = center_h - roi_h // 2
        start_w = center_w - roi_w // 2

        end_d = start_d + roi_d
        end_h = start_h + roi_h
        end_w = start_w + roi_w

        src_start_d = max(start_d, 0)
        src_start_h = max(start_h, 0)
        src_start_w = max(start_w, 0)

        src_end_d = min(end_d, depth)
        src_end_h = min(end_h, height)
        src_end_w = min(end_w, width)

        roi = torch.zeros(
            (
                1,
                1,
                roi_d,
                roi_h,
                roi_w,
            ),
            device=image.device,
            dtype=image.dtype,
        )

        dst_start_d = src_start_d - start_d
        dst_start_h = src_start_h - start_h
        dst_start_w = src_start_w - start_w

        dst_end_d = (
            dst_start_d
            + src_end_d
            - src_start_d
        )

        dst_end_h = (
            dst_start_h
            + src_end_h
            - src_start_h
        )

        dst_end_w = (
            dst_start_w
            + src_end_w
            - src_start_w
        )

        roi[
            :,
            :,
            dst_start_d:dst_end_d,
            dst_start_h:dst_end_h,
            dst_start_w:dst_end_w,
        ] = image[
            :,
            :,
            src_start_d:src_end_d,
            src_start_h:src_end_h,
            src_start_w:src_end_w,
        ]

        return roi, (
            start_d,
            start_h,
            start_w,
        )

    # ==================================================
    # GENERATE FULL OCCLUSION MAP
    # ==================================================

    def generate(
        self,
        image_path: str | Path,
    ):
        """
        Generate a full-volume 3D occlusion map.

        The actual occlusion experiment is performed
        inside the 96^3 spleen ROI.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Input image not found: "
                f"{image_path}"
            )

        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        print(
            "Preprocessing input..."
        )

        image, affine = (
            self._preprocess(
                image_path
            )
        )

        full_shape = tuple(
            int(value)
            for value in image.shape[2:]
        )

        print(
            "Preprocessed shape:",
            tuple(image.shape),
        )

        # --------------------------------------------------
        # Baseline full prediction
        # --------------------------------------------------

        print(
            "Running baseline "
            "sliding-window inference..."
        )

        with torch.inference_mode():

            full_prediction = (
                self.segmenter.inferer(
                    inputs=image,
                    network=self.model,
                )
            )

        print(
            "Baseline inference complete."
        )

        # --------------------------------------------------
        # Find spleen
        # --------------------------------------------------

        center = (
            self._find_spleen_center(
                full_prediction
            )
        )

        # --------------------------------------------------
        # Extract ROI
        # --------------------------------------------------

        roi, roi_start = (
            self._extract_roi(
                image,
                center,
            )
        )

        print(
            "ROI shape:",
            tuple(roi.shape),
        )

        print(
            "ROI start:",
            roi_start,
        )

        # --------------------------------------------------
        # Baseline patch score
        # --------------------------------------------------

        baseline_score = (
            self._predict_patch(
                roi
            )
        )

        print(
            f"Baseline spleen score: "
            f"{baseline_score:.6f}"
        )

        # --------------------------------------------------
        # Create ROI heatmap
        # --------------------------------------------------

        roi_d, roi_h, roi_w = (
            self.roi_size
        )

        block_d, block_h, block_w = (
            self.block_size
        )

        roi_heatmap = np.zeros(
            (
                roi_d,
                roi_h,
                roi_w,
            ),
            dtype=np.float32,
        )

        num_d = roi_d // block_d
        num_h = roi_h // block_h
        num_w = roi_w // block_w

        total_blocks = (
            num_d
            * num_h
            * num_w
        )

        print(
            f"Occlusion block size: "
            f"{self.block_size}"
        )

        print(
            f"Total occlusion blocks: "
            f"{total_blocks}"
        )

        processed = 0

        # --------------------------------------------------
        # Occlusion loop
        # --------------------------------------------------

        for d in range(num_d):

            for h in range(num_h):

                for w in range(num_w):

                    start_d = (
                        d * block_d
                    )

                    end_d = (
                        start_d + block_d
                    )

                    start_h = (
                        h * block_h
                    )

                    end_h = (
                        start_h + block_h
                    )

                    start_w = (
                        w * block_w
                    )

                    end_w = (
                        start_w + block_w
                    )

                    # Copy ROI

                    occluded = roi.clone()

                    # Zero out block

                    occluded[
                        :,
                        :,
                        start_d:end_d,
                        start_h:end_h,
                        start_w:end_w,
                    ] = 0.0

                    # Run model

                    occluded_score = (
                        self._predict_patch(
                            occluded
                        )
                    )

                    # Positive = confidence decrease

                    score_drop = (
                        baseline_score
                        - occluded_score
                    )

                    roi_heatmap[
                        start_d:end_d,
                        start_h:end_h,
                        start_w:end_w,
                    ] = score_drop

                    processed += 1

                    print(
                        f"Occlusion "
                        f"{processed}/"
                        f"{total_blocks} "
                        f"| score drop: "
                        f"{score_drop:.6f}"
                    )

        # --------------------------------------------------
        # Remove negative importance
        # --------------------------------------------------

        roi_heatmap = np.maximum(
            roi_heatmap,
            0.0,
        )

        # --------------------------------------------------
        # Normalize ROI
        # --------------------------------------------------

        roi_max = roi_heatmap.max()

        if roi_max > 0:

            roi_heatmap = (
                roi_heatmap / roi_max
            )

        # --------------------------------------------------
        # Insert ROI into complete volume
        # --------------------------------------------------

        full_heatmap = np.zeros(
            full_shape,
            dtype=np.float32,
        )

        start_d, start_h, start_w = (
            roi_start
        )

        dst_start_d = max(
            start_d,
            0,
        )

        dst_start_h = max(
            start_h,
            0,
        )

        dst_start_w = max(
            start_w,
            0,
        )

        dst_end_d = min(
            start_d + roi_d,
            full_shape[0],
        )

        dst_end_h = min(
            start_h + roi_h,
            full_shape[1],
        )

        dst_end_w = min(
            start_w + roi_w,
            full_shape[2],
        )

        src_start_d = max(
            -start_d,
            0,
        )

        src_start_h = max(
            -start_h,
            0,
        )

        src_start_w = max(
            -start_w,
            0,
        )

        src_end_d = (
            src_start_d
            + dst_end_d
            - dst_start_d
        )

        src_end_h = (
            src_start_h
            + dst_end_h
            - dst_start_h
        )

        src_end_w = (
            src_start_w
            + dst_end_w
            - dst_start_w
        )

        full_heatmap[
            dst_start_d:dst_end_d,
            dst_start_h:dst_end_h,
            dst_start_w:dst_end_w,
        ] = roi_heatmap[
            src_start_d:src_end_d,
            src_start_h:src_end_h,
            src_start_w:src_end_w,
        ]

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        processed_input = (
            image[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return {
            "heatmap": full_heatmap,

            "roi_heatmap": roi_heatmap,

            "input": processed_input,

            "target_score": baseline_score,

            "center": center,

            "roi_start": roi_start,

            "roi_size": self.roi_size,

            "block_size": self.block_size,

            "preprocessed_shape": full_shape,

            "preprocessed_affine": affine,
        }

    # ==================================================
    # SAVE NIFTI
    # ==================================================

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        output_path: str | Path,
        affine=None,
    ):
        """
        Save heatmap to NIfTI.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if affine is None:

            affine = np.eye(4)

        nifti = nib.Nifti1Image(
            heatmap.astype(np.float32),
            affine=np.asarray(affine),
        )

        nib.save(
            nifti,
            str(output_path),
        )

        print(
            f"Occlusion heatmap saved: "
            f"{output_path}"
        )

    # ==================================================
    # SAVE PNG SLICES
    # ==================================================

    def save_slice_heatmaps(
        self,
        heatmap: np.ndarray,
        output_directory: str | Path,
    ):
        """
        Save full-volume occlusion slices.
        """

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        num_slices = heatmap.shape[2]

        print(
            f"Creating {num_slices} "
            "occlusion heatmap slices..."
        )

        for slice_index in range(
            num_slices
        ):

            current_slice = heatmap[
                :,
                :,
                slice_index,
            ]

            current_slice = np.rot90(
                current_slice
            )

            output_path = (
                output_directory
                / f"slice_{slice_index:03d}.png"
            )

            plt.figure(
                figsize=(6, 6)
            )

            plt.imshow(
                current_slice,
                cmap="jet",
                vmin=0,
                vmax=1,
            )

            plt.title(
                f"Occlusion - Slice {slice_index}"
            )

            plt.axis("off")

            plt.savefig(
                output_path,
                dpi=120,
                bbox_inches="tight",
            )

            plt.close()

        print(
            f"Occlusion slices saved to: "
            f"{output_directory}"
        )