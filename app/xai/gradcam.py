from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpleenGradCAM:
    """
    Patch-based 3D Grad-CAM for the MONAI spleen U-Net.

    Model classes:
        0 = background
        1 = spleen

    Grad-CAM is calculated in MONAI's preprocessed
    model space, and the resulting heatmap retains
    the preprocessing affine so it can later be
    transformed back to original CT space.
    """

    def __init__(
        self,
        segmenter,
        target_layer: nn.Module | None = None,
        roi_size=(96, 96, 96),
    ):
        self.segmenter = segmenter
        self.model = segmenter.model
        self.device = segmenter.device
        self.roi_size = tuple(roi_size)

        self.activations = None

        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer

        self.hook = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    # --------------------------------------------------
    # Find target layer
    # --------------------------------------------------

    def _find_target_layer(self) -> nn.Module:
        candidates = []

        for name, module in self.model.named_modules():

            if isinstance(module, nn.Conv3d):

                if module.out_channels >= 16:

                    candidates.append(
                        (
                            name,
                            module,
                        )
                    )

        if not candidates:

            raise RuntimeError(
                "Could not find a suitable Conv3d "
                "layer for Grad-CAM."
            )

        name, layer = max(
            candidates,
            key=lambda item: item[1].out_channels,
        )

        print(
            f"Grad-CAM target layer: {name}"
        )

        print(
            f"Target layer: {layer}"
        )

        print(
            f"Output channels: "
            f"{layer.out_channels}"
        )

        return layer

    # --------------------------------------------------
    # Forward hook
    # --------------------------------------------------

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    # --------------------------------------------------
    # Preprocess input
    # --------------------------------------------------

    def _preprocess(
        self,
        image_path: Path,
    ):
        """
        Apply the exact same preprocessing pipeline
        used by the segmentation model.

        Returns:
            image tensor
            preprocessing affine
        """

        data = self.segmenter.preprocessing(
            {
                "image": str(image_path),
            }
        )

        processed_image = data["image"]

        # --------------------------------------------------
        # Get affine BEFORE converting to a plain tensor
        # --------------------------------------------------

        affine = np.asarray(
            processed_image.affine.detach().cpu()
        )

        image = processed_image.unsqueeze(0)

        image = image.to(self.device)

        return image, affine

    # --------------------------------------------------
    # Normal inference
    # --------------------------------------------------

    @torch.inference_mode()
    def _get_prediction(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        prediction = self.segmenter.inferer(
            inputs=image,
            network=self.model,
        )

        return prediction

    # --------------------------------------------------
    # Find spleen center
    # --------------------------------------------------

    def _find_spleen_center(
        self,
        prediction: torch.Tensor,
    ):

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

            print(
                "No spleen prediction found."
            )

            shape = predicted_class.shape

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

        print(
            "Predicted spleen bounding box:"
        )

        print(
            "min:",
            mins.detach().cpu().tolist()
        )

        print(
            "max:",
            maxs.detach().cpu().tolist()
        )

        print(
            "center:",
            center.detach().cpu().tolist()
        )

        return tuple(
            int(value)
            for value in (
                center
                .detach()
                .cpu()
                .tolist()
            )
        )

    # --------------------------------------------------
    # Extract 96³ patch
    # --------------------------------------------------

    def _extract_patch(
        self,
        image: torch.Tensor,
        center,
    ):

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

        patch = torch.zeros(
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

        dst_end_d = dst_start_d + (
            src_end_d - src_start_d
        )

        dst_end_h = dst_start_h + (
            src_end_h - src_start_h
        )

        dst_end_w = dst_start_w + (
            src_end_w - src_start_w
        )

        patch[
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

        return patch, {
            "start": (
                start_d,
                start_h,
                start_w,
            ),
        }

    # --------------------------------------------------
    # Grad-CAM on 96³ patch
    # --------------------------------------------------

    @torch.enable_grad()
    def _generate_patch_cam(
        self,
        patch: torch.Tensor,
    ):

        self.activations = None

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            patch
        )

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM hook did not capture "
                "activations."
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

            target_score = (
                spleen_probability[
                    spleen_mask
                ].mean()
            )

        else:

            target_score = (
                spleen_probability.mean()
            )

        print(
            "Grad-CAM target score:",
            float(
                target_score.detach().cpu()
            ),
        )

        gradients = torch.autograd.grad(
            outputs=target_score,
            inputs=self.activations,
            retain_graph=False,
            create_graph=False,
        )[0]

        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=patch.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        heatmap = cam[0, 0]

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()

        heatmap = (
            heatmap - heatmap_min
        ) / (
            heatmap_max
            - heatmap_min
            + 1e-8
        )

        return (
            heatmap.detach().cpu().numpy(),
            float(
                target_score.detach().cpu()
            ),
        )

    # --------------------------------------------------
    # Generate complete heatmap
    # --------------------------------------------------

    def generate(
        self,
        image_path: str | Path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Input image not found: "
                f"{image_path}"
            )

        print(
            "Preprocessing input..."
        )

        image, preprocessing_affine = (
            self._preprocess(
                image_path
            )
        )

        print(
            "Preprocessed volume shape:",
            tuple(image.shape),
        )

        print(
            "Preprocessing affine:"
        )

        print(
            preprocessing_affine
        )

        print(
            "Running normal sliding-window inference..."
        )

        prediction_logits = (
            self._get_prediction(
                image
            )
        )

        print(
            "Inference complete."
        )

        print(
            "Prediction shape:",
            tuple(
                prediction_logits.shape
            ),
        )

        center = (
            self._find_spleen_center(
                prediction_logits
            )
        )

        patch, metadata = (
            self._extract_patch(
                image,
                center,
            )
        )

        print(
            "Grad-CAM patch shape:",
            tuple(patch.shape),
        )

        print(
            "Patch start:",
            metadata["start"],
        )

        heatmap_patch, target_score = (
            self._generate_patch_cam(
                patch
            )
        )

        # --------------------------------------------------
        # Create full-volume heatmap
        # --------------------------------------------------

        full_shape = tuple(
            int(value)
            for value in image.shape[2:]
        )

        full_heatmap = np.zeros(
            full_shape,
            dtype=np.float32,
        )

        start_d, start_h, start_w = (
            metadata["start"]
        )

        roi_d, roi_h, roi_w = (
            self.roi_size
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
            + (
                dst_end_d
                - dst_start_d
            )
        )

        src_end_h = (
            src_start_h
            + (
                dst_end_h
                - dst_start_h
            )
        )

        src_end_w = (
            src_start_w
            + (
                dst_end_w
                - dst_start_w
            )
        )

        full_heatmap[
            dst_start_d:dst_end_d,
            dst_start_h:dst_end_h,
            dst_start_w:dst_end_w,
        ] = heatmap_patch[
            src_start_d:src_end_d,
            src_start_h:src_end_h,
            src_start_w:src_end_w,
        ]

        max_value = full_heatmap.max()

        if max_value > 0:

            full_heatmap /= max_value

        prediction = torch.argmax(
            prediction_logits,
            dim=1,
        )[0].detach().cpu().numpy()

        processed_input = (
            image[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        return {
            "heatmap": full_heatmap,

            "prediction": prediction.astype(
                np.uint8
            ),

            "input": processed_input.astype(
                np.float32
            ),

            "target_score": target_score,

            "center": center,

            "patch_size": self.roi_size,

            "preprocessed_affine": (
                preprocessing_affine
            ),
        }

    # --------------------------------------------------
    # Save heatmap with correct affine
    # --------------------------------------------------

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        output_path: str | Path,
        affine=None,
    ):

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
            f"Heatmap saved to: "
            f"{output_path}"
        )

    # --------------------------------------------------
    # Save PNG slices
    # --------------------------------------------------

    def save_slice_heatmaps(
        self,
        heatmap: np.ndarray,
        output_directory: str | Path,
    ):

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
            "Grad-CAM slice images..."
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
                f"Grad-CAM - Slice {slice_index}"
            )

            plt.axis("off")

            plt.savefig(
                output_path,
                dpi=120,
                bbox_inches="tight",
            )

            plt.close()

        print(
            f"Grad-CAM images saved to: "
            f"{output_directory}"
        )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def close(self):

        if self.hook is not None:

            self.hook.remove()

            self.hook = None

    def __del__(self):

        try:

            self.close()

        except Exception:

            pass