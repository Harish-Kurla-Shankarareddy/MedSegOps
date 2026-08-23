from pathlib import Path

import torch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)


class SpleenSegmenter:
    """MONAI-based 3D spleen segmentation inference engine."""

    def __init__(
        self,
        model_path: str | Path = "models/monai/model.pt",
        device: str | None = None,
    ):
        self.model_path = Path(model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )

        state_dict = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessing = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(
                    keys="image",
                    pixdim=(1.5, 1.5, 2.0),
                    mode="bilinear",
                ),
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-57,
                    a_max=164,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                EnsureTyped(keys="image"),
            ]
        )

        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            overlap=0.5,
        )

    @torch.inference_mode()
    def predict_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Run segmentation on a preprocessed tensor."""

        image = image.to(self.device)

        prediction = self.inferer(
            inputs=image,
            network=self.model,
        )

        return prediction

    def predict(self, image_path: str | Path, output_dir: str | Path):
        """Run end-to-end segmentation and save the result."""

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        data = self.preprocessing(
            {"image": str(image_path)}
        )

        image = data["image"].unsqueeze(0).to(self.device)

        with torch.inference_mode():
            prediction = self.inferer(
                inputs=image,
                network=self.model,
            )

        data["pred"] = prediction[0].cpu()

        postprocessing = Compose(
            [
                Activationsd(
                    keys="pred",
                    softmax=True,
                ),
                Invertd(
                    keys="pred",
                    transform=self.preprocessing,
                    orig_keys="image",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(
                    keys="pred",
                    argmax=True,
                ),
                SaveImaged(
                    keys="pred",
                    output_dir=str(output_dir),
                    output_ext=".nii.gz",
                    output_dtype="uint8",
                    output_postfix="seg",
                    separate_folder=False,
                ),
            ]
        )

        result = postprocessing(data)

        return result