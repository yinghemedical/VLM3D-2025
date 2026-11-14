from PIL import Image

import monai.transforms as mtf
import numpy as np
import torch
from transformers import AutoConfig, PretrainedConfig

from dinov2.models.vision_transformer import vit_large
from lvlm.model.encoder.base import EncoderModel


def dinov2_load_config(model_arguments):
    encoder_image_config = DINOv2Config()
    encoder_image_config.img_size = [
        32,
        encoder_image_config.image_size,
        encoder_image_config.image_size,
    ]
    return encoder_image_config


class ImageProcessor:
    def __init__(self, config):
        self.transform_3d_train = mtf.Compose(
            [
                mtf.CropForeground(),
                mtf.Resize(spatial_size=[32, config.encoder_image_config.image_size, config.encoder_image_config.image_size], mode="bilinear"),
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                # mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                # mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        self.transform_3d_val = mtf.Compose(
            [
                mtf.CropForeground(),
                mtf.Resize(spatial_size=[32, config.encoder_image_config.image_size, config.encoder_image_config.image_size], mode="bilinear"),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

    def __call__(self, image, mode):
        if isinstance(image, Image.Image):  # process 2D image
            image = self.processor(image, return_tensors="pt")
            image = image["pixel_values"][0]  # [3, H, W]
        elif isinstance(image, np.ndarray):  # process 3D image: [D, H, W], [0, 255]
            # image = image[np.newaxis, ...]
            if mode == "train":
                image = self.transform_3d_train(image)
            else:
                image = self.transform_3d_val(image)  # [1, D, H, W], [0, 255]
        return image


class DINOv2Model(EncoderModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config.encoder_image_config
        self.processor = ImageProcessor(config)

        state_dict = torch.load(
            f=config.encoder_image_name_or_path,
            map_location='cpu',
        )
        s_or_t = next(iter(state_dict))
        keys_to_pop = []
        keys_to_rename = []
        for key in list(state_dict[s_or_t].keys()):
            if "dino_head" in key:
                keys_to_pop.append(key)
            if "ibot_head" in key:
                keys_to_pop.append(key)
            if "backbone." in key:
                keys_to_rename.append((key, key.replace("backbone.", "")))

        for key in keys_to_pop:
            state_dict[s_or_t].pop(key)
        for old_key, new_key in keys_to_rename:
            state_dict[s_or_t][new_key] = state_dict[s_or_t][old_key]
            state_dict[s_or_t].pop(old_key)

        fix_state_dict = state_dict[s_or_t]

        self.encoder = vit_large(img_size=(512, 512), init_values=1)
        self.encoder.load_state_dict(state_dict=fix_state_dict, strict=True)
        self.encoder.requires_grad_(False)

    def forward(self, x, select_layer, select_feature):
        features = self.encoder.get_intermediate_layers(x, n=5)
        features = features[select_layer]

        if select_feature == "patch":
            features = features[:, 1:]
        elif select_feature == "cls_patch":
            features = features

        return features


class DINOv2Config(PretrainedConfig):
    model_type = "dinov2-custom"
    def __init__(
        self,
        args=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = 1152
        self.image_size = 512


if __name__ == "__main__":
    # ckpt = torch.load("/data/jiawei_workspace/weights/dinov2/tmp_models/DINOv2_vitl16-1152_projectx150w-512_ckpt185k.pth")
    state_dict = torch.load(
        f="/data/jiawei_workspace/weights/dinov2/tmp_models/DINOv2_vitl16-1152_projectx150w-512_ckpt185k.pth",
        map_location='cpu',
    )
    s_or_t = next(iter(state_dict))
    keys_to_pop = []
    keys_to_rename = []
    for key in list(state_dict[s_or_t].keys()):
        if "dino_head" in key:
            keys_to_pop.append(key)
        if "ibot_head" in key:
            keys_to_pop.append(key)
        if "backbone." in key:
            keys_to_rename.append((key, key.replace("backbone.", "")))

    for key in keys_to_pop:
        state_dict[s_or_t].pop(key)
    for old_key, new_key in keys_to_rename:
        state_dict[s_or_t][new_key] = state_dict[s_or_t][old_key]
        state_dict[s_or_t].pop(old_key)
    import pdb; pdb.set_trace()
    fix_state_dict = state_dict[s_or_t]
    # encoder.load_state_dict(state_dict=fix_state_dict, strict=True)

    # encoder_cls = globals().get("vit_large")
    # encoder = encoder_cls(img_size=(490, 490), init_values=1)
    import sys
    sys.path.append("/data/jiawei_workspace/report_generate_docker/DINOv2")
    from dinov2.models.vision_transformer import vit_large
    encoder = vit_large(img_size=(512, 512), init_values=1)
    encoder.load_state_dict(state_dict=fix_state_dict, strict=True)

    image3d = np.load("/path/to/jiawei_database/ctrate_raw/valid/valid_1/valid_1a/valid_1_a_1.npy")
    transform_3d_val = mtf.Compose(
        [
            mtf.CropForeground(),
            mtf.Resize(spatial_size=[32, 512, 512], mode="bilinear"),
            mtf.ToTensor(dtype=torch.float),
        ]
    )
    image3d = transform_3d_val(image3d)
    image3d = image3d.permute(1, 0, 2, 3)  # [D, 1, H, W]
    features = encoder.get_intermediate_layers(image3d, n=5)
    features = features[-1]  # [B, 1024, D]
    print()
    print(1)