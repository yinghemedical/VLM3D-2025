import json
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import torch
from torch.utils.data import Dataset
# from batchviewer import view_batch
from .base_datamodule import BaseDataModule
# from .blosc2io import Blosc2IO
import nibabel as nib
import pandas as pd

import numpy as np
from scipy.ndimage import zoom

from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, EnsureType
from monai import transforms
import monai.transforms as mtf

class ZscoreNormWithOptionClip(transforms.MapTransform):
    """
        Initialize clamped img with zscore.
    """
    def __init__(self, clip: bool, clip_min_value: int = None, clip_max_value: int = None) -> None:
        self.clip = clip  
        self.clip_min_value = clip_min_value
        self.clip_max_value = clip_max_value
    
    def __call__(self, data: dict[torch.Tensor]) -> dict:
        img = data
        if self.clip:
            img = torch.clamp(img, self.clip_min_value, self.clip_max_value)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (max(std, 1e-8))

        return img
    
class ctrate_Data(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        self.split = split
        if split == "train":
            # self.img_dir = Path("/path/to/Public_data/CT-RATE/dataset/train_fixed")
            self.img_dir = Path("/path/to/fangyunhan/VLM/SSL3D_classification/data/train_hr")
            label_file = "/path/to/Public_data/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv"
        else:
            # self.img_dir = Path("/path/to/Public_data/CT-RATE/dataset/valid_fixed")
            self.img_dir = Path("/path/to/fangyunhan/VLM/SSL3D_classification/data/valid_hr")
            label_file  = "/path/to/Public_data/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"


        self.img_files = sorted([str(p.relative_to(self.img_dir)) for p in self.img_dir.glob("*.npy")])

        labels_df = pd.read_csv(label_file)

        self.labels = {row['VolumeName']: np.array(row.drop('VolumeName').values, dtype=np.float32)
                       for _, row in labels_df.iterrows()}

        self.transform = transform


    def __getitem__(self, idx):
        img1 = np.load(self.img_dir /self.img_files[idx])

        img1 = img1.astype(np.float32)
        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img1[...])})["image"]
            img = img.permute(0, 3, 1, 2)  # Change to (C, D, H, W)
        else:
            img = img1[...]
        # print(f"imgshape:{img.shape}")
        if self.split == "val":
            t = mtf.Resize(spatial_size=[320, 320, 160], mode="trilinear")
            img = t(img1)
            img = img.permute(0, 3, 1, 2)  # Change to (C, D, H, W)
        label = self.labels[Path(self.img_files[idx]).name.replace(".npy", ".nii.gz")]
        return img, label

    def __len__(self):
        return len(self.img_files)

# from torch.utils.data import Subset
# class ctrate_DataModule(BaseDataModule):
#     def __init__(self, **params):
#         super(ctrate_DataModule, self).__init__(**params)

#     def setup(self, stage: str):
#         full_train_dataset = ctrate_Data(
#             self.data_path,
#             split="train",
#             transform=self.train_transforms,
#             fold=self.fold,
#         )
#         full_val_dataset = ctrate_Data(
#             self.data_path,
#             split="val",
#             transform=self.test_transforms,
#             fold=self.fold,
#         )

#         self.train_dataset = Subset(full_train_dataset, list(range(20)))  # 仅前20个
#         self.val_dataset = Subset(full_val_dataset, list(range(10)))      # 仅前10个
#         self.test_dataset = self.val_dataset


class ctrate_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(ctrate_DataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = ctrate_Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = ctrate_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )
        self.test_dataset = ctrate_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )
        print(f"Setup called with stage: {stage}")
        # if stage == "fit" or stage is None:
        #     self.train_dataset = ctrate_Data(root='/path/to/Public_data/CT-RATE/dataset', split='train', fold=self.fold)
        #     print(f"train_dataset initialized with size: {len(self.train_dataset)}")

# # dataset = ctrate_Data(root=None, split="valid", fold=0, transform=None)
# # for img, label in dataset:
# #     print(f"Image shape: {img.shape}, Label: {label}")
# #     break