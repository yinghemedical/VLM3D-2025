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
            # 定义训练和验证的图像目录
            self.train_img_dir = Path("/path/to/fangyunhan/data/ctrate/train_crop_with_mask")
            self.valid_img_dir = Path("/path/to/fangyunhan/data/ctrate/valid_crop_with_mask")
            # 合并两个目录下的 .npy 文件
            self.img_files = sorted(
                [str(p.relative_to(self.train_img_dir)) for p in self.train_img_dir.glob("*.npy")] +
                [str(p.relative_to(self.valid_img_dir)) for p in self.valid_img_dir.glob("*.npy")]
            )
            # 读取训练和验证的标签文件
            train_label_file = "/path/to/fangyunhan/VLM/SSL3D_classification/data/label/train_crop_labels.csv"
            valid_label_file = "/path/to/fangyunhan/VLM/SSL3D_classification/data/label/valid_crop_labels_all.csv"
            train_labels_df = pd.read_csv(train_label_file)
            valid_labels_df = pd.read_csv(valid_label_file)
            # 合并标签数据
            labels_df = pd.concat([train_labels_df, valid_labels_df], ignore_index=True)
        else:
            self.img_dir = Path("/path/to/fangyunhan/data/ctrate/valid_crop_with_mask")
            label_file = "/path/to/fangyunhan/VLM/SSL3D_classification/data/label/valid_crop_labels_all.csv"
            self.img_files = sorted([str(p.relative_to(self.img_dir)) for p in self.img_dir.glob("*.npy")])
            labels_df = pd.read_csv(label_file)

        # 创建文件名到标签的映射
        self.labels = {row['FileName']: np.array(row.drop('FileName').values, dtype=np.float32)
                       for _, row in labels_df.iterrows()}

        self.transform = transform
        self.image_transforms = Compose([
            Resize(spatial_size=(256,256,128), mode="trilinear"),
            ZscoreNormWithOptionClip(
                clip=False
            ),
        ])

    def __getitem__(self, idx):
        # print("len()",len(self.img_files))
        if self.split == "train":
            if "train" in self.img_files[idx]:
                img1 = np.load(self.train_img_dir /self.img_files[idx])
            else:
                img1 = np.load(self.valid_img_dir /self.img_files[idx])
        else:
            img1 = np.load(self.img_dir /self.img_files[idx])
        img1 = img1[0,:,:,:].astype(np.float32)
        img1 = np.expand_dims(img1, axis=0)  # 添加通道维度

        if self.transform:
            # img = self.transform(**{"image": img1[...]})["image"]
            img = self.transform(**{"image": torch.from_numpy(img1[...])})["image"]
        else:
            img = img1[...]#torch.from_numpy(img1[...])

        label = self.labels[Path(self.img_files[idx]).name.replace("material", "medical_material")]  #.replace(".npy", ".nii.gz")]
        idd = Path(self.img_files[idx]).name.replace(".npy", ".nii.gz")
        # view_batch(img, width=600, height=600)
        # return label_mask(与structure对应，structue从文件名中解析), img, idd
        # structure与label_mask的映射表
        label_masks = {
            "material": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # medical
            "arterial": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "heart":    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "lung":     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            "trachea":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            "hiatal":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "lymphadenopathy": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        structure = Path(self.img_files[idx]).name.split("_")[4].replace(".npy", "")  # 从文件名中解析结构

        s_dict = {
            "material": [1, 0, 0, 0, 0, 0, 0],
            "arterial": [0, 1, 0, 0, 0, 0, 0],
            "heart":    [0, 0, 1, 0, 0, 0, 0],
            "lung":     [0, 0, 0, 1, 0, 0, 0],
            "trachea":  [0, 0, 0, 0, 1, 0, 0], 
            "hiatal":   [0, 0, 0, 0, 0, 1, 0], 
            "lymphadenopathy": [0, 0, 0, 0, 0, 0, 1]
        }
        self.categories = [
        "material", "arterial", "heart", "lung", "trachea", "hiatal", "lymphadenopathy"
        ]
        s_label_idx = self.categories.index(structure)  # 转换为类别索引 (0-6)
        s_label = torch.tensor(s_label_idx, dtype=torch.long)
        # s_label = torch.tensor(s_dict[structure], dtype=torch.float)  # 默认值为全零
        if structure in label_masks:
            # label_mask = label_masks[structure]
            label_mask = label_masks[structure]
            if isinstance(label_mask, list):
                label_mask = torch.tensor(label_mask)  # 按批次维度拼接
        else:
            print(f"Warning: Structure '{structure}' not found in label_masks")
        # print(f"Processing {self.img_files[idx]}: structure={structure},label={label} label_s={s_label}")
        return img, label, label_mask ,s_label #, idd #, category_id#

    def __len__(self):
        return len(self.img_files)

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

# dataset = ctrate_Data(root=None, split="valid", fold=0, transform=None)
# for img, label in dataset:
#     print(f"Image shape: {img.shape}, Label: {label}")
#     break