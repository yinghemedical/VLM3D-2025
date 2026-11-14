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
import SimpleITK as sitk

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
## val_dataset
        # self.img_dir = Path("/path/to/Public_data/CT-RATE/dataset/valid_fixed")
        self.img_dir = Path(root)
        #Path("/path/to/fangyunhan/VLM/SSL3D_classification/data/valid")
        # label_file  = "/path/to/Public_data/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
        # print(self.img_dir)

        self.img_files = sorted([str(p.relative_to(self.img_dir)) for p in self.img_dir.glob("**/*")if p.is_file()])

        # labels_df = pd.read_csv(label_file)

        # self.labels = {row['VolumeName']: np.array(row.drop('VolumeName').values, dtype=np.float32)
                    #    for _, row in labels_df.iterrows()}

        self.transform = transform
        self.image_transforms = Compose([
            Resize(spatial_size=(256,256,128), mode="trilinear"),
            ZscoreNormWithOptionClip(
                clip=False
            ),
        ])

    def __getitem__(self, idx):
        img_path = str(self.img_dir / self.img_files[idx])
        # print(img_path)
        if img_path.endswith('.npy'):
            img1 = np.load(img_path)
            img = img1[...] #self.transform(**{"image": torch.from_numpy(img1[...])})["image"]
        elif img_path.endswith('.nii.gz')or img_path.endswith(".nii"):
            nii_img = nib.load(img_path)
            img1 = nii_img.get_fdata()
            img1 = np.expand_dims(img1, axis=0).astype(np.float32)
            img1 = self.image_transforms(img1)
            img1 = img1.astype(np.float16)
            img = img1[...] #self.transform(**{"image": img1[...]})["image"]
        elif img_path.endswith('.mha'):
            itk = sitk.ReadImage(str(img_path))
            img = sitk.GetArrayFromImage(itk).astype("float32")
            img = np.transpose(img, (2, 1, 0))  # z, y, x
            img = np.expand_dims(img, axis=0)
            img = self.image_transforms(img)
            img = img.astype(np.float16)
            img = img[...] #self.transform(**{"image": img[...].astype(np.float32)})
        # print(f"img shape: {img.shape}, img type: {type(img)}")
        # label = self.labels[Path(self.img_files[idx]).name.replace(".npy", ".nii.gz")]
        idd = Path(self.img_files[idx]).name.replace(".npy", ".nii.gz")

        return img, idd

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