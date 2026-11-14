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
import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import label
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from pathlib import Path
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, EnsureType
from monai import transforms
from typing import Union, Tuple, List
from copy import deepcopy
from torch.nn import functional as F


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
    
nnUNet_results = "/path/to/fangyunhan/workspace/ckpts/nnUNet_results"

transform = Compose([
        Resize(spatial_size=(128,256,256), mode="trilinear"),
        ZscoreNormWithOptionClip(clip=False),
    ])
transform_mask = Compose([
        Resize(spatial_size=(128,256,256), mode="nearest"), ])
    
fullres_predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring= False, #True, #False
    perform_everything_on_device=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)
# initializes the network architecture, loads the checkpoint
fullres_predictor.initialize_from_trained_model_folder(
    join(nnUNet_results, 'Dataset665_Totalsegmentator_v201_Anatomy/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres'),
    use_folds=(0, ),
    checkpoint_name='checkpoint_final.pth',
)

input_folder = Path("/path/to/Public_data/CT-RATE/dataset/valid_fixed")

input_path_list = sorted([str(p.relative_to(input_folder)) for p in input_folder.glob("**/*")if p.is_file()])
print(f"Found {len(input_path_list)} input files.")


class ctrate_Data(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()

        # self.img_dir = Path("/path/to/fangyunhan/data/ctrate/valid_crop_with_mask")
        self.img_dir = Path("/path/to/Public_data/CT-RATE/dataset/valid_fixed")
        # self.img_dir = Path("/path/to/fangyunhan/VLM/SSL3D_classification/data/valid_crop")
        label_file  = "/path/to/Public_data/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
        # label_file  = "/path/to/fangyunhan/VLM/SSL3D_classification/data/label/valid_crop_labels_all.csv"

        self.img_files = sorted([str(p.relative_to(input_folder)) for p in input_folder.glob("**/*")if p.is_file()])[0:20]

        labels_df = pd.read_csv(label_file)

        self.labels = {row['VolumeName']: np.array(row.drop('VolumeName').values, dtype=np.float32)
                       for _, row in labels_df.iterrows()}

        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_dir /self.img_files[idx]
        output_path = "/path/to/fangyunhan/VLM/SSL3D_classification/crop_infer/mask2"
        mask_path = os.path.join(output_path, os.path.basename(self.img_files[idx]))

        import time
        time_s = time.time()

        img_nii = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img_nii)
        # img, props = SimpleITKIO().read_images([img_path])
        # pred_array = fullres_predictor.predict_single_npy_array(img, props, None, None, False)
        # pred_array = pred_array.astype(np.int16)
        nii_seg = sitk.ReadImage(mask_path)
        pred_array = sitk.GetArrayFromImage(nii_seg)
        print("nnunet, ",  time.time() - time_s)
        print("post start")
        time_s = time.time()

        seg_label = {
            'material': [],  # 医用材料（假设为空，需确认具体标签）
            'arterial': [5,6],  # 2 # 动脉壁钙化
            'heart': [1],  # 3,4,5# 心肌肥大，心包积液，冠状动脉壁钙化
            'lung': [2],  # 6,7,8,9,10,11,12,14,16 # 肺相关疾病
            'trachea': [3],  # 13,15 # 支气管周围增厚，支气管扩张
            'hiatal': [4],  # 17 # 食管裂孔疝
            'lymphadenopathy': [5,7],  # 18 # 淋巴结病
        }

        seg_data = torch.from_numpy(pred_array.copy()).cuda()
        img = torch.from_numpy(img).cuda() #np.squeeze(img, axis=0)
        pred_array = torch.from_numpy(pred_array).half().cuda()
        
        #处理每一张图片
        roi_data_list = []
        for region_name, labels in seg_label.items():
            mask = torch.zeros_like(pred_array, dtype=torch.bool)
            if region_name == 'material':
                #直接保存原图
                scaled_roi_array = img
                roi_data = scaled_roi_array[None].half()
                roi_data = transform(roi_data)
                # roi_data = np.squeeze(roi_data, axis=0)  
                roi_data = roi_data.half() # 原图（128，256，256），dtype=np.float16
                roi_data_list.append(roi_data.cpu())
                continue

            if len(labels) > 0:
                for label in labels:
                    mask |= (seg_data == label)

                min_size = 1000
                m = 1.2
                non_zero_indices = torch.nonzero(mask > 0) # 非0元素的索引
                if len(non_zero_indices) == 0:
                    print(f"Warning: No non-zero indices for region {region_name} in {self.img_files[idx]}, using zero array")
                    roi_data = torch.zeros((1, 128, 256, 256), dtype=torch.float16)  # 直接设置全 0 数组
                else:
                    x_min, y_min, z_min = non_zero_indices.min(axis=0)[0]
                    x_max, y_max, z_max = non_zero_indices.max(axis=0)[0]
                    # Calculate the center of the current bounding box
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    z_center = (z_min + z_max) / 2

                    # Calculate the new radius range after scaling by factor `m`
                    x_range = (x_max - x_min + 1) * m / 2
                    y_range = (y_max - y_min + 1) * m / 2
                    z_range = (z_max - z_min + 1) * m / 2
                    
                    # print("x_range, y_range, z_range: ", x_range, y_range, z_range)
                    
                    # Calculate the new bounding box while ensuring it doesn't exceed the mask_array boundaries
                    x_min_new = max(0, int(x_center - x_range))
                    x_max_new = min(mask.shape[0] - 1, int(x_center + x_range))
                    y_min_new = max(0, int(y_center - y_range))
                    y_max_new = min(mask.shape[1] - 1, int(y_center + y_range))
                    z_min_new = max(0, int(z_center - z_range))
                    z_max_new = min(mask.shape[2] - 1, int(z_center + z_range))
                    # print('4',datetime.now())

                    scaled_roi_array = img[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1]

                    roi_data = scaled_roi_array[None].half()
                    roi_data = transform(roi_data.cuda()).cpu()
                    # torch.cuda.empty_cache()
                roi_data_list.append(roi_data.cpu())

        stacked_roi_data = torch.stack(roi_data_list, axis=0)  # 形状: (num_regions, 128, 256, 256)
        label = self.labels[Path(self.img_files[idx]).name] #.replace("material", "medical_material")]  #.replace(".npy", ".nii.gz")]
        idd = Path(self.img_files[idx]).name #.replace(".npy", ".nii.gz")

        print("post", time.time() - time_s)

        return stacked_roi_data.numpy(), label, idd ##idd #

    def __len__(self):
        return len(self.img_files)

class ctrate_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(ctrate_DataModule, self).__init__(**params)

    def setup(self, stage: str):
        # print(f"Setup called with stage: {stage}, fold: {self.fold}, time={datetime.now()}")
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
