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
    verbose=True,
    verbose_preprocessing=True,
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
print(f"Found {len(input_path_list)} input files.,{input_path_list[0:2]}")

for input_path in tqdm(input_path_list[0:2]):

    input_path_full = join(input_folder, input_path)

    img, props = SimpleITKIO().read_images([input_path_full])
    pred_array = fullres_predictor.predict_single_npy_array(img, props, None, None, False)
    pred_array = pred_array.astype(np.int16)
    print(f"pred_array.shape: {pred_array.shape}, pred_array.dtype: {pred_array.dtype}")

    seg_label = {
        'material': [],  # 医用材料（假设为空，需确认具体标签）
        'arterial': [5,6],  # 2 # 动脉壁钙化
        'heart': [1],  # 3,4,5# 心肌肥大，心包积液，冠状动脉壁钙化
        'lung': [2],  # 6,7,8,9,10,11,12,14,16 # 肺相关疾病
        'trachea': [3],  # 13,15 # 支气管周围增厚，支气管扩张
        'hiatal': [4],  # 17 # 食管裂孔疝
        'lymphadenopathy': [5,7],  # 18 # 淋巴结病
    }

    seg_data = pred_array.copy()
    img = np.squeeze(img, axis=0)  # 去掉通道维度
    
    #处理每一张图片
    roi_data_list = []
    for region_name, labels in seg_label.items():
        # print(f"Processing region: {region_name}")
        mask = np.zeros_like(pred_array, dtype=bool)
        if region_name == 'material':
            #直接保存原图
            scaled_roi_array = img
            roi_data = np.expand_dims(scaled_roi_array, axis=0).astype(np.float16)
            roi_data = transform(roi_data)
            roi_data = np.squeeze(roi_data, axis=0)  # 去掉通
            roi_data = roi_data.astype(np.float16) # 原图（128，256，256），dtype=np.float16
            roi_data_list.append(roi_data)
            continue

        if len(labels) > 0:
            mask_save = np.zeros_like(seg_data, dtype=seg_data.dtype)
            for label in labels:
                mask |= (seg_data == label)
                mask_save = np.where(seg_data == label, seg_data, mask_save)
            # mask = np.isin(seg_data, labels)
            # print(f"Mask shape: {mask_save.shape}, unique values: {np.unique(mask_save)}")
            # print(f"Processing region: {region_name}, Labels: {labels}, Mask shape: {mask.shape}")

            min_size = 1000
            # Remove small objects
            # mask = remove_small_objects_binary(mask, min_size=min_size).astype(np.uint8)
            # print('2',datetime.now())
            m = 1.2
            non_zero_indices = np.argwhere(mask > 0) # 非0元素的索引
            # print('3',datetime.now())
            # Get the minimum and maximum values for x, y, z dimensions
            x_min, y_min, z_min = non_zero_indices.min(axis=0)
            x_max, y_max, z_max = non_zero_indices.max(axis=0)
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

            roi_data = np.expand_dims(scaled_roi_array, axis=0).astype(np.float16)
            roi_data = transform(roi_data)
            roi_data = np.squeeze(roi_data, axis=0)  # 去掉通道维度
            roi_data_list.append(roi_data)

        stacked_roi_data = np.stack(roi_data_list, axis=0)  # 形状: (num_regions, 128, 256, 256)
        print(f"Stacked ROI data shape: {stacked_roi_data.shape}")
            


