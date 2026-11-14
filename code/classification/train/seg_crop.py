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

def resample_torch_simple(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        mode='linear'
):
    if mode == 'linear':
        if data.ndim == 4:
            torch_mode = 'trilinear'
        elif data.ndim == 3:
            torch_mode = 'bilinear'
        else:
            raise RuntimeError
    else:
        torch_mode = mode

    if isinstance(new_shape, np.ndarray):
        new_shape = [int(i) for i in new_shape]

    if all([i == j for i, j in zip(new_shape, data.shape[1:])]):
        return data
    else:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
        new_shape = tuple(new_shape)
        with torch.no_grad():

            input_was_numpy = isinstance(data, np.ndarray)
            if input_was_numpy:
                data = torch.from_numpy(data).to(device)
            else:
                orig_device = deepcopy(data.device)
                data = data.to(device)

            if is_seg:
                unique_values = torch.unique(data)
                result_dtype = torch.int8 if max(unique_values) < 127 else torch.int16
                result = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
                if not memefficient_seg_resampling:
                    # believe it or not, the implementation below is 3x as fast (at least on Liver CT and on CPU)
                    # Why? Because argmax is slow. The implementation below immediately sets most locations and only lets the
                    # uncertain ones be determined by argmax

                    # unique_values = torch.unique(data)
                    # result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16)
                    # for i, u in enumerate(unique_values):
                    #     result[i] = F.interpolate((data[None] == u).float() * 1000, new_shape, mode='trilinear', antialias=False)[0]
                    # result = unique_values[result.argmax(0)]

                    result_tmp = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16,
                                             device=device)
                    scale_factor = 1000
                    done_mask = torch.zeros_like(result, dtype=torch.bool, device=device)
                    for i, u in enumerate(unique_values):
                        result_tmp[i] = \
                            F.interpolate((data[None] == u).float() * scale_factor, new_shape, mode=torch_mode,
                                          antialias=False)[0]
                        mask = result_tmp[i] > (0.7 * scale_factor)
                        result[mask] = u.item()
                        done_mask |= mask
                    if not torch.all(done_mask):
                        # print('resolving argmax', torch.sum(~done_mask), "voxels to go")
                        result[~done_mask] = unique_values[result_tmp[:, ~done_mask].argmax(0)].to(result_dtype)
                else:
                    for i, u in enumerate(unique_values):
                        if u == 0:
                            pass
                        result[F.interpolate((data[None] == u).float(), new_shape, mode=torch_mode, antialias=False)[
                                   0] > 0.5] = u
            else:
                result = F.interpolate(data[None].float(), new_shape, mode=torch_mode, antialias=False)[0]
            if input_was_numpy:
                result = result.cpu().numpy()
            else:
                result = result.to(orig_device)
        torch.set_num_threads(n_threads)
        return result
    

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
def zscore(img):
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
output_folder = "/path/to/fangyunhan/VLM/SSL3D_classification/seg_output"
output_dir = Path("/path/to/fangyunhan/VLM/SSL3D_classification/seg_output/crop")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# input_path_list = os.listdir(input_folder)
input_path_list = sorted([str(p.relative_to(input_folder)) for p in input_folder.glob("**/*")if p.is_file()])
print(f"Found {len(input_path_list)} input files.,{input_path_list[0:2]}")

for input_path in tqdm(input_path_list[0:10]):
    # print(f"input:{input_path}")
    input_path_full = join(input_folder, input_path)
    # print(f"input_path_full:{input_path_full}")

    # Set output path
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # 提取文件名无扩展名，例如 "valid_1_a_1"
    output_filename = f"{base_name}_mask.nii.gz"  # 生成 "valid_1_a_1_mask.nii.gz"
    output_path = join(output_folder, output_filename)   #os.path.splitext(os.path.splitext(input_path)[0])[0].replace("_0000", "") + '.nii.gz'  # Remove double extensions if present

    # if os.path.exists(output_path):
    #     continue
    
    # predict a single numpy array
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
    
    for region_name, labels in seg_label.items():
        # print(f"Processing region: {region_name}")
        mask = np.zeros_like(pred_array, dtype=bool)
        if region_name == 'material':
            #直接保存原图
            scaled_roi_array = img
            roi_data = np.expand_dims(scaled_roi_array, axis=0).astype(np.float16)
            scaled_roi_array = resample_torch_simple(roi_data, (128, 256, 256), is_seg=False)
            roi_data = zscore(scaled_roi_array)
            roi_data = roi_data.astype(np.float16) # 原图（128，256，256），dtype=np.float16
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

            # # Extract the scaled roi-array
            
            scaled_roi_array = img[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1]
            # print("scaled_roi_array.shape: ", scaled_roi_array.shape)

            # #######保存裁剪的图片及mask
            # mask_roi_array = mask_save[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1]
            
                            # # 预处理：添加通道维度
            print(f"scaled_roi_array.shape before transform: {scaled_roi_array.shape}")
            roi_data = np.expand_dims(scaled_roi_array, axis=0).astype(np.float16)
            roi_data = transform(roi_data)
            # roi_data = resample_torch_simple(roi_data, (128, 256, 256), is_seg=False)
            # roi_data = zscore(roi_data)
            roi_data = np.squeeze(roi_data, axis=0)  # 去掉通道维度
            print("roi_data.shape: ", roi_data.shape)

            # mask_roi_data = np.expand_dims(mask_roi_array, axis=0).astype(np.float16)
            # # mask_roi_data = transform_mask(mask_roi_data)
            # mask_roi_data = resample_torch_simple(mask_roi_data, (128, 256, 256), is_seg=True)
            # mask_roi_data = np.squeeze(mask_roi_data, axis=0)  # 去掉通道维度
            # print("mask_roi_data.shape: ", mask_roi_data.shape)

            # # roi_image = sitk.GetImageFromArray(scaled_roi_array)
            # # origin = sitk.ReadImage(img_dir / img_file)
            # sitk.WriteImage(sitk.GetImageFromArray(roi_data), output_dir / f"{Path(input_path).stem}_{region_name}.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray(mask_roi_data), output_dir / f"{Path(input_path).stem}_{region_name}_mask.nii.gz")
            #######保存裁剪的图片及mask

    #         '''# 预处理：添加通道维度
    #         roi_data = np.expand_dims(scaled_roi_array, axis=0).astype(np.float16)
    #         roi_data = transform(roi_data)

    #         mask_roi_data = np.expand_dims(mask_roi_array, axis=0).astype(np.float16)
    #         mask_roi_data = transform_mask(mask_roi_data)
    #         #拼接通道
    #         roi_data = np.concatenate((roi_data, mask_roi_data), axis=0)
    #         # print("roi_data.shape after concat: ", roi_data.shape) 
    #         # print(roi_data[1, :, :, :].min(), roi_data[1, :, :, :].max(), np.unique(roi_data[1, :, :, :]))
    #         roi_data = roi_data.astype(np.float16)

    #         # print('6',datetime.now())
            
    #         # 保存为 .npy 文件
    #         base_name = Path(img_file).name.replace(".nii.gz", "")
    #         output_file = output_dir / f"{base_name}_{region_name}.npy"
    #         np.save(output_file, roi_data)
    #         # print('7',datetime.now())
    #     else: # 如果没有对应的标签，创建一个空的 ROI 数据
    #         zero_data = np.zeros((2, 128, 256, 256), dtype=np.float16)# np.zeros_like(img_data, dtype=np.float16)
    #         # 创建(2, 128, 256, 256)，dtype=np.float16的全空数组
            
    #         # 保存为 .npy 文件
    #         base_name = Path(img_file).name.replace(".nii.gz", "")
    #         output_file = output_dir / f"{base_name}_{region_name}.npy"
    #         np.save(output_file, zero_data)'''
    
    # # Load the original image to get header information
    # original_image = sitk.ReadImage(input_path_full)
    
    # # Create new SimpleITK image from numpy array
    # pred_image = sitk.GetImageFromArray(pred_array)
    
    # # Copy metadata from original image
    # pred_image.CopyInformation(original_image)
    
    # # Save the prediction
    # sitk.WriteImage(pred_image, output_path)
