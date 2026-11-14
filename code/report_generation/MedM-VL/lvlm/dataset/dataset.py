import copy
import json
import os.path as osp
from collections import defaultdict
from PIL import Image
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from lvlm.dataset.template import TEMPlATE_FACTORY


TARGET_SPACING   = (1.5, 0.75, 0.75)

def resize_vol_safe(vol: torch.Tensor, spacing: tuple[float, float, float]):
    z, x, y = spacing
    current_shape = vol.shape[2:]  

    spacing_fixed = []
    for s in spacing:
        if s <= 0:
            spacing_fixed.append(1.0)
        else:
            spacing_fixed.append(s)
    spacing = tuple(spacing_fixed)

    scaling_factors = [spacing[i] / TARGET_SPACING[i] for i in range(3)]
    new_shape = []
    for i in range(3):
        new_size = int(round(current_shape[i] * scaling_factors[i]))
        new_size = max(1, new_size)  
        new_shape.append(new_size)
    print(f"Resize: {current_shape} @ {spacing} → {new_shape} @ {TARGET_SPACING}")

    return F.interpolate(vol, size=new_shape, mode="trilinear", align_corners=False)

def resize_vol(vol: torch.Tensor, spacing: tuple[float, float, float]):
    z, x, y = spacing
    current_shape = vol.shape[2:]
    current_spacing = spacing
    scaling_factors = [
        current_spacing[i] / TARGET_SPACING[i] for i in range(len(current_spacing))
    ]
    new_shape = [
        int(current_shape[i] * scaling_factors[i]) for i in range(len(current_shape))
    ]
    return F.interpolate(vol, size=new_shape, mode="trilinear", align_corners=False)


class MultiModalDataset(Dataset):
    def __init__(self, model, data, data_arguments, mode):
        super(MultiModalDataset, self).__init__()
        self.data_arguments = data_arguments
        self.mode = mode
        self.data = data

        self.tokenizer = model.tokenizer
        self.template = TEMPlATE_FACTORY[data_arguments.conv_version]()

        if model.encoder_image is not None:
            self.preprocessor_image = model.encoder_image.processor
        else:
            self.preprocessor_image = None

        if model.encoder_image3d is not None:
            self.preprocessor_image3d = model.encoder_image3d.processor
        else:
            self.preprocessor_image3d = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_dict = self.template.encode(
            messages=copy.deepcopy(data_item["conversations"]),
            tokenizer=self.tokenizer,
            mode=self.mode,
        )

        if "image" in data_item:  
            data_dict["image"] = []
            for filename in data_item["image"]:
                image_path = osp.join(self.data_arguments.image_path, filename)
                image = Image.open(image_path).convert("RGB")
                image = self.preprocessor_image(image, mode=self.mode)
                data_dict["image"].append(image)
        else:
            data_dict["image"] = None

        if "image3d" in data_item:
            valid_image3d_list = [] 
            for filename in data_item["image3d"]:
                image3d_path = osp.join(self.data_arguments.image3d_path, filename)
                try:
                    if image3d_path.endswith(".npy"):
                        image3d = np.load(image3d_path)

                    elif image3d_path.endswith(".nii") or image3d_path.endswith(".nii.gz"):
                        nifti_img = nib.load(image3d_path)
                        image3d = nifti_img.get_fdata()
                        
                        
                        if len(image3d.shape) < 3 or any(s <= 0 for s in image3d.shape[:3]):
                            print(f"跳过无效 NIfTI: {image3d_path} | shape={image3d.shape}")
                            continue
                        
                        # 胸部CT的HU值范围应该是 -1000 到 1000
                        # -1000: 空气, 0: 水, 1000: 骨骼
                        # 肺组织通常在 -800 到 -200 之间，软组织在 -100 到 100 之间
                        HU_MIN, HU_MAX = -1000, 1000
                        image3d = np.clip(image3d, HU_MIN, HU_MAX)

                        # 获取spacing信息（在resize之前）
                        sx, sy, sz = nifti_img.header.get_zooms()[:3]
                        
                        # 转置：NIfTI默认是 (x, y, z)，需要转为 (z, x, y) 用于后续处理
                        image3d = np.transpose(image3d, (2, 0, 1))
                        
                        # 转换为torch tensor并添加batch和channel维度用于resize
                        image3d = torch.from_numpy(image3d.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                       
                        # 检查resize前的shape
                        if any(s <= 0 for s in image3d.shape[2:]):
                            continue
                            
                        # Resize到目标spacing (1.5, 0.75, 0.75)
                        # 在原始HU值范围内进行resize，保持数值精度
                        image3d = resize_vol_safe(image3d, (sz, sy, sx))  # [1, 1, D, H, W]
                        image3d = image3d.squeeze(0)  # [1, D, H, W]
                        
                        # 检查resize后的shape
                        if any(s <= 0 for s in image3d.shape[1:]):
                            continue
                            
                        # 归一化到 [0, 255] 范围（在resize之后进行，避免resize时的数值精度问题）
                        # 使用torch操作保持数据类型一致性
                        image3d = (image3d - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
                        image3d = image3d.numpy()  

                    elif image3d_path.endswith(".mha"):
                        image3d_sitk = sitk.ReadImage(image3d_path)
                        sx, sy, sz = image3d_sitk.GetSpacing()
                        image3d = sitk.GetArrayFromImage(image3d_sitk).astype("float32")

                        
                        if len(image3d.shape) < 3 or any(s <= 0 for s in image3d.shape[:3]):
                            continue

                        # 胸部CT的HU值范围应该是 -1000 到 1000
                        HU_MIN, HU_MAX = -1000, 1000
                        image3d = np.clip(image3d, HU_MIN, HU_MAX)
                        
                        # 转换为torch tensor并添加batch和channel维度用于resize
                        image3d = torch.from_numpy(image3d).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                        
                        # 检查resize前的shape
                        if any(s <= 0 for s in image3d.shape[2:]):
                            continue          
                        
                        # Resize到目标spacing (1.5, 0.75, 0.75)
                        # 在原始HU值范围内进行resize，保持数值精度
                        image3d = resize_vol_safe(image3d, (sz, sy, sx))  # [1, 1, D, H, W]
                        image3d = image3d.squeeze(0)  # [1, D, H, W]
                        
                        # 检查resize后的shape
                        if any(s <= 0 for s in image3d.shape[1:]):
                            continue
                        
                        # 归一化到 [0, 255] 范围（在resize之后进行，避免resize时的数值精度问题）
                        image3d = (image3d - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
                        image3d = image3d.numpy()


                    if len(image3d.shape) < 3 or any(s <= 0 for s in image3d.shape):
                        continue

                    if self.preprocessor_image3d is not None:
                        image3d = self.preprocessor_image3d(image3d, mode=self.mode)
                    elif self.preprocessor_image is not None:
                        image3d = self.preprocessor_image(image3d, mode=self.mode)
                    
                    valid_image3d_list.append(image3d)
                    
                except Exception as e:
                    print(f"处理失败，跳过: {image3d_path} ")
                    continue


            if valid_image3d_list:
                data_dict["image3d"] = valid_image3d_list
            else:
                print(f" {data_item.get('id', idx)}的image3d 无效，设为 None")
                data_dict["image3d"] = None
        else:
            data_dict["image3d"] = None

        return data_dict



class DataCollatorForMultiModalDataset:
    def __init__(self, tokenizer, mode):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, instances):
       
        valid_instances = []
        for inst in instances:
            if inst.get("image3d") is not None or inst.get("image") is not None:
                valid_instances.append(inst)
            else:
                print(f"无图像，跳过: {inst.get('id', 'unknown')}")
                continue
        
        if len(valid_instances) == 0:
            dummy = {
                "input_ids": torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[0]], dtype=torch.long),
                "image": None,
                "image3d": None,
            }
            if self.mode == "train":
                dummy["labels"] = torch.tensor([[-100]], dtype=torch.long)
            return dummy

        instances = valid_instances


        input_ids = [instance["input_ids"] for instance in instances]
        if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.mode == "train":
            labels = [instance["labels"] for instance in instances]
            if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                for label in labels:
                    label[label == self.tokenizer.eos_token_id] = -300

            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            labels = labels[:, : self.tokenizer.model_max_length]
            batch["labels"] = labels

        image_list = []
        for instance in instances:
            if instance["image"] is not None:
                image_list.extend(instance["image"])
        batch["image"] = torch.stack(image_list) if image_list else None

        image3d_list = []
        for instance in instances:
            if instance["image3d"] is not None:
                image3d_list.extend(instance["image3d"])
        batch["image3d"] = torch.stack(image3d_list) if image3d_list else None
        return batch

def create_data_module(model, data_arguments, mode):
    with open(data_arguments.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_dataset = MultiModalDataset(
        model=model,
        data=data,
        data_arguments=data_arguments,
        mode=mode,
    )
    data_collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode=mode)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def create_multi_data_module(model, data_arguments, ratio_dict, mode):
    with open(data_arguments.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    task_data_map = defaultdict(list)
    for item in all_data:
        task = item["task"]
        task_data_map[task].append(item)

    task_loaders = {"train":[]}
    for task_name, data_list in task_data_map.items():
        dataset = MultiModalDataset(
            model=model, 
            data=data_list, 
            data_arguments=data_arguments, 
            mode=mode,
        )
        dataset.sample_ratio = ratio_dict[task_name]
        dataset.task = task_name
        task_loaders["train"].append(dataset)
    collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode=mode)
    return task_loaders, collator