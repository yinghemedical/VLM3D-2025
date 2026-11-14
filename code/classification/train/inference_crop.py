import os
from pathlib import Path
from uuid import uuid4

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from parsing_utils import make_omegaconf_resolvers
import numpy as np

def extract_label_name(filename):
    # 从文件名中提取标签名称，例如 'valid_1005_a_1_arterial.nii.gz' -> 'arterial'
    return filename.split('_')[4].replace(".nii.gz", "") 

def fuse_predictions(y, idd, label_masks):
    # 初始化融合结果为 18 维零向量
    fused_result = np.zeros(18, dtype=np.float32)
    
    # 遍历每个预测向量和对应的文件名
    for i, filename in enumerate(idd):
        # 提取标签名称
        label_name = extract_label_name(filename)
        
        # 获取对应的掩码，如果标签不在 label_masks 中，使用零向量
        mask = np.array(label_masks.get(label_name)) # , np.zeros(18)
        
        # 将预测向量 y[i] 与掩码相乘，并与融合结果进行逻辑或操作
        masked_y = y[i] * mask  # 只保留掩码为 1 的位置
        fused_result = np.maximum(fused_result, masked_y)  # 逐位取最大值（逻辑或）
    
    return fused_result

@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer")
def inference(cfg):
    # delete automatically created hydra logger
    try:
        Path(
            "./main.log"
        ).unlink()
    except:
        pass

    # check if a fold was given, if yes scan only the fold dir for the checkpoint path
    # if cfg.fold:
    #     ckp_paths = list(Path(Path(cfg.exp_dir) / str(cfg.fold)).glob("*.ckpt"))
    # else:
    #     ckp_paths = list(Path(cfg.exp_dir).glob("*/*.ckpt"))
    ckp_paths = ['/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop/0/epoch=209-crop_mask-val_F1=0.00.ckpt']
    # ['/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop_mask/0/epoch=272-crop_mask-val_F1=0.00.ckpt']
    # ['/path/to/fangyunhan/VLM/SSL3D_classification/run/exp/catrate/checkpoints/output_crop/0/epoch=11-crop-256-val_F1=0.00.ckpt']
    # ['/path/to/fangyunhan/VLM/SSL3D_classification/run/exp/catrate/checkpoints/output/0/epoch=155-480-val_F1=0.00.ckpt']
    #  ['/path/to/fangyunhan/VLM/SSL3D_classification/run/exp/catrate/checkpoints/output/0/epoch=260-val_loss=0.00.ckpt']
    # ,'/path/to/fangyunhan/VLM/SSL3D_classification/run/exp/catrate/checkpoints/output/0/epoch=152-val_loss=0.00.ckpt']
    # ['/path/to/fangyunhan/VLM/SSL3D_classification/run/logger/best_f10.0794_epoch2.pth']
    
    logits = []
    for ckp_path in ckp_paths:

        # load the config that was used during training
        used_training_cfg = OmegaConf.load(os.path.join(cfg.exp_dir, "config.yaml"))
        used_training_cfg.trainer.pop("logger")
        used_training_cfg.trainer.pop("callbacks")
        used_training_cfg.model.metrics = cfg.metrics  # overwrite metrics

        # instantiate the model using this config
        model = instantiate(used_training_cfg.model)
        # load the weights
        model.load_state_dict(torch.load(ckp_path)["state_dict"])
        model.eval()

        # instantiate the dataset from the config if not some other dataset is specified in the infer.yaml
        dataset = instantiate(used_training_cfg.data).module
        # instantiate the trainer and also pass the new metrics
        trainer = instantiate(used_training_cfg.trainer)
       
        # run trainer.predict
        y, y_hat = zip(*trainer.predict(model, dataset))
        y = torch.cat(y)
        logits.append(torch.cat(y_hat))

    # if len(logits)>1:
    #     # for an ensemble we sum the logits first
    #     final_pred = torch.argmax(torch.sum(torch.stack(logits), 0), 1)
    # else:
    #     final_pred = torch.argmax(logits[0], 1)
    logits_tensor = torch.cat(logits, dim=0)  # 合并成一个 tensor
    final_pred = torch.sigmoid(logits_tensor)

    # final_pred = torch.sigmoid(logits)
    
    if cfg.pred_dir is not None:
        os.makedirs(cfg.pred_dir, exist_ok=True)
        #TODO

    # calculate metrics
    y = y.long()
    ##融合y 和final_pred
    # print('y:',y)
    # print('final_pred:',final_pred)


    print(model.val_metrics(final_pred, y))

if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
