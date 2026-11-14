#!/usr/bin/env python3
"""
ctclip_classifier_pipeline.py
=============================

Abnormality‑classification inference script for the competition docker.
The pipeline **shares exactly the same CT pre‑processing procedure** as
`ctchat_pipeline.py` so that both tasks consume identically prepared
volumes.

Fixed container layout
----------------------
```
/input                                ← CT volumes (*.mha / *.nii / *.nii.gz)
/output                               ← will receive results.json

/opt/app/models/clip_visual_encoder.pth            ← CTViT weights
/opt/app/models/BiomedVLP-CXR-BERT-specialized     ← Biomed‑BERT tokenizer + ckpt
/opt/app/models/ctclip_classifier.pth              ← ImageLatentsClassifier weights
```

Output schema (must match challenge spec)
----------------------------------------
```json
{
  "name": "Generated probabilities",
  "type": "Abnormality Classification",
  "version": {"major": 1, "minor": 0},
  "predictions": [
    {
      "input_image_name": "<volume>.nii.gz",
      "scores": {"Medical material": 0.00, "Arterial wall calcification": 0.42, …}
    }
  ]
}
```

The script runs on **CUDA** when available and falls back to CPU.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

from PIL import Image          # <-- put with your other imports
import time
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
import re
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
#  HARD‑WIRED PATHS & CONSTANTS
# ---------------------------------------------------------------------------
# VOL_DIR          = Path("/path/to/Public_data/CT-RATE/dataset/valid_fixed")
VOL_DIR = Path("/input")
OUT_FILE = Path("/output/results.json")

CLASSIFIER_PATH = Path("/opt/app/ckpts/epoch=263-crop_resume-val_F1=0.00.ckpt")
CLASSIFIER_PATH2 = Path("/opt/app/ckpts/epoch=371-crop_cls_head-val_F1=0.00.ckpt")
CLASSIFIER_PATH3 = Path("/opt/app/ckpts/epoch=92-crop_cls_head-f-val_F1=0.00.ckpt")

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
NUM_CLASSES = len(PATHOLOGIES)

# ---------------------------------------------------------------------------
#  INFERENCE
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer")
def inference(cfg):
    # delete automatically created hydra logger
    try:
        Path(
            "./main.log"
        ).unlink()
    except:
        pass

    print(f"Processing checkpoint: {CLASSIFIER_PATH}")    
### set model
    used_training_cfg = OmegaConf.load(os.path.join(cfg.exp_dir, "config.yaml"))
    model = instantiate(used_training_cfg.model)
    model.load_state_dict(torch.load(CLASSIFIER_PATH)["state_dict"])
    model.eval()

    used_training_cfg2 = OmegaConf.load(os.path.join(cfg.exp_dir2, "config.yaml"))
    model2 = instantiate(used_training_cfg2.model)
    model2.load_state_dict(torch.load(CLASSIFIER_PATH2)["state_dict"])
    model2.eval()

    # used_training_cfg3 = OmegaConf.load(os.path.join(cfg.exp_dir3, "config.yaml"))
    model3 = instantiate(used_training_cfg2.model)
    model3.load_state_dict(torch.load(CLASSIFIER_PATH3)["state_dict"])
    model3.eval()

### set dataset
    used_training_cfg.data.module.data_root_dir = VOL_DIR
    data_module = instantiate(used_training_cfg.data).module
    data_module.setup(stage="predict")

    test_dataset = data_module.test_dataset
    print(test_dataset)
    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=data_module.batch_size,  # 使用 DataModule 中的 batch_size
        shuffle=False,  # 推理时通常不需要 shuffle
        num_workers=data_module.num_workers,  # 使用 DataModule 中的 num_workers
        pin_memory=True,  # 优化 GPU 传输
        persistent_workers=False,  # 根据需求调整
    )

### infer
    device = torch.device("cuda:0")
    model = model.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)

    y_hat, idd = [], []
    y_hat2,y_hat3 = [], []
    for batch in tqdm(dataloader, desc="Processing batches"):
        x, id = batch  
        print(f"Batch  dtype: {x.dtype}")
        with torch.no_grad():
            with autocast():
                y_hat_batch = model.predict_step(batch, batch_idx=0)[0] 
                y_hat2_batch = model2.predict_step(batch, batch_idx=0)[0]
                y_hat3_batch = model3.predict_step(batch, batch_idx=0)[0]

                y_hat.append(y_hat_batch)
                y_hat2.append(y_hat2_batch)
                y_hat3.append(y_hat3_batch)

                idd.append(id)

        # logits.append((y_hat[0]+y_hat2[0])/2)
    y_hat_tensor = torch.cat(y_hat, dim=0)
    y_hat2_tensor = torch.cat(y_hat2, dim=0)
    y_hat3_tensor = torch.cat(y_hat3, dim=0)

    logits_tensor = (torch.sigmoid(y_hat_tensor) + torch.sigmoid(y_hat2_tensor)+torch.sigmoid(y_hat3_tensor)) / 3
    # logits_tensor = torch.cat(logits, dim=0)  
    # final_pred = torch.sigmoid(logits_tensor)
    final_pred = logits_tensor

    # Create predictions list for JSON output
    flat_idd = [item for sublist in idd for item in sublist]  # Flatten the list of lists
    predictions = []

    for img_id, probs in zip(flat_idd, final_pred):
        img_name = str(img_id).replace(".mha","").replace(".nii.gz","").replace(".npy","")
        # Create probabilities dictionary mapping PATHOLOGIES to probabilities
        prob_dict = {pathology: float(prob) for pathology, prob in zip(PATHOLOGIES, probs)}
        predictions.append({
            "input_image_name": img_name,
            "probabilities": prob_dict
        })

    # Create result dictionary
    result = {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"major": 1, "minor": 0},
        "predictions": predictions
    }

    # Save to JSON file
    output_file = OUT_FILE 
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"Saved → {output_file}")
    ###

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_omegaconf_resolvers()
    time_s = time.time()
    inference()
    print("infer", time.time() - time_s)