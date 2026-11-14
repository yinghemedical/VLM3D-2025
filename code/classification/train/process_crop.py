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

# ---------------------------------------------------------------------------
#  HARD‑WIRED PATHS & CONSTANTS
# ---------------------------------------------------------------------------
VOL_DIR          = Path("/path/to/fangyunhan/VLM/SSL3D_classification/data/valid")
# Path("/path/to/Public_data/CT-RATE/dataset/valid_fixed")
OUT_FILE         = Path("output/all")

# CLASSIFIER_PATH  = Path('/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop_lung/0/epoch=302-crop_lung-val_F1=0.00.ckpt')
# CLASSIFIER_PATH  = Path("/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop/0/epoch=392-crop_mask-val_F1=0.00.ckpt")
# CLASSIFIER_PATH = Path("/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop_resume/0/epoch=287-crop_resume-val_F1=0.00.ckpt")  #182
# CLASSIFIER_PATH = Path("/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop_cls_head/0/epoch=278-crop_cls_head-val_F1=0.00.ckpt")
# CLASSIFIER_PATH = Path("/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_2cls/0/epoch=272-crop_2cls-val_F1=0.00.ckpt")
CLASSIFIER_PATH = Path("/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/all-finetune/0/epoch=92-crop_cls_head-f-val_F1=0.00.ckpt")
CHECKPOINTS_DIR = Path("/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/bce-inhance-finetune/0/")
METRICS_LOG_FILE = OUT_FILE / "metrics_log.txt"

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

    # ckp_path = CLASSIFIER_PATH
    # checkpoint_files =  list(CHECKPOINTS_DIR.glob("*.ckpt"))

    checkpoint_files = sorted(
        CHECKPOINTS_DIR.glob("*.ckpt"),
        key=lambda x: int(re.search(r'epoch=(\d+)', x.name).group(1))
    )
    for f in checkpoint_files:
        print(f"Found checkpoint: {f}")
    print(len(checkpoint_files))

    checkpoint_files = [CLASSIFIER_PATH]
    with METRICS_LOG_FILE.open("a") as metrics_file:
        for ckp_path in checkpoint_files:
            print(f"Processing checkpoint: {ckp_path}")    
            logits = []
        ###
            used_training_cfg = OmegaConf.load(os.path.join(cfg.exp_dir, "config.yaml"))
            used_training_cfg.trainer.pop("logger")
            used_training_cfg.trainer.pop("callbacks")
            used_training_cfg.model.metrics = cfg.metrics  # overwrite metrics

            # used_training_cfg.data.module.data_root_dir = VOL_DIR
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
            y, y_hat,idd = zip(*trainer.predict(model, dataset))
            y = torch.cat(y)
            logits.append(torch.cat(y_hat))

            logits_tensor = torch.cat(logits, dim=0)  # 合并成一个 tensor
            final_pred = torch.sigmoid(logits_tensor)

            ###计算指标
            y = y.long()
            metrics = model.val_metrics(final_pred, y)
            print(metrics)
            metrics_file.write(f"Checkpoint: {ckp_path.name}\n")
            metrics_file.write(f"Metrics: {metrics}\n\n")
            metrics_file.flush()  # Ensure immediate write to disk
            ###

            # Create predictions list for JSON output
            flat_idd = [item for sublist in idd for item in sublist]  # Flatten the list of lists
            sampled_idd = flat_idd[::7]
            predictions = []

            print(f"final_pred.shape: {final_pred.shape}")
            print(f"idd: {len(flat_idd)}")
            for img_id, probs in zip(sampled_idd, final_pred):
                # Remove file extensions from image ID

                # print(f"img_id: {img_id}")
                img_name = str(img_id).replace(".mha","").replace(".nii.gz","").replace(".npy","")
                # Create probabilities dictionary mapping PATHOLOGIES to probabilities
                prob_dict = {pathology: float(prob) for pathology, prob in zip(PATHOLOGIES, probs)}
                predictions.append({
                    "input_image_name": img_name,
                    "scores": prob_dict
                })

            # Create result dictionary
            result = {
                "name": "Generated probabilities",
                "type": "Abnormality Classification",
                "version": {"major": 1, "minor": 0},
                "predictions": predictions
            }

            # Save to JSON file
            checkpoint_name = ckp_path.stem 
            output_file = OUT_FILE/ f"results_{checkpoint_name}.json" #Path(cfg.output_file)  # Assume output_file is specified in cfg
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
    inference()