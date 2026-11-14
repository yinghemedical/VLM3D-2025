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
    ckp_paths = ['/path/to/fangyunhan/VLM/SSL3D_classification/run/exp/catrate/checkpoints/output/0/epoch=92-posweight-val_F1=0.00-v1.ckpt']
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
    print(model.val_metrics(final_pred, y))

if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
