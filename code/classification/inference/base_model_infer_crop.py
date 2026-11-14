import math
import warnings

import lightning as L
import torch
import torch.nn as nn
import wandb
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.aggregation import CatMetric
from metrics.balanced_accuracy import BalancedAccuracy
from augmentation.mixup import mixup_criterion, mixup_data
from metrics.conf_mat import ConfusionMatrix
from regularization.sam import SAM
from torch.utils.data import DataLoader, RandomSampler,SubsetRandomSampler
import numpy as np

def extract_label_name(filename):
    # 从文件名中提取标签名称，例如 'valid_1005_a_1_arterial.nii.gz' -> 'arterial'
    return filename.split('_')[4].replace(".nii.gz", "") 

def extract_prefix(filename):
    # 提取前缀，比如 'valid_1005_a_1_arterial.nii.gz' -> 'valid_1005_a_1'
    return "_".join(filename.split('_')[:4])

def fuse_predictions(y, idd): #, label_masks):
    # 初始化融合结果为 18 维零向量
    label_masks = {
        "material": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "arterial": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "heart":    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "lung":     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        "trachea":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        "hiatal":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "lymphadenopathy": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
        # ===== 前缀一致性检查 =====
    # prefixes = [extract_prefix(fname) for fname in idd]
    # if len(set(prefixes)) > 1:
    #     print(f"WARNING: 文件名前缀不一致: {set(prefixes)}")
    fused_result = torch.zeros(18, dtype=torch.float32, device=y.device)
    
    # 遍历每个预测向量和对应的文件名
    for i, y_i in enumerate(y):
        mask_keys = list(label_masks.keys())
        label_name = mask_keys[i] if i < len(mask_keys) else "unknown"

        # 提取标签名称
        # label_name = extract_label_name(filename)
        # print(f"Processing file: {filename},label_name: {label_name}")
        
        # 获取对应的掩码，如果标签不在 label_masks 中，使用零向量
        mask = label_masks[label_name] # , np.zeros(18)
        mask = torch.tensor(mask, dtype=torch.float32)  # 转换为tensor
        mask = mask.to(y.device)  # 确保掩码在同一设备上

        
        # 将预测向量 y[i] 与掩码相乘，并与融合结果进行逻辑或操作
        masked_y = y_i * mask  # 只保留掩码为 1 的位置
        fused_result += masked_y
        # print(f"fused_result: {fused_result}, y[i]:{y[i]},masked_y: {masked_y}, mask: {mask}")
        # fused_result = torch.maximum(fused_result, masked_y)  # 逐位取最大值（逻辑或）
    # print(f"y:{y},fused_result: {fused_result}")
    
    return fused_result

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, query_mask):
        """
        logits: shape [B, C]
        targets: shape [B, C]
        query_mask: shape [B, C], 0或1
        """
        # 确保所有输入在同一设备
        device = logits.device
        logits = logits.to(device)
        targets = targets.to(device)
        query_mask = query_mask.to(device)
        self.pos_weight = self.pos_weight.to(device)        
        # 逐样本逐类别loss（不reduction）
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')  # shape: [B, C]

        # mask掉query_mask=0的loss
        masked_loss = bce_loss * query_mask

        # 平均loss：除以mask区域的数量
        reduced_loss = torch.sum(masked_loss) / (torch.sum(query_mask) + 1e-14)

        return reduced_loss

# 实例化


# 调用
# loss = criterion(logits, targets, query_mask)


class BaseModel(L.LightningModule):
    def __init__(
            self,
            task,
            metric_computation_mode,
            result_plot,
            metrics,
            num_classes,
            name,
            lr,
            weight_decay,
            optimizer,
            nesterov,
            sam,
            adaptive_sam,
            scheduler,
            T_max,
            warmstart,
            epochs,
            mixup,
            mixup_alpha,
            label_smoothing,
            stochastic_depth,
            resnet_dropout,
            squeeze_excitation,
            apply_shakedrop,
            undecay_norm,
            zero_init_residual,
            input_dim,
            input_channels,
            pretrained,
            *args,
            **kwargs
    ):
        super(BaseModel, self).__init__()

        # Task
        self.task = task
        self._best_val_acc = 0

        # Metrics
        self.metric_computation_mode = metric_computation_mode
        self.result_plot_setting = result_plot
        metrics_dict = {}

        self.subtask = kwargs["subtask"]

        if self.subtask == "multiclass":
            metric_task = "multiclass"
        elif self.subtask == "multilabel":
            metric_task = "multilabel"

        if self.task == "Classification":
            if "acc" in metrics:
                metrics_dict["Accuracy"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )
            if "balanced_acc" in metrics:
                if "balanced_acc" in metrics:
                    metrics_dict["Balanced_Accuracy"] = BalancedAccuracy(
                        task=metric_task,
                        num_classes=num_classes,
                    )

            if "f1" in metrics:
                metrics_dict["F1"] = F1Score(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "f1_per_class" in metrics:
                metrics_dict["F1_per_class"] = F1Score(
                    average=None,
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "pr" in metrics:
                metrics_dict["Precision"] = Precision(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
                metrics_dict["Recall"] = Recall(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "top5acc" in metrics:
                metrics_dict["Accuracy_top5"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    top_k=5,
                    num_labels=num_classes,
                )
            if "auroc" in metrics:
                metrics_dict["AUROC"] = AUROC(
                    average="macro",
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )
            if "ap" in metrics:
                metrics_dict["AP"] = AveragePrecision(
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )

        elif self.task == "Regression":
            if "mse" in metrics:
                metrics_dict["MSE"] = MeanSquaredError()
            if "mae" in metrics:
                metrics_dict["MAE"] = MeanAbsoluteError()

        if self.result_plot_setting in ["val", "all"]:
            if self.task == "Classification":
                self.val_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task == "Regression":
                self.val_pred_list = []
                self.val_label_list = []
        if self.result_plot_setting == "all":
            if self.task == "Classification":
                self.train_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task == "Regression":
                self.train_pred_list = []
                self.train_label_list = []

        self.save_preds = True if kwargs["save_preds"] else False
        if self.save_preds:
            self.val_preds = CatMetric(dist_sync_on_step=False)
            self.val_labels = CatMetric(dist_sync_on_step=False)
            self.val_indices = CatMetric(dist_sync_on_step=False)

        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Val/")

        # Training Args
        self.name = name
        # self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.nesterov = nesterov
        self.sam = sam
        self.adaptive_sam = adaptive_sam
        self.scheduler = scheduler
        self.T_max = T_max
        self.warmstart = warmstart
        self.warmstart2 = kwargs["warmstart2"]
        self.epochs = epochs
        self.pretrained = pretrained

        # Regularization techniques
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha  # 0.2
        self.label_smoothing = label_smoothing  # 0.1
        self.stochastic_depth = (
            stochastic_depth  # 0.1 (with higher resolution maybe 0.2)
        )
        self.resnet_dropout = resnet_dropout  # 0.5
        self.se = squeeze_excitation
        self.apply_shakedrop = apply_shakedrop
        self.undecay_norm = undecay_norm
        self.zero_init_residual = zero_init_residual

        # Finetuning method
        self.finetuning_method = kwargs["finetune_method"]

        # Data and Dataloading
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.num_classes = num_classes

        # switch to manual optimization for Sharpness Aware Minimization
        if self.sam:
            self.automatic_optimization = False

        # Loss
        if self.task == "Classification":
            if self.subtask == "multiclass":
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing
                )
            elif self.subtask == "multilabel":
                weights = torch.tensor([9.211362733, 2.384068466, 8.295479204, 32.8629776, 2.992233613,
                        6.064870808, 3.176470588, 4.187083754, 3.022222222, 1.216071737,
                        1.677849552, 3.152851834, 7.123261694, 18.16629381, 13.8480647,
                        6.335045662, 10.81701149, 13.40695067]).cuda()
                
                # weights = torch.tensor([7, 2.5 , 8, 12,2.9,5,2.8,4,3,1.2,1.5,2.7,7,11,8,4,8,11]).cuda()
                # 平滑权重：w_i = sqrt(w_i)# log(1 + w_i)
                smoothed_weights = torch.sqrt(weights)
                # torch.log(1 + weights)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=smoothed_weights,reduction='none')  

                # self.criterion = MaskedBCEWithLogitsLoss(pos_weight=smoothed_weights) #自定义的加入structure_mask的BCEWithLogitsLoss
        elif self.task == "Regression":
            self.criterion = nn.MSELoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # print(f"Training step: batch_idx={batch_idx}, batch_size={len(batch[0])}")
        x, y, struct_mask = batch

        if self.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
            y_hat = self(inputs)

        else:
            y_hat = self(x)
            if self.num_classes == 1:
                y_hat = y_hat.view(-1)

        if x.shape[0] == 1 and len(y_hat.shape) == 1:
            # for cases where batch size is 1 and y_hat doesn't have a batch dim
            y_hat = y_hat.unsqueeze(0)

        if self.sam:
            opt = self.optimizers()

            # first forward-backward pass
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            opt.first_step(zero_grad=True)

            # second forward-backward pass
            if self.mixup:
                self.manual_backward(
                    mixup_criterion(
                        self.criterion, self(inputs), targets_a, targets_b, lam
                    )
                )
            else:
                if self.num_classes == 1:
                    self.manual_backward(self.criterion(self(x).view(-1), y))
                else:
                    self.manual_backward(self.criterion(self(x), y))
            opt.second_step(zero_grad=True)

        else:
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                # print("use BCE loss with label masks")
                # print(f"y_hat.shape={y_hat.shape}, y.shape={y.shape}, struct_mask.shape={struct_mask.shape}")
                # print(f"y_hat type: {type(y_hat)}, y type: {type(y)}, struct_mask type: {type(struct_mask)}")
                # print(f"y_hat: {y_hat}, y: {y}, struct_mask: {struct_mask}")
                bce_loss = self.criterion(
                    y_hat, y.float() if self.subtask == "multilabel" else y #, struct_mask
                )
                masked_loss = bce_loss * struct_mask
                # 平均loss：除以mask区域的数量
                loss = torch.sum(masked_loss) / (torch.sum(struct_mask) + 1e-14)


        self.log(
            "Train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if torch.isnan(y_hat).any():
            print("######################################### Model predicts NaNs!")

        # save metrics
        if self.metric_computation_mode == "stepwise":
            metrics_res = self.train_metrics(y_hat, y)
            if "Train/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Train/F1_per_class"]):
                    metrics_res["Train/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Train/F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        elif self.metric_computation_mode == "epochwise":
            if self.task == "Classification":
                if self.subtask == "multilabel":
                    self.train_metrics.update(torch.sigmoid(y_hat.detach()), y)
                elif self.subtask == "multiclass":
                    self.train_metrics.update(F.softmax(y_hat.detach(), dim=-1), y)
            else:
                self.train_metrics.update(y_hat.detach(), y)

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.update(y_hat, y)
        if hasattr(self, "train_pred_list"):
            self.train_pred_list.extend(y_hat)
            self.train_label_list.extend(y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, struct_mask = batch #,idd



        # self.predictions.append(y_hat)


        # print('fused_y:',fused_y)
        # print('fused_y_hat:',fused_y_hat)
        # return fused_y.unsqueeze(0), fused_y_hat.unsqueeze(0) # ,label_mask,idd
    
        y_hat = self(x)

        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        bce_loss = self.criterion(
            y_hat, y.float() if self.subtask == "multilabel" else y
        )
        masked_loss = bce_loss * struct_mask
        # 平均loss：除以mask区域的数量
        val_loss = torch.sum(masked_loss) / (torch.sum(struct_mask) + 1e-14)
        self.log(
            "Val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
        )

        # save metrics
        if self.metric_computation_mode == "stepwise":
            metrics_res = self.val_metrics(y_hat, y)
            if "Val/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Val/F1_per_class"]):
                    metrics_res["Val/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Val/F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )
        elif self.metric_computation_mode == "epochwise":
            if self.task == "Classification":
                if self.subtask == "multilabel":
                    self.val_metrics.update(torch.sigmoid(y_hat.detach()), y)
                elif self.subtask == "multiclass":
                    self.val_metrics.update(F.softmax(y_hat.detach(), dim=-1), y)
            else:
                self.val_metrics.update(y_hat.detach(), y)


        # if hasattr(self, "val_conf_mat"):
        #     self.val_conf_mat.update(y_hat, y)
        if hasattr(self, "val_preds"):
            """self.val_pred_list.extend(y_hat.detach().cpu())
            self.val_label_list.extend(y.detach().cpu())"""
            actual_batch_size = x.size(0)  # dynamic size (works for last batch)
            start_idx = batch_idx * self.trainer.val_dataloaders.batch_size
            idx = torch.arange(
                start_idx, start_idx + actual_batch_size, device=self.device
            )

            self.val_preds.update(y_hat.detach())
            self.val_labels.update(y.detach())
            self.val_indices.update(idx)

    # def predict_step(self, batch, batch_idx): # bs=1

    #     x, y , idd= batch
    #     # x:bs,7,1,128,256,256
    #     x = x.squeeze(0) 

    #     y_hat = self(x)
    #     print(f"y_hat:{y_hat.shape}")
     
    #     # fused_y = fuse_predictions(y, idd) #, label_mask)
    #     fused_y_hat = fuse_predictions(y_hat, idd) #, label_mask)
    #     return y, fused_y_hat.unsqueeze(0), idd # ,label_mask,idd
    def predict_step(self, batch, batch_idx):
        x, idd= batch
        x = x.to(self.device)

        # x:bs,7,1,128,256,256
        bs = x.size(0)
        # print(bs)
        y_hat_list = []
        for i in range(bs):
            x_single = x[i:i+1]
            y_hat_single = self(x_single.squeeze(0))
            # print(y_hat_single)
            fused_y_hat = fuse_predictions(y_hat_single,idd[i])
            y_hat_list.append(fused_y_hat)
        y_hat = torch.stack(y_hat_list, dim=0) # torch.cat(y_hat_list,dim=0) # y_hat:torch.Size([36])

        return  y_hat, idd # y:torch.Size([2, 18])


    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get("Val/loss", None)
        if val_loss is not None and self.trainer.is_global_zero:
            print(f"[Val Epoch {self.current_epoch}] Val/Loss: {val_loss:.4f}")
        
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.val_metrics.compute()
            if self.trainer.is_global_zero:
                metrics_str = " | ".join(
                    f"{k}: {v:.4f}" if not torch.isnan(v) else f"{k}: NaN"
                    for k, v in metrics_res.items()
                )
                print(f"[val Epoch {self.current_epoch}] {metrics_str}")

            if "Val/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Val/F1_per_class"]):
                    metrics_res["Val/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Val/F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )

            self.val_metrics.reset()

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.save_state(self, "val")
            self.val_conf_mat.reset()
        if hasattr(self, "val_preds"):
            """# Stack tensors along batch dim
            val_preds = torch.stack(self.val_pred_list, dim=0).to(self.device)
            val_labels = torch.stack(self.val_label_list, dim=0).to(self.device)
            # print(len(self.val_pred_list), val_preds.shape)
            # Gather from all GPUs
            preds_all = self.all_gather(val_preds)
            preds_all = preds_all.view(-1, *preds_all.shape[2:])
            labels_all = self.all_gather(val_labels)
            labels_all = labels_all.view(-1, *labels_all.shape[2:])
            # print(preds_all.shape)"""

            preds_all = self.val_preds.compute()  # shape: [N_total, C]
            labels_all = self.val_labels.compute()
            indices = self.val_indices.compute()

            if self.trainer.is_global_zero:
                # Sort by original index to preserve dataset order
                sorted_idx = torch.argsort(indices)
                preds_all = preds_all[sorted_idx]
                labels_all = labels_all[sorted_idx]
                if self.task == "Regression":
                    data = [[x, y] for (x, y) in zip(labels_all, preds_all)]
                    table = wandb.Table(
                        data=data, columns=["Ground Truth", "Prediction"]
                    )
                    wandb.log(
                        {
                            "Val Scatterplot": wandb.plot.scatter(
                                table,
                                "Ground Truth",
                                "Prediction",
                                "Validation Scatterplot",
                            )
                        }
                    )
                if self.save_preds:

                    if self.task == "Classification":
                        columns = (
                                      (["GT_" + str(i) for i in range(len(labels_all[0]))])
                                      if self.subtask == "multilabel"
                                      else ["GT"]
                                  ) + ["Pred_" + str(i) for i in range(len(preds_all[0]))]
                        data = [
                            (
                                    (x.tolist() if self.subtask == "multilabel" else [x])
                                    + (
                                        F.softmax(y, dim=-1)
                                        if self.subtask == "multiclass"
                                        else torch.sigmoid(y)
                                    ).tolist()
                            )
                            for x, y in zip(labels_all, preds_all)
                        ]
                        table = wandb.Table(data=data, columns=columns)
                        wandb.log({"Val Predictions": table})
                    else:
                        raise NotImplementedError

            # reset
            self.val_preds.reset()
            self.val_labels.reset()
            self.val_indices.reset()

        # save ckpt
        # if hasattr(self.trainer, "current_epoch") and self.trainer.current_epoch > 30:
        #     val_acc = self.trainer.callback_metrics.get("Val/F1") #or \
        #             #   self.trainer.callback_metrics.get("Val/acc") or \
        #             #   self.trainer.callback_metrics.get("val_acc")
        #     if val_acc is not None:
        #         val_acc = val_acc.item() if hasattr(val_acc, "item") else val_acc
        #         # if not hasattr(self, "_best_val_acc"):
        #         #     self._best_val_acc = val_acc
        #         if val_acc > self._best_val_acc:
        #             self._best_val_acc = val_acc
        #             if self.trainer.is_global_zero:
        #                 save_path = f"{self.trainer.log_dir}/best_f1{val_acc:.4f}_epoch{self.trainer.current_epoch}.pth"
        #                 torch.save(self.state_dict(), save_path)
        #                 print(f"[INFO] Epoch {self.trainer.current_epoch}: val_f1 improved to {val_acc:.4f}, model saved to {save_path}")



    def on_train_epoch_end(self) -> None:
        train_loss = self.trainer.callback_metrics.get("Train/loss", None)
        if train_loss is not None and self.trainer.is_global_zero:
            optimizers = self.trainer.optimizers
            if optimizers:
                lr_list = [group["lr"] for group in optimizers[0].param_groups]  # 假设使用单个优化器
                if len(lr_list) == 1:
                    print(f"[[Train Epoch {self.current_epoch}] Train/Loss: {train_loss:.4f}, Learning Rate: {lr_list[0]:.6f}")
                else:
                    lr_str = " | ".join(f"LR_group_{i}: {lr:.6f}" for i, lr in enumerate(lr_list))
                    print(f"[Train Epoch {self.current_epoch}] Train/Loss: {train_loss:.4f}, Learning Rates: {lr_str}")
            else:
                print(f"[Train Epoch {self.current_epoch}] Train/Loss: {train_loss:.4f}")
                
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.train_metrics.compute()
                    # ===== 显式打印本轮 loss 和 metric =====
            if self.trainer.is_global_zero:
                metrics_str = " | ".join(
                    f"{k}: {v:.4f}" if not torch.isnan(v) else f"{k}: NaN"
                    for k, v in metrics_res.items()
                )
                print(f"[Train Epoch {self.current_epoch}] {metrics_str}")
                
            if "Train/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Train/F1_per_class"]):
                    metrics_res["Train/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Train/F1_per_class"]

            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )

            self.train_metrics.reset()

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.save_state(self, "train")
            self.train_conf_mat.reset()
        if hasattr(self, "train_pred_list"):
            data = [
                [x, y] for (x, y) in zip(self.train_label_list, self.train_pred_list)
            ]
            table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
            wandb.log(
                {
                    "Train Scatterplot": wandb.plot.scatter(
                        table, "Ground Truth", "Prediction", "Train Scatterplot"
                    )
                }
            )
            # reset
            self.train_pred_list = []
            self.train_label_list = []

    def on_train_start(self):
        # from models.preact_resnet import PreActBlock, PreActBottleneck
        # from models.pyramidnet import BasicBlock as BasicBlock_pyramid
        # from models.pyramidnet import Bottleneck as Bottleneck_pyramid
        # from models.resnet import BasicBlock, Bottleneck
        # from models.wide_resnet import BasicBlock as Wide_BasicBlock
        # from models.wide_resnet import Bottleneck as Wide_Bottleneck
        # total_len = 50#len(self.train_dataset)
        # subset_size = int(total_len * 0.1)
        # indices = np.random.choice(total_len, size=subset_size, replace=False)
        # self.current_sampler = SubsetRandomSampler(indices)
        # self.trainer.reset_train_dataloader()
        print("Training start")
        if not self.pretrained:
            print("Initializing weights")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    # nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            import os
            self.pretrained_path = "/path/to/fangyunhan/VLM/SSL3D_classification/run/ckpt/ctrate/checkpoints/output_crop/0/epoch=311-crop_mask-val_F1=0.00.ckpt"
            if hasattr(self, "pretrained_path") and os.path.exists(self.pretrained_path):
                print(f"Loading pretrained weights from {self.pretrained_path}")
                checkpoint = torch.load(self.pretrained_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)  # 兼容不同格式的检查点
                self.load_state_dict(state_dict, strict=True)
                print("Pretrained weights loaded successfully")
            
            
            # "/path/to/fangyunhan/VLM/SSL3D_classification/pretrain/checkpoint_latest.pth"
            # if hasattr(self, "pretrained_path") and os.path.exists(self.pretrained_path):
            #     print(f"Loading pretrained weights from {self.pretrained_path}")
            #     checkpoint = torch.load(self.pretrained_path, map_location="cpu")
            
            #     encoder_state_dict = {
            #             f"res_unet.{k}": v for k, v in checkpoint["network_weights"].items()
            #             if not k.startswith("decoder.seg_layers.")
            #         }
                
            #     missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            #     print(f"缺失的键: {missing_keys}")
            #     print(f"多余的键: {unexpected_keys}")

            else:
                print(f"Warning: Pretrained path {self.pretrained_path} does not exist!")
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            # TODO
            # if self.zero_init_residual:
            #     if "PreAct" in self.name:
            #         for m in self.modules():
            #             if isinstance(m, PreActBottleneck):
            #                 nn.init.constant_(m.conv3.weight, 0)
            #             elif isinstance(m, PreActBlock):
            #                 nn.init.constant_(m.conv2.weight, 0)
            #
            #     elif "ResNet" in self.name or "WRN" in self.name:
            #         for m in self.modules():
            #             if isinstance(m, Bottleneck) or isinstance(m, Wide_Bottleneck):
            #                 nn.init.constant_(m.bn3.weight, 0)
            #             elif isinstance(m, BasicBlock) or isinstance(
            #                     m, Wide_BasicBlock
            #             ):
            #                 nn.init.constant_(m.bn2.weight, 0)
            #
            #     elif "Pyramid" in self.name:
            #         for m in self.modules():
            #             if isinstance(m, Bottleneck_pyramid):
            #                 nn.init.constant_(m.bn4.weight, 0)
            #             elif isinstance(m, BasicBlock_pyramid):
            #                 nn.init.constant_(m.bn3.weight, 0)

    def configure_optimizers(self):
        # leave bias and params of batch norm undecayed as in https://arxiv.org/pdf/1812.01187.pdf (Bag of tricks)
        if self.undecay_norm:
            model_params = []
            norm_params = []
            for name, p in self.named_parameters():
                if p.requires_grad:
                    if "norm" in name or "bias" in name or "bn" in name:
                        norm_params += [p]
                    else:
                        model_params += [p]
            params = [
                {"params": model_params},
                {"params": norm_params, "weight_decay": 0},
            ]
        else:
            params = self.parameters()

        if self.finetuning_method == "full_sawtooth":
            # Separate encoder and cls_head parameters
            encoder_params = []
            cls_head_params = []

            for name, param in self.named_parameters():
                if "encoder" in name:
                    encoder_params.append(param)
                elif "cls_head" in name:
                    cls_head_params.append(param)

        if not self.sam:
            if self.optimizer == "SGD":
                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.SGD(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "nesterov": self.nesterov,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "nesterov": self.nesterov,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.SGD(
                        params,
                        lr=self.lr,
                        momentum=0.9,
                        weight_decay=self.weight_decay,
                        nesterov=self.nesterov,
                    )
            elif self.optimizer == "Adam":
                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.Adam(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.Adam(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "AdamW":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.AdamW(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "Rmsprop":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = RMSpropTF(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = RMSpropTF(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "Madgrad":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = MADGRAD(
                        [
                            {
                                "params": cls_head_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = MADGRAD(
                        params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
                    )

        else:
            # ASAM paper suggests 10x larger rho for adaptive SAM than in normal SAM
            rho = 0.5 if self.adaptive_sam else 0.05

            if self.optimizer == "SGD":
                base_optimizer = torch.optim.SGD
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    nesterov=self.nesterov,
                    rho=rho,
                )
            elif self.optimizer == "Madgrad":
                base_optimizer = MADGRAD
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "Adam":
                base_optimizer = torch.optim.Adam
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "AdamW":
                base_optimizer = torch.optim.AdamW
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "Rmsprop":
                base_optimizer = RMSpropTF
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )

        if not self.scheduler:
            return [optimizer]
        else:
            if self.scheduler == "CosineAnneal" and self.warmstart == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max
                )
            elif self.scheduler == "CosineAnneal" and self.warmstart > 0:
                if self.finetuning_method == "full_sawtooth":
                    scheduler = CosineAnnealingLR_DoubleWarmstart(
                        optimizer,
                        T_max=self.T_max,
                        warmstart1=self.warmstart,
                        warmstart2=self.warmstart2,
                    )
                else:
                    scheduler = CosineAnnealingLR_Warmstart(
                        optimizer,
                        T_max=self.T_max,
                        warmstart=self.warmstart,
                    )
            elif self.scheduler == "Step":
                # decays every 1/4 epochs
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.epochs // 4, gamma=0.1
                )
            elif self.scheduler == "MultiStep":
                # decays lr with 0.1 after half of epochs and 3/4 of epochs
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [self.epochs // 2, self.epochs * 3 // 4]
                )

            return [optimizer], [scheduler]


class CosineAnnealingLR_Warmstart(_LRScheduler):
    """
    Same as CosineAnnealingLR but includes a warmstart option that will gradually increase the LR
    for the amount of specified warmup epochs as described in https://arxiv.org/pdf/1706.02677.pdf
    """

    def __init__(
            self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmstart=0
    ):
        self.T_max = T_max - warmstart  # do not consider warmstart epochs for T_max
        self.eta_min = eta_min
        self.warmstart = warmstart
        self.T = 0

        super(CosineAnnealingLR_Warmstart, self).__init__(
            optimizer, last_epoch,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        # Warmstart
        if self.last_epoch < self.warmstart:
            addrates = [(lr / (self.warmstart + 1)) for lr in self.base_lrs]
            updated_lr = [
                addrates[i] * (self.last_epoch + 1)
                for i, group in enumerate(self.optimizer.param_groups)
            ]

            return updated_lr

        else:
            if self.T == 0:
                self.T += 1
                return self.base_lrs
            elif (self.T - 1 - self.T_max) % (2 * self.T_max) == 0:
                updated_lr = [
                    group["lr"]
                    + (base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max))
                    / 2
                    for base_lr, group in zip(
                        self.base_lrs, self.optimizer.param_groups
                    )
                ]

                self.T += 1
                return updated_lr

            updated_lr = [
                (1 + math.cos(math.pi * self.T / self.T_max))
                / (1 + math.cos(math.pi * (self.T - 1) / self.T_max))
                * (group["lr"] - self.eta_min)
                + self.eta_min
                for group in self.optimizer.param_groups
            ]

            self.T += 1
            return updated_lr


class CosineAnnealingLR_DoubleWarmstart(_LRScheduler):
    """
    CosineAnnealingLR with two consecutive warmup phases.

    - Warmup 1: Increases LR from 0 to base LR, **only for `cls_head`**.
    - Warmup 2: Increases LR from 0 to base LR, **for both `cls_head` and `encoder`**.
    - Cosine Annealing: Decays LR **for both `cls_head` and `encoder`**.
    """

    def __init__(
            self,
            optimizer,
            T_max,
            eta_min=0,
            last_epoch=-1,
            verbose=False,
            warmstart1=0,
            warmstart2=0,
    ):
        self.warmstart1 = warmstart1
        self.warmstart2 = warmstart2
        self.eta_min = eta_min
        self.T_max = T_max - (warmstart1 + warmstart2)  # Effective decay period
        self.T = 0  # Internal counter

        # Identify param groups: assume "cls_head" and "encoder" are named properly in optimizer param_groups
        self.cls_head_group = None
        self.encoder_group = None

        for param_group in optimizer.param_groups:
            if param_group.get("name") == "cls_head":
                self.cls_head_group = param_group
            elif param_group.get("name") == "encoder":
                self.encoder_group = param_group

        if self.cls_head_group is None:
            raise ValueError("Optimizer must have a parameter group named 'cls_head'.")
        if self.encoder_group is None:
            raise ValueError("Optimizer must have a parameter group named 'encoder'.")

        super(CosineAnnealingLR_DoubleWarmstart, self).__init__(
            optimizer, last_epoch,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        warmup_total = self.warmstart1 + self.warmstart2

        # First warmup phase (only `cls_head` is trained)
        if self.last_epoch < self.warmstart1:
            warmup_factor = (self.last_epoch + 1) / self.warmstart1
            updated_lr = []

            for group in self.optimizer.param_groups:
                if group is self.cls_head_group:
                    updated_lr.append(group["initial_lr"] * warmup_factor)
                else:  # Keep encoder frozen
                    updated_lr.append(0)

            return updated_lr

        # Second warmup phase (both `cls_head` and `encoder` are trained)
        elif self.last_epoch < warmup_total:
            warmup_factor = (self.last_epoch - self.warmstart1 + 1) / self.warmstart2
            updated_lr = [
                group["initial_lr"] * warmup_factor
                for group in self.optimizer.param_groups
            ]
            return updated_lr

        # Cosine annealing phase (both `cls_head` and `encoder`)
        else:
            epoch_cosine = self.last_epoch - warmup_total  # Shifted epoch count
            updated_lr = [
                self.eta_min
                + (group["initial_lr"] - self.eta_min)
                * 0.5
                * (1 + math.cos(math.pi * epoch_cosine / self.T_max))
                for group in self.optimizer.param_groups
            ]
            return updated_lr


class ModelConstructor(BaseModel):
    def __init__(self, model, **kwargs):
        super(ModelConstructor, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
