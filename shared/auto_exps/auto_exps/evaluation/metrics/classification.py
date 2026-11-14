# -*- coding: utf-8 -*-
"""
Classification Metrics - 分类指标计算

提供多标签分类的完整指标计算：
- calc_cls_metrics: precision/recall/f1/sensitivity/specificity (per-class + macro/micro)
- calc_crg_metrics: CT-RATE Grand Challenge 指标
- calc_per_sample_metrics: 每样本指标
- align_predictions: GT 与预测对齐
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score
)


def align_predictions(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_cols: List[str],
    pred_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐预测和真值数据

    Args:
        gt_df: 真值 DataFrame
        pred_df: 预测 DataFrame
        id_col: ID 列名
        gt_cols: 真值列名列表
        pred_cols: 预测列名列表（默认使用 gt_cols）

    Returns:
        (对齐后的真值 DataFrame, 对齐后的预测 DataFrame)
    """
    if pred_cols is None:
        pred_cols = gt_cols

    # 清理 ID 列中的后缀
    gt_ids = (
        gt_df[id_col].astype(str)
        .str.replace('.nii.gz', '', regex=False)
        .str.replace('.nii', '', regex=False)
    )
    pred_ids = (
        pred_df[id_col].astype(str)
        .str.replace('.nii.gz', '', regex=False)
        .str.replace('.nii', '', regex=False)
    )

    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    gt_df[id_col] = gt_ids
    pred_df[id_col] = pred_ids

    # 设置索引并对齐
    gt_df = gt_df.set_index(id_col)
    pred_df = pred_df.set_index(id_col)

    common_idx = gt_df.index.intersection(pred_df.index)

    gt_aligned = gt_df.loc[common_idx, gt_cols]
    pred_aligned = pred_df.loc[common_idx, pred_cols]

    return gt_aligned, pred_aligned


def calc_auc_block(
    gt_aligned: pd.DataFrame,
    pred_aligned: pd.DataFrame,
    gt_cols: List[str],
    pred_cols: List[str]
) -> Dict[str, Any]:
    """
    阈值无关的排序指标：逐类 AUC（ROC-AUC）与 AP（average precision）。

    **必须输入连续分数**。报告链（生成文本 -> 标签抽取）在抽取器只吐 0/1 时
    拿不到分数，此时 AUC 无定义——这里显式返回 auc=None 并给出原因，而不是
    把二值预测当分数算出一个会被误读成 AUC 的数（那其实等于 balanced accuracy）。

    GT 里只有一个类别的类（全 0 或全 1）AUC 无定义，跳过并记入 skipped_classes。
    macro AUC 在这些"有定义"的类上取平均，与被跳过的类数一并返回，
    因为 MR/Abd 这类长尾标签集里两者可能相差很多。

    Returns:
        {'auc','ap','per_class_auc','per_class_ap','n_classes','skipped_classes','source'}
    """
    vals = pred_aligned.values
    if vals.size == 0:
        return {
            'auc': None, 'ap': None, 'per_class_auc': {}, 'per_class_ap': {},
            'n_classes': 0, 'skipped_classes': list(gt_cols), 'source': None,
            'reason': 'no samples'
        }

    uniq = np.unique(vals[~np.isnan(vals.astype(float))])
    is_binary = np.all(np.isin(uniq, [0, 1])) if uniq.size else True
    if is_binary:
        return {
            'auc': None, 'ap': None, 'per_class_auc': {}, 'per_class_ap': {},
            'n_classes': 0, 'skipped_classes': list(gt_cols), 'source': 'binary',
            'reason': ('predictions are 0/1 labels with no score attached '
                       '(label-extraction head); AUC needs continuous scores')
        }

    aucs, aps, skipped = {}, {}, []
    for i, gt_col in enumerate(gt_cols):
        y = gt_aligned[gt_col].values.astype(int)
        s = pred_aligned[pred_cols[i]].values.astype(float)
        if 0 < y.sum() < len(y) and not np.all(np.isnan(s)):
            aucs[gt_col] = float(roc_auc_score(y, s))
            aps[gt_col] = float(average_precision_score(y, s))
        else:
            skipped.append(gt_col)

    n = max(len(aucs), 1)
    return {
        'auc': float(sum(aucs.values()) / n) if aucs else None,
        'ap': float(sum(aps.values()) / n) if aps else None,
        'per_class_auc': aucs,
        'per_class_ap': aps,
        'n_classes': len(aucs),
        'skipped_classes': skipped,
        'source': 'probability',
    }


def calc_cls_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_cols: List[str],
    pred_cols: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    计算分类指标 (v1 完整格式)

    Args:
        gt_df: 真值 DataFrame
        pred_df: 预测 DataFrame (概率值或二值)
        id_col: ID 列名
        gt_cols: 真值列名列表
        pred_cols: 预测列名列表（默认使用 gt_cols）
        threshold: 二值化阈值

    Returns:
        指标字典，包含 per_pathology、macro、micro 三部分
    """
    if pred_cols is None:
        pred_cols = gt_cols

    gt_aligned, pred_aligned = align_predictions(gt_df, pred_df, id_col, gt_cols, pred_cols)

    # 阈值无关的 AUC/AP：分类任务的主指标
    # （口径见 U-VLM 评估文档）
    auc_block = calc_auc_block(gt_aligned, pred_aligned, gt_cols, pred_cols)

    # 判断是否需要二值化
    max_val = pred_aligned.values.max()
    if max_val <= 1.0 and pred_aligned.values.min() >= 0.0:
        pred_binary = (pred_aligned >= threshold).astype(int)
    else:
        pred_binary = pred_aligned.astype(int)

    per_pathology = []
    for i, gt_col in enumerate(gt_cols):
        pred_col = pred_cols[i]
        y_true = gt_aligned[gt_col].values.astype(int)
        y_pred = pred_binary[pred_col].values.astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_pathology.append({
            'name': gt_col,
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(f1),
            'accuracy': float(acc),
            'auc': auc_block['per_class_auc'].get(gt_col),
            'ap': auc_block['per_class_ap'].get(gt_col),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })

    # Macro 平均
    macro = {
        'precision': float(np.mean([p['precision'] for p in per_pathology])),
        'recall': float(np.mean([p['recall'] for p in per_pathology])),
        'sensitivity': float(np.mean([p['sensitivity'] for p in per_pathology])),
        'specificity': float(np.mean([p['specificity'] for p in per_pathology])),
        'f1': float(np.mean([p['f1'] for p in per_pathology])),
        'accuracy': float(np.mean([p['accuracy'] for p in per_pathology])),
        'auc': auc_block['auc'],
        'ap': auc_block['ap'],
    }

    # Micro 平均
    total_tp = sum(p['tp'] for p in per_pathology)
    total_fp = sum(p['fp'] for p in per_pathology)
    total_fn = sum(p['fn'] for p in per_pathology)
    total_tn = sum(p['tn'] for p in per_pathology)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    micro = {
        'precision': float(micro_precision),
        'recall': float(micro_recall),
        'sensitivity': float(micro_recall),
        'specificity': float(total_tn / (total_tn + total_fp)) if (total_tn + total_fp) > 0 else 0.0,
        'f1': float(micro_f1),
        'accuracy': float(micro_accuracy),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn
    }

    return {
        'per_pathology': per_pathology,
        'macro': macro,
        'micro': micro,
        'auc': auc_block,
        'primary_metric': 'macro_auc',
        'num_samples': len(gt_aligned),
        'num_classes': len(gt_cols),
        'threshold': threshold
    }


def calc_crg_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_cols: List[str],
    pred_cols: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    计算 CRG 指标 (CT-RATE Grand Challenge 格式)

    CRG = max(0, min(1, (TP - FP) / A + 1/3))
    A = TP + FN (实际正例总数)

    Args:
        gt_df: 真值 DataFrame
        pred_df: 预测 DataFrame
        id_col: ID 列名
        gt_cols: 真值列名列表
        pred_cols: 预测列名列表（默认使用 gt_cols）
        threshold: 二值化阈值

    Returns:
        CRG 指标字典
    """
    if pred_cols is None:
        pred_cols = gt_cols

    gt_aligned, pred_aligned = align_predictions(gt_df, pred_df, id_col, gt_cols, pred_cols)

    max_val = pred_aligned.values.max()
    if max_val <= 1.0 and pred_aligned.values.min() >= 0.0:
        pred_binary = (pred_aligned >= threshold).astype(int)
    else:
        pred_binary = pred_aligned.astype(int)

    gt_values = gt_aligned.values.astype(int)
    pred_values = pred_binary.values.astype(int)

    tp = int(np.sum((gt_values == 1) & (pred_values == 1)))
    fn = int(np.sum((gt_values == 1) & (pred_values == 0)))
    fp = int(np.sum((gt_values == 0) & (pred_values == 1)))
    tn = int(np.sum((gt_values == 0) & (pred_values == 0)))

    a = tp + fn  # 实际正例总数
    x = gt_values.size  # 总标签数

    # CRG 计算
    if a == 0:
        crg = 1.0 if (tp == 0 and fp == 0) else 0.0
    else:
        crg = max(0.0, min(1.0, (tp - fp) / a + 1/3))

    u = fp + fn / 2
    score_s = tp - u

    return {
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'TN': tn,
        'A': a,
        'X': x,
        'U': float(u),
        'score_s': float(score_s),
        'CRG': float(crg),
        'threshold': threshold
    }


def calc_per_sample_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_cols: List[str],
    pred_cols: Optional[List[str]] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    计算每个样本的指标

    Args:
        gt_df: 真值 DataFrame
        pred_df: 预测 DataFrame
        id_col: ID 列名
        gt_cols: 真值列名列表
        pred_cols: 预测列名列表（默认使用 gt_cols）
        threshold: 二值化阈值

    Returns:
        包含每样本指标的 DataFrame
    """
    if pred_cols is None:
        pred_cols = gt_cols

    gt_aligned, pred_aligned = align_predictions(gt_df, pred_df, id_col, gt_cols, pred_cols)

    max_val = pred_aligned.values.max()
    if max_val <= 1.0 and pred_aligned.values.min() >= 0.0:
        pred_binary = (pred_aligned >= threshold).astype(int)
    else:
        pred_binary = pred_aligned.astype(int)

    results = []
    for idx in gt_aligned.index:
        y_true = gt_aligned.loc[idx].values.astype(int)
        y_pred = pred_binary.loc[idx].values.astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        results.append({
            id_col: idx,
            'TP': tp,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        })

    return pd.DataFrame(results)
