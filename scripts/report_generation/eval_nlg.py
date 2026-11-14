#!/usr/bin/env python3
"""Evaluate MedM-VL generated reports with the CT-RATE protocol.

BLEU is computed with pycocoevalcap.
ROUGE-L and CIDEr are added for a fuller local picture.  METEOR is optional
because pycocoevalcap's Meteor needs a Java runtime.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--pred-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--id-col", default="series_id")
    p.add_argument("--gt-report-col", default="report")
    p.add_argument("--pred-report-col", default="generated_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from nlg_metrics import clean_report_text, calc_nlg_metrics

    gt = pd.read_csv(args.gt_csv)
    pred = pd.read_csv(args.pred_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bleu_metrics = calc_nlg_metrics(
        gt, pred, args.id_col, args.gt_report_col, args.pred_report_col
    )
    result = dict(bleu_metrics)

    # Additional metrics on the same cleaned, ID-matched pairs.
    try:
        gt_ids = gt[args.id_col].astype(str).str.replace(r"\.nii(\.gz)?$", "", regex=True)
        pred_ids = pred[args.id_col].astype(str).str.replace(r"\.nii(\.gz)?$", "", regex=True)
        gt_map = dict(zip(gt_ids, gt[args.gt_report_col]))
        pred_map = dict(zip(pred_ids, pred[args.pred_report_col]))
        common = sorted(set(gt_map) & set(pred_map))
        gts = {i: [clean_report_text(gt_map[k])] for i, k in enumerate(common)}
        res = {i: [clean_report_text(pred_map[k])] for i, k in enumerate(common)}

        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        rouge_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        result["ROUGE_L"] = float(rouge_score)
        result["CIDEr"] = float(cider_score)

        if shutil.which("java"):
            try:
                from pycocoevalcap.meteor.meteor import Meteor

                meteor_score, _ = Meteor().compute_score(gts, res)
                result["METEOR"] = float(meteor_score)
            except Exception as exc:
                result["METEOR"] = None
                result["METEOR_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["METEOR"] = None
            result["METEOR_error"] = "java not available"
    except Exception as exc:
        result["extra_metrics_error"] = f"{type(exc).__name__}: {exc}"

    result["num_samples"] = int(len(pred))
    (out_dir / "metrics_nlg.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
