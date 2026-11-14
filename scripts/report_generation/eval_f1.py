#!/usr/bin/env python3
"""Compute CT-RATE report-generation F1 with the AutoExps/RadBERT protocol.

This mirrors AutoExps `CTRateReportEvaluator`:
  generated report -> RadBERT -> 18 binary labels -> macro/micro F1 vs GT.
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--report-col", default="generated_report")
    p.add_argument("--autoexps-root", default=str(root / "shared/auto_exps"))
    p.add_argument(
        "--radbert-model",
        default="/path/to/RadBertClassifier.pth",
    )
    p.add_argument(
        "--roberta-dir",
        default="/path/to/robert",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    autoexps_root = Path(args.autoexps_root).resolve()
    autoexps_src = str(autoexps_root / "src" if (autoexps_root / "src").is_dir() else autoexps_root)
    if autoexps_src not in sys.path:
        sys.path.insert(0, autoexps_src)

    from auto_exps.evaluation.extractors.radbert import (  # noqa: E402
        CTRATE_CLS_COLS,
        RadBertExtractor,
    )
    from auto_exps.evaluation.metrics.classification import (  # noqa: E402
        calc_cls_metrics,
        calc_crg_metrics,
    )

    gt = pd.read_csv(args.gt_csv, dtype={args.id_col: str})
    pred = pd.read_csv(args.pred_csv, dtype={args.id_col: str})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    extractor = RadBertExtractor(
        args.radbert_model,
        CTRATE_CLS_COLS,
        base_model_dir=args.roberta_dir,
        device=args.device,
    )
    extracted = extractor.extract(pred, args.id_col, args.report_col)
    extracted.to_csv(out / "output_cls.csv", index=False)

    cls = calc_cls_metrics(
        gt, extracted, args.id_col, CTRATE_CLS_COLS, threshold=0.5
    )
    (out / "metrics_cls.json").write_text(json.dumps(cls, indent=2) + "\n")

    try:
        crg = calc_crg_metrics(
            gt, extracted, args.id_col, CTRATE_CLS_COLS, threshold=0.5
        )
        (out / "metrics_crg.json").write_text(json.dumps(crg, indent=2) + "\n")
    except Exception as exc:
        crg = {"error": f"{type(exc).__name__}: {exc}"}
        (out / "metrics_crg.json").write_text(json.dumps(crg, indent=2) + "\n")

    print(
        json.dumps(
            {
                "num_samples": cls["num_samples"],
                "macro_f1": cls["macro"]["f1"],
                "micro_f1": cls["micro"]["f1"],
                "macro_precision": cls["macro"]["precision"],
                "macro_recall": cls["macro"]["recall"],
                "macro_auc": cls["macro"].get("auc"),
                "crg": crg.get("CRG") if isinstance(crg, dict) else None,
                "output_dir": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
