#!/usr/bin/env python3
"""Combine one CT_CLS8 prediction and one generated report into case JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--classification-csv", required=True)
    p.add_argument("--report-json", required=True)
    p.add_argument("--report-input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.classification_csv, dtype={"series_id": str})
    row = df[df["series_id"].astype(str) == str(args.case_id)]
    if len(row) != 1:
        raise RuntimeError(f"expected one classification row for {args.case_id}")
    row = row.iloc[0]

    report_obj = json.loads(Path(args.report_json).read_text())
    report_rec = report_obj[0] if isinstance(report_obj, list) else report_obj
    report_input = json.loads(Path(args.report_input).read_text())[0]

    result = {
        "case_id": args.case_id,
        "image": str(Path(args.image).resolve()),
        "classification": {
            "probabilities": {c: float(row[c]) for c in LABELS},
            "binary_0.5": {c: int(float(row[c]) >= 0.5) for c in LABELS},
        },
        "prompt": report_input["conversations"][0]["value"],
        "report": report_rec.get("generated_report", ""),
        "success": bool(report_rec.get("success", False)),
        "error": report_rec.get("error"),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
