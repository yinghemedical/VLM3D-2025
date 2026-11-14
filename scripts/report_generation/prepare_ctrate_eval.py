#!/usr/bin/env python3
"""Build the CT-RATE validation/test manifest for MedM-VL report inference.

The output follows the U-VLM CT-RATE report contract where possible:
  * GT CSV has `series_id`, `volume_name`, `report` and 18 label columns.
  * Predictions use `series_id` + `generated_report`.
The MedM-VL-specific input JSON adds the human prompt and absolute image path.
"""
from __future__ import annotations

import argparse
import json
import re
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

PROMPT_TEMPLATE = (
    "<image3d>\n"
    "Generate radiology report for the CT."
    "This patient may have the following diseases: {pos}\n"
    "This patient does not have the following diseases: {neg}\n"
)


def strip_volume_suffix(name: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", str(name))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ctrate-root", default="/path/to/Public_data/MM_data/CT-RATE")
    p.add_argument(
        "--reports",
        default="/path/to/Public_data/MM_data/CT-RATE_blosc2/radiology_text_reports/validation_reports.csv",
    )
    p.add_argument(
        "--labels",
        default="/path/to/Public_data/MM_data/CT-RATE_blosc2/multi_abnormality_labels/valid_predicted_labels.csv",
        help="GT label CSV used in gt.csv",
    )
    p.add_argument(
        "--prompt-labels",
        default=None,
        help="Optional classification-model output CSV; if set, prompt labels come from it instead of --labels.",
    )
    p.add_argument("--prompt-threshold", type=float, default=0.5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = pd.read_csv(args.reports)
    labels = pd.read_csv(args.labels)
    if "VolumeName" not in reports.columns or "VolumeName" not in labels.columns:
        raise ValueError("both CSVs must contain VolumeName")

    image_root = Path(args.ctrate_root) / "dataset" / "valid_fixed"
    if not image_root.is_dir():
        raise RuntimeError(f"missing image root: {image_root}")

    def image_path_for(series_id: str) -> Path | None:
        # CT-RATE layout: valid_1_a_1 -> valid_1/valid_1_a/valid_1_a_1.nii.gz
        parts = series_id.split("_")
        if len(parts) >= 4 and parts[0] == "valid":
            pid, scan = parts[1], parts[2]
            candidate = (
                image_root
                / f"valid_{pid}"
                / f"valid_{pid}_{scan}"
                / f"{series_id}.nii.gz"
            )
            if candidate.is_file():
                return candidate.resolve()
        # Rare/non-standard layout: locate only this one case.
        for candidate in image_root.rglob(f"{series_id}.nii.gz"):
            return candidate.resolve()
        return None

    label_map = labels.set_index("VolumeName")
    prompt_label_map = None
    if args.prompt_labels:
        prompt_labels = pd.read_csv(args.prompt_labels)
        id_col = None
        for candidate in ("series_id", "VolumeName", "AccessionNo", "identifier"):
            if candidate in prompt_labels.columns:
                id_col = candidate
                break
        if id_col is None:
            raise ValueError(
                f"prompt label CSV must contain series_id/VolumeName/AccessionNo/identifier: "
                f"{list(prompt_labels.columns)}"
            )
        prompt_labels = prompt_labels.drop_duplicates(subset=[id_col], keep="first").copy()
        prompt_labels["_series_id"] = prompt_labels[id_col].map(strip_volume_suffix)
        prompt_label_map = prompt_labels.set_index("_series_id")

    records = []
    prompt_values_list = []
    missing = []
    for _, row in reports.iterrows():
        volume_name = str(row["VolumeName"])
        series_id = strip_volume_suffix(volume_name)
        image_path = image_path_for(series_id)
        if image_path is None:
            missing.append(series_id)
            continue
        if volume_name not in label_map.index:
            raise KeyError(f"no labels for {volume_name}")
        lab = label_map.loc[volume_name]
        values = {c: int(float(lab[c]) > 0.5) for c in LABELS}

        if prompt_label_map is not None:
            if series_id not in prompt_label_map.index:
                raise KeyError(f"no classification output for {series_id}")
            plab = prompt_label_map.loc[series_id]
            prompt_values = {
                c: int(float(plab[c]) >= args.prompt_threshold) for c in LABELS
            }
        else:
            prompt_values = dict(values)

        report = (
            f"Findings: {str(row.get('Findings_EN', '')).strip()} "
            f"Impressions: {str(row.get('Impressions_EN', '')).strip()}"
        ).strip()
        rec = {
            "series_id": series_id,
            "volume_name": volume_name,
            "image_path": str(image_path),
            "report": report,
            **values,
        }
        records.append(rec)
        prompt_values_list.append(prompt_values)

    if missing and not args.allow_missing:
        raise RuntimeError(f"{len(missing)} volumes missing, first: {missing[:10]}")
    if args.max_cases is not None:
        records = records[: args.max_cases]
    if not records:
        raise RuntimeError("no records selected")

    gt = pd.DataFrame(records)
    gt.to_csv(out_dir / "gt.csv", index=False)

    items = [
        {
            "id": r["series_id"],
            "series_id": r["series_id"],
            "image3d": [r["image_path"]],
            "conversations": [
                {
                    "from": "human",
                    "value": PROMPT_TEMPLATE.format(
                        pos=", ".join([c for c in LABELS if prompt_values[c]]) or "None",
                        neg=", ".join([c for c in LABELS if not prompt_values[c]]) or "None",
                    ),
                }
            ],
        }
        for r, prompt_values in zip(records, prompt_values_list)
    ]
    (out_dir / "inputs.all.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")

    n = len(items)
    shards = [[] for _ in range(args.num_shards)]
    for i, item in enumerate(items):
        shards[i % args.num_shards].append(item)
    for rank, shard in enumerate(shards):
        p = out_dir / f"inputs.shard{rank:02d}.json"
        p.write_text(json.dumps(shard, indent=2, ensure_ascii=False) + "\n")
        print(f"shard {rank:02d}: {len(shard)} cases -> {p}")

    manifest = {
        "ctrate_root": args.ctrate_root,
        "reports": args.reports,
        "labels": args.labels,
        "prompt_labels": args.prompt_labels,
        "prompt_threshold": args.prompt_threshold,
        "num_cases": n,
        "num_shards": args.num_shards,
        "missing": len(missing),
        "prompt_template": PROMPT_TEMPLATE,
        "gt_csv": str(out_dir / "gt.csv"),
        "inputs_all": str(out_dir / "inputs.all.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
