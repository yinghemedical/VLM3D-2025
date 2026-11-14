#!/usr/bin/env python3
"""Run the original CT_CLS8 inference implementation on one shard directory.

CT_CLS8's `process.py` expects a container layout (`/input`, `/output`,
`/opt/app/ckpts`).  This wrapper keeps the original code untouched, runs it
from the original `docker5` tree (so `nnUNet_results` resolves), and patches
only the input/output/checkpoint globals.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument(
        "--cls-root",
        default=os.environ.get("CT_CLS_ROOT", str(Path(__file__).resolve().parents[2] / "code/classification/inference")),
        help="Original CT_CLS8 build context containing nnUNet_results and ckpts",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cls_root).resolve()
    if not (root / "process.py").is_file():
        raise FileNotFoundError(root / "process.py")
    if not (root / "nnUNet_results").is_dir():
        raise FileNotFoundError(root / "nnUNet_results")

    # The original dataset module initializes nnUNetPredictor at import time
    # with './nnUNet_results'; cwd must therefore be the original tree.
    os.chdir(root)
    sys.path.insert(0, str(root))

    import process_ct_cls8 as process  # noqa: E402

    process.make_omegaconf_resolvers()
    process.VOL_DIR = Path(args.shard_dir).resolve()
    process.OUT_FILE = Path(args.output_json).resolve()
    ckpt_dir = Path(
        os.environ.get(
            "CT_CLS_CKPT_DIR",
            "/path/to/VLM3D/weights/classification/docker_ct_cls8",
        )
    )
    process.CLASSIFIER_PATH = ckpt_dir / "epoch=263-crop_resume-val_F1=0.00.ckpt"
    process.CLASSIFIER_PATH2 = ckpt_dir / "epoch=371-crop_cls_head-val_F1=0.00.ckpt"
    process.CLASSIFIER_PATH3 = ckpt_dir / "epoch=92-crop_cls_head-f-val_F1=0.00.ckpt"
    process.OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {"exp_dir": "cli_configs/crop", "exp_dir2": "cli_configs/cls-head"}
    )
    process.inference.__wrapped__(cfg)


if __name__ == "__main__":
    main()
