#!/usr/bin/env python3
"""Run MedM-VL 8B report generation on a JSON case list.

This is a small, self-contained wrapper around the copied MedM-VL code.  It
uses one process per GPU/shard, writes predictions incrementally, and keeps
failed cases in the output instead of silently dropping them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]  # .../VLM3D
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument(
        "--medmvl-root",
        default=str(root / "code/jiawei_report_generate_docker/MedM-VL"),
    )
    p.add_argument("--conv-version", default="qwen3")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    medmvl_root = str(Path(args.medmvl_root).resolve())
    if medmvl_root not in sys.path:
        sys.path.insert(0, medmvl_root)

    from lvlm.dataset.dataset import (  # noqa: E402
        DataCollatorForMultiModalDataset,
        MultiModalDataset,
    )
    from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration  # noqa: E402
    from lvlm.utils.arguments import set_seed  # noqa: E402

    set_seed(42)
    data = json.loads(Path(args.input_json).read_text())
    if args.limit is not None:
        data = data[: args.limit]

    # The dataset resolves relative image paths against image3d_path; our
    # prepare script stores absolute paths, so "/" is a neutral base.
    data_args = argparse.Namespace(
        conv_version=args.conv_version,
        image_path=None,
        image3d_path="/",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16
    print(f"loading model: {args.model_dir}", flush=True)
    model = LVLMForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(dtype=model_dtype, device=device)
    model.eval()

    dataset = MultiModalDataset(
        model=model,
        data=data,
        data_arguments=data_args,
        mode="eval",
    )
    collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode="eval")

    records: list[dict] = []

    def save() -> None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        )
        pd.DataFrame(records).to_csv(args.output_csv, index=False)

    start = time.time()
    with torch.no_grad():
        for idx, item in enumerate(data):
            sid = item.get("series_id", item.get("id", str(idx)))
            try:
                sample = dataset[idx]
                if sample.get("image3d") is None and sample.get("image") is None:
                    raise RuntimeError("image preprocessing returned None")
                batch = collator([sample])
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.to(device)
                        if k in ("image", "image3d"):
                            batch[k] = v.to(dtype=model_dtype, device=device)

                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                )
                if args.temperature and args.temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=args.temperature)
                else:
                    gen_kwargs.update(do_sample=False)

                output_ids = model.generate(**batch, **gen_kwargs)
                text = model.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                rec = {
                    "series_id": sid,
                    "generated_report": text,
                    "success": True,
                    "error": None,
                }
            except Exception as exc:  # keep going, record the failure
                rec = {
                    "series_id": sid,
                    "generated_report": "",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[{sid}] FAILED: {rec['error']}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()

            records.append(rec)
            if (idx + 1) % args.save_every == 0 or idx == len(data) - 1:
                save()
                elapsed = time.time() - start
                print(
                    f"[{idx + 1}/{len(data)}] {sid} elapsed={elapsed:.0f}s "
                    f"success={sum(r['success'] for r in records)}",
                    flush=True,
                )

    save()
    print(
        json.dumps(
            {
                "cases": len(records),
                "success": sum(r["success"] for r in records),
                "failed": sum(not r["success"] for r in records),
                "output_json": args.output_json,
                "output_csv": args.output_csv,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
