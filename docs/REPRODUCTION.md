# Reproduction Notes

## Environment

- Conda env used by the original report-generation run: `medmvl`
  (`/path/to/miniconda3/envs/medmvl`).
- Code snapshot: `code/jiawei_report_generate_docker/`.
- Base models used by training:
  - `cache_dir_hf/Qwen3-8B`
  - `cache_dir_hf/models--google--siglip2-so400m-patch14-384/...`
  These live in the original Jiawei workspace; they were not copied into this
  workspace (they are dependencies, not final weights).
- Final report weights: `weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune/`.

## Path sanitization required

The copied scripts still contain old absolute paths:

```text
/path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/...
/path/to/jiawei_database/ctrate_raw
```

`/path/to/jiawei_workspace` and `/path/to/jiawei_database/ctrate_raw`
no longer exist. Before public release:

1. Replace the old workspace prefix with the public repo root.
2. Replace `image3d_path` with a locally available CT-RATE preprocessed
   directory.
3. Remove private host paths and credentials.

## Report-generation training outline

From `scripts/former/final_siglip_384_qwen3_4b_pretrain.sh`:

- DeepSpeed ZeRO-3, 8 GPUs.
- LLM and image encoder frozen; connector trained.
- 1 epoch; per-device batch 8; gradient accumulation 2; LR 1e-3.
- Data: `datasets/final_data/final_pretrain.json`.

From `scripts/former/final_siglip_384_qwen3_4b_finetune.sh`:

- Resume from the pretrain output.
- LLM, image encoder and connector full-finetuned.
- 1 epoch; per-device batch 8; gradient accumulation 2; LR 2e-5.
- Data: `datasets/final_data/final_finetune.json`.

Training data JSONs are excluded from this workspace (1.6 GB). The generators
are kept in `code/jiawei_report_generate_docker/datasets/final_data/`; obtain
CT-RATE under its upstream terms.

## Inference

The original inference script is under
`code/jiawei_report_generate_docker/scripts/new/inference.sh` and points at a
local `siglip384-finetune` checkpoint. For the final model, change
`MODEL_PATH` to:

```text
weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune
```

and set `--image3d_path` to the local CT-RATE preprocessed validation/inference
directory.

## Weights

See `docs/MODEL_WEIGHTS.md`. The final report checkpoint is copied; the
classification final checkpoint is still to be confirmed.

## Docker

No final 2025 challenge Docker image or base image was found in the local
snapshots. The 2026 reference repo documents a private `med-swift` base image
that is not publicly distributed. Document the actual 2025 image/base status
before claiming end-to-end Docker reproducibility.

## Validation checklist

- [ ] Load final weights without missing/unexpected keys.
- [ ] Run one CT-RATE case end-to-end.
- [ ] Verify output schema and report body.
- [ ] Verify offline mode.
- [ ] Verify no patient identifiers in logs/outputs.
