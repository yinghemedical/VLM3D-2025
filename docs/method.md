# Method

## Tasks

The 2025 workspace contains final artifacts for:

1. **Multi-Abnormality Classification** on CT-RATE chest CT.
2. **Report Generation** on CT-RATE chest CT.

No final artifact was found for Abnormality Localization or Text-to-CT
Generation; confirm with the team before claiming those tasks.

## Report Generation

- Model family: **MedM-VL**.
- Image encoder: **SigLIP2-so400m-patch14-384**.
- LLM: **Qwen3-8B** (`cache_dir_hf/Qwen3-8B`).
- Training recipe:
  1. **Pretrain**: image encoder and LLM frozen; connector trained for 1 epoch.
  2. **Fine-tune**: image encoder, LLM and connector full-finetuned for 1 epoch.
- Final local weight snapshot:
  `weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune/`.
- Training records: `final_pretrain.json` 50,174; `final_finetune.json`
  50,179; VQA variant 767,932.
- Input at inference: CT volume + instruction.
- Output: structured findings + free-text report section.

## Classification

- Code lines found:
  - `code/jiawei_vlm3d_methods/` — challenge evaluation and class_model docs.
  - `code/yunhan_vlm3d_cls/` — VLM3D classification train/inference scripts.
  - `code/yunhan_vlm3d_dockers/abnclass_evaluation/` — challenge evaluation.
- Candidate checkpoints:
  - Jiawei LVLM `checkpoint_2699_epoch_1_0.2706.pt`.
  - Yunhan LVLM `checkpoint_2699_epoch_2_0.3047.pt`.
  - Yunhan SSL3D `best_f10.2578_epoch31.pth`.
- Exact final submitted checkpoint: **TODO — confirm with team**.

## Data

- Source: CT-RATE.
- Original training path recorded in scripts:
  `/path/to/jiawei_database/ctrate_raw` (now missing).
- Public reproduction must remap this to a locally available CT-RATE
  preprocessed directory.
- Images, reports, and patient-level identifiers are not redistributed.

## Artifact caveats

- Copied code still contains old absolute paths; sanitize before public release.
- The classification final checkpoint and final Docker image are not confirmed.
- Local metrics are not official challenge scores.
