# Model Weights

## Public weight repositories

- Task 1 (Report Generation):
  https://huggingface.co/yinghemedical/VLM3D2025-Task1-ReportGeneration
- Task 2 (Multi-Abnormality Classification):
  https://huggingface.co/yinghemedical/VLM3D2025-Task2-AbnormalityClassifications

## Report Generation

- Local snapshot:
  `weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune/`
- Size: ~17 GB; 4 safetensors shards + tokenizer/config files.
- Source:
  `/path/to/luyan_mllm/VLM3D_challenge/jiawei_workspace/report_generate_docker/exps/former/final-MedM-VL-siglip-384-qwen3-8b-finetune/`
- Expected files:
  - `model-00001-of-00004.safetensors`
  - `model-00002-of-00004.safetensors`
  - `model-00003-of-00004.safetensors`
  - `model-00004-of-00004.safetensors`
  - `config.json`, `tokenizer*`, `model.safetensors.index.json`
- Public URL: `https://huggingface.co/yinghemedical/VLM3D2025-Task1-ReportGeneration`
- SHA-256: see `manifests/weights_sha256.txt`

## Docker-internal weights

### Classification CT_CLS8

Extracted to `weights/classification/docker_ct_cls8/`:

| Checkpoint | SHA-256 |
|---|---|
| `epoch=263-crop_resume-val_F1=0.00.ckpt` | `a5b6a7ad...` |
| `epoch=371-crop_cls_head-val_F1=0.00.ckpt` | `81b78e7a...` |
| `epoch=92-crop_cls_head-f-val_F1=0.00.ckpt` | `91348c9e...` |

### Report Generation improved-clip

Extracted to `weights/report_generation/docker_improved_clip_medm_vl_3b/`:

- 2 shards of `MedM-VL-CT-Chest-3B-en` (~6 GB).
- SHA-256: `b94bdaab...` and `6eddcbed...`, exact match to local 3B baseline.
- **Not** the final 8B model.

## Classification candidates

| File | Size | Source |
|---|---:|---|
| `classification/jiawei_lvlm_checkpoint_2699_epoch1_f1_0.2706.pt` | 1.6 GB | jiawei class_model/lvlm ckpt |
| `classification/yunhan_lvlm_checkpoint_2699_epoch2_f1_0.3047.pt` | 1.6 GB | yunhan VLM/lvlm ckpt |
| `classification/yunhan_ssl3d_best_f1_0.2578_epoch31.pth` | 391 MB | yunhan SSL3D run/logger |

Exact final submitted checkpoint: **TODO — confirm with team**.

## Weight mount convention

For a public release, arrange each final model in its own Hugging Face repo and
document a mount path, e.g.:

```text
/weights/model/config.json
/weights/model/model-0000x-of-0000y.safetensors
```

Do not commit weights to GitHub. Do not redistribute CT-RATE images/reports.

## License

- Code in this repository: Apache-2.0 (see `LICENSE`).
- Final report-generation weights: **CC BY-NC 4.0 (non-commercial use only)**.
- Final CT_CLS8 weights and TotalSegmentator bundle: **CC BY-NC 4.0** for
  team-provided weights; upstream components retain their own terms.
- Base models: Qwen3-8B Apache-2.0; SigLIP2 upstream license; MedM-VL Apache-2.0.
- CT-RATE data remains under its upstream terms.
