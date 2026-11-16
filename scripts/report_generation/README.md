# CT-RATE Report-Generation Evaluation

This directory contains the local evaluation pipeline for the VLM3D 2025 8B
report model. It uses the CT-RATE `valid_fixed` split (3,039 report-bearing
volumes) as the local test set and follows the CT-RATE report contract:

```text
series_id, report, generated_report
```

MedM-VL uses its own inference code. We use the CT-RATE CSV layout and a
pycocoevalcap-based report metric.

## Files

| File | Purpose |
|---|---|
| `prepare_runtime_model.py` | Creates `weights/report_generation/runtime-8b-ctrate/` with symlinked 8B shards and a config patched to local SigLIP2/Qwen3 assets. |
| `prepare_ctrate_eval.py` | Builds `gt.csv`, `inputs.all.json`, and `inputs.shardNN.json` from `validation_reports.csv` + `valid_predicted_labels.csv`. |
| `run_medmvl_report.py` | One process per shard/GPU; writes `predictions.shardNN.{json,csv}` incrementally. |
| `merge_results.py` | Merges shards in GT order into `predictions.{json,csv}`. |
| `eval_nlg.py` | Calls the CT-RATE BLEU metric and adds ROUGE-L/CIDEr (METEOR if Java exists). |
| `eval_f1.py` | RadBERT report-label extraction + 18-class macro/micro F1 + CRG. |
| `run_ctrate_eval_8b.sh` | End-to-end smoke/full launcher used by the cluster YAMLs. |
| `run_f1_eval.sh` | Runs `eval_f1.py` on an existing full run. |

## Data and model

- Images: `/path/to/Public_data/MM_data/CT-RATE/dataset/valid_fixed/`
- Reports: `/path/to/Public_data/MM_data/CT-RATE_blosc2/radiology_text_reports/validation_reports.csv`
- Labels for prompt: `/path/to/Public_data/MM_data/CT-RATE_blosc2/multi_abnormality_labels/valid_predicted_labels.csv`
- Model: `weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune/`
- Local assets: SigLIP2 snapshot and Qwen3-8B under
  `/path/to/luyan_mllm/VLM3D_challenge/jiawei_workspace/cache_dir_hf/`

Prompt (matching the final fine-tune data):

```text
<image3d>
Generate radiology report for the CT.This patient may have the following diseases: ...
This patient does not have the following diseases: ...
```

## Run

Smoke (8 cases, 1 GPU):

```bash
bash /path/to/VLM3D/scripts/report_generation/run_ctrate_eval_8b.sh smoke
```

Full (3,039 volumes, 8 shards/GPUs), GT-label prompt:

```bash
bash /path/to/VLM3D/scripts/report_generation/run_ctrate_eval_8b.sh full
```

Full with **classification-model output** as prompt labels:

```bash
bash /path/to/VLM3D/scripts/report_generation/run_ctrate_eval_8b.sh full_predcls
```

The default classifier output is the CT-RATE classification CSV in
`/path/to/shipengcheng/code/CT-RATE_eval_scripts/eval_cls_csv/`; override
with `PROMPT_LABELS=/path/to/classification.csv`. GT labels in `gt.csv` remain
the official CT-RATE labels; only the prompt is classification-conditioned.
The `full_predcls` mode also runs `run_f1_eval.sh` after report generation.

Cluster submission:

```bash
batch_cmd run -i /path/to/VLM3D/configs/report_generation/ctrate_report_8b_smoke.yaml -y
batch_cmd run -i /path/to/VLM3D/configs/report_generation/ctrate_report_8b_full.yaml -y
```

F1 / CRG after a full run (RadBERT, same 3,039 volumes):

```bash
batch_cmd run -i /path/to/VLM3D/configs/report_generation/ctrate_report_8b_f1.yaml -y
```

This reads `predictions.csv` + `gt.csv`, writes `output_cls.csv`,
`metrics_cls.json`, and `metrics_crg.json` into the full run directory. The
RadBERT base-model warning about `pooler.*` is expected during base RoBERTa
initialization; `RadBertClassifier.pth` contains those pooler weights and they
are loaded by the downstream classifier.

## Important implementation notes

- `medals_local_test` is used for inference because it has Transformers 4.51
  (Qwen3 support), torch, and MONAI. The runtime config forces `sdpa` because
  that env does not ship `flash_attn`.
- `medmvl` is used for NLG evaluation because it has `pycocoevalcap`.
- The copied MedM-VL code has two local patches for offline loading:
  - `siglip2.py`: uses `AutoImageProcessor` directly and `local_files_only=True`
    (the cached SigLIP2 tokenizer files are incomplete and irrelevant to the
    vision path).
  - `qwen.py`: all model/tokenizer/config loads use `local_files_only=True`.
- `run_medmvl_report.py` records failed cases as `success=False` instead of
  silently dropping them, preserving row alignment with `gt.csv`.
