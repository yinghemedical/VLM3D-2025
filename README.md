# VLM3D 2025 Code

Two tracks, one repository. Training and inference live in separate
directories; shared orchestration is under `pipeline/`.

## Layout

```text
code/classification/
  train/          SSL3D / CT_CLS8 training code and configs
  inference/      CT_CLS8 ROI + 3-checkpoint ensemble inference

code/report_generation/
  MedM-VL/        shared MedM-VL model/dataset code
  train/          report pretrain/finetune scripts, data builders
  inference/      report inference wrappers
  eval/           NLG + RadBERT F1/CRG evaluation

pipeline/         end-to-end CT_CLS8 -> 8B report, per case or batch
scripts/          wrappers used by pipeline and cluster YAMLs
configs/          classification/report cluster YAMLs
docs/             method, results, release notes
pipeline/examples/ placeholder examples; no CT-RATE data
```

## Entrypoints

Classification:

```text
code/classification/train/main.py
code/classification/inference/process.py
scripts/classification/run_ct_cls8_all.sh
```

Report generation:

```text
code/report_generation/train/MedM-VL/lvlm/train.py
code/report_generation/train/scripts/former/final_siglip_384_qwen3_4b_*.sh
code/report_generation/inference/run_medmvl_report.py
pipeline/run_case.sh
```

## Data policy

No CT-RATE images, reports, patient identifiers, or label CSVs are included.
`pipeline/examples/cases.example.csv` uses `/path/to/...` placeholders.

## Setup

Weights are in the separate repository `VLM3D-2025-weights`:

```text
classification/ct_cls8/                 # 3 CT_CLS8 checkpoints
classification/total_segmentator/       # Dataset665 checkpoint
report_generation/final-8b/             # MedM-VL 8B, 4 shards
```

Copy the environment template and set local paths:

```bash
cp pipeline/configs/pipeline.env.example pipeline/configs/pipeline.env
# edit VLM3D_ROOT, WEIGHTS_ROOT, CT_CLS_ROOT, CACHE_DIR, MEDMVL_ROOT
```

For CT_CLS8 inference, `CT_CLS_ROOT` must contain `process.py` and a local
`nnUNet_results` symlink pointing at the Dataset665 weights.

## Quick start

One case:

```bash
bash pipeline/run_case.sh /path/to/case.nii.gz CASE_ID pipeline/outputs
# -> pipeline/outputs/CASE_ID/case.json
```

A few examples:

```bash
bash pipeline/run_examples.sh
```

Full CT-RATE `valid_fixed` batch (3,039 cases):

```bash
bash pipeline/run_3039.sh
```

RadBERT/RoBERTa are public and are not included. Point the evaluation script to
the upstream RadBERT checkpoint for F1/CRG reproduction.
