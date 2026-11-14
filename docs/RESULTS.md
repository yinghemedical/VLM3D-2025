# Results

## Official results

The official VLM3D 2025 leaderboard is the source of truth. This workspace does
not contain a platform submission ID or official score snapshot.

## Local evaluation — Report Generation

From `VLM3D_Methods/eval_scripts/eval/eval_clshead/leaderboard.json`:

| Metric | Value |
|---|---:|
| BLEU-1 | 0.3836 |
| BLEU-2 | 0.2526 |
| BLEU-3 | 0.1685 |
| BLEU-4 | 0.1152 |
| METEOR | 0.2428 |
| ROUGE-L | 0.2613 |
| CIDEr | 0.0229 |

## 2026-11-13 local CT-RATE re-evaluation

Pipeline: `scripts/report_generation/`; model:
`final-MedM-VL-siglip-384-qwen3-8b-finetune`; split:
CT-RATE `valid_fixed` (3,039 records).

Full run (all 3,039 CT-RATE `valid_fixed` records; completed 2026-11-13):

| Metric | Value |
|---|---:|
| BLEU-1 | 0.4450 |
| BLEU-2 | 0.3427 |
| BLEU-3 | 0.2793 |
| BLEU-4 | 0.2383 |
| BLEU-mean | 0.3263 |
| ROUGE-L | 0.3632 |
| CIDEr | 0.3033 |
| METEOR | not computed (no Java) |

**Report-derived label F1** (AutoExps/RadBERT, 18 classes, threshold 0.5,
same 3,039 cases):

| Metric | Value |
|---|---:|
| Macro F1 | 0.9083 |
| Micro F1 | 0.9332 |
| Macro precision | 0.9950 |
| Macro recall | 0.8501 |
| Macro AUC | 0.9644 |
| CRG | 1.0000 |

**Important caveat:** the final 8B training file
`final_finetune.json` was built from **both** the CT-RATE train and valid
splits. These 3,039 valid cases are therefore seen during fine-tuning; the
numbers are an in-sample upper bound, not a held-out test score. For a clean
held-out number, evaluate on the official hidden CT-RATE test set or another
external set not used in training. The 0.9083 F1 is report-derived (RadBERT)
and is **not** the same metric as U-VLM's classification F1 0.414.

### Classification-output prompt run (deployment-like)

Prompt labels were replaced with the 18-class **U-VLM classification model
output** on the same 3,039 cases, instead of the CT-RATE label CSV:

```text
/path/to/shipengcheng/code/CT-RATE_eval_scripts/eval_cls_csv/
  nnXNetTrainer_ResEncoderUVL_v6_with_reportgen_only_cls_balanced_nnUNetAug_w_seg_pretrained_256_384_384_Plans_epoch410.csv
```

The classification output itself scores macro F1 0.5593 / micro F1 on these
3,039 cases. Report generation was rerun with those predicted labels.

| Metric | Value |
|---|---:|
| BLEU-1 | 0.4137 |
| BLEU-2 | 0.3153 |
| BLEU-3 | 0.2558 |
| BLEU-4 | 0.2177 |
| BLEU-mean | 0.3006 |
| ROUGE-L | 0.3457 |
| CIDEr | 0.2088 |
| Macro F1 (RadBERT) | 0.5056 |
| Micro F1 (RadBERT) | 0.5580 |
| Macro precision | 0.6250 |
| Macro recall | 0.4488 |
| Macro AUC | 0.7852 |
| CRG | 0.5567 |
| Samples | 3039 |

Outputs: `runs/report_generation/ctrate_valid_8b/full_predcls/`.
The GT file for evaluation still uses the official CT-RATE labels; only the
prompt labels come from the classification model.

### CT_CLS8-output prompt run

Prompt labels were replaced with the **CT_CLS8** ensemble output on the same
3,039 cases (`runs/classification/ctrate_valid_ct_cls8/full/ct_cls8_predictions.csv`).
CT_CLS8 itself scores macro F1 **0.6129** / micro F1 **0.6363** on these cases.

| Metric | Value |
|---|---:|
| BLEU-1 | 0.4159 |
| BLEU-2 | 0.3154 |
| BLEU-3 | 0.2539 |
| BLEU-4 | 0.2146 |
| BLEU-mean | 0.3000 |
| ROUGE-L | 0.3424 |
| CIDEr | 0.1975 |
| Macro F1 (RadBERT) | 0.5494 |
| Micro F1 (RadBERT) | 0.5963 |
| Macro precision | 0.5831 |
| Macro recall | 0.5427 |
| Macro AUC | 0.7948 |
| CRG | 0.5431 |
| Samples | 3039 |

Outputs: `runs/report_generation/ctrate_valid_8b/full_predcls_ctcls8/`.
The cluster job itself exited 127 after inference because of a conda
activation issue; the predictions were complete and post-processing was rerun
with the absolute `medmvl` interpreter. See `pipeline/README.md`.

Smoke run (first 8 cases, not representative):

| Metric | Value |
|---|---:|
| BLEU-1 | 0.4655 |
| BLEU-2 | 0.3773 |
| BLEU-3 | 0.3219 |
| BLEU-4 | 0.2849 |
| BLEU-mean | 0.3624 |
| ROUGE-L | 0.3878 |
| CIDEr | 0.0008 |

## Local evaluation — Classification

From the same local `leaderboard.json`:

| Metric | Value |
|---|---:|
| Macro F1 | 0.4948 |
| Accuracy | 0.7711 |
| Macro precision | 0.4555 |
| Macro recall | 0.5532 |
| CRG | 0.4662 |

## Caveat

These are **local** evaluation values tied to a local checkpoint/eval snapshot.
They are not official challenge scores and should not be presented as such.
For the survey, use the official leaderboard value if available; otherwise label
the number as local and include the evaluator/checkpoint identity.
