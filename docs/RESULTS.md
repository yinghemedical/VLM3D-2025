# Results

Team: **Medical Image Insights**

## Official Results

| Track | Overall rank | Leaderboard |
|---|---:|---|
| Report Generation | **#1** | https://research.forithmus.com/challenges/ct-report-generation#leaderboard |
| Multi-Abnormality Classification | **#3** | https://research.forithmus.com/challenges/ct-abnormality-classification#leaderboard |

The official leaderboard is the source of truth for challenge scores. The
numbers below are local evaluation results, not official challenge scores.

## Evaluation Split

Both tracks were evaluated on CT-RATE `valid_fixed`: 3,039 CT volumes from
1,304 patients. The report-generation metrics use prompts built from CT_CLS8
predictions.

The report fine-tuning data included the CT-RATE train and valid splits, so
the report-generation numbers are in-sample and do not estimate held-out
performance.

## Report Generation

Model: final MedM-VL 8B checkpoint (`SigLIP2-so400m-patch14-384` +
`Qwen3-8B`), conditioned on CT_CLS8 predictions.

| Metric | Value |
|---|---:|
| BLEU-1 | 0.4159 |
| BLEU-2 | 0.3154 |
| BLEU-3 | 0.2539 |
| BLEU-4 | 0.2146 |
| BLEU-mean | 0.3000 |
| ROUGE-L | 0.3424 |
| CIDEr | 0.1975 |
| RadBERT Macro F1 | 0.5494 |
| RadBERT Micro F1 | 0.5963 |
| Macro precision | 0.5831 |
| Macro recall | 0.5427 |
| Macro AUC | 0.7948 |
| CRG | 0.5431 |
| Samples | 3,039 |

## Multi-Abnormality Classification

Model: final CT_CLS8 three-checkpoint ROI ensemble.

| Metric | Value |
|---|---:|
| Macro F1 | 0.6129 |
| Micro F1 | 0.6363 |
| Macro precision | 0.5774 |
| Macro recall | 0.6578 |
| Samples | 3,039 |

## Evaluation Code

- Report generation: `scripts/report_generation/`
- Classification: `scripts/classification/`
- End-to-end pipeline: `pipeline/`
