# VLM3D 2025

Team: **Medical Image Insights**

This repository contains code for the VLM3D@MICCAI 2025 multi-abnormality
classification and report generation tracks. Classification code is in
`code/classification/`, report generation code in `code/report_generation/`,
and shared code in `pipeline/`.

## Team Members

**Report Generation**  
Jiawei Chen², Yiming Shi³, Yunhan Fang⁴, Pengcheng Shi¹, Shuheng Zhang¹, Jiaqi Liu¹, Xinglin Zhang¹  
¹ Medical Image Insights · ² Xi’an Jiaotong University · ³ Tsinghua University · ⁴ Shanghai Jiaotong University

**Multi-Abnormality Classification**  
Yunhan Fang², Pengcheng Shi¹, Jiawei Chen³, Yiming Shi⁴, Shuheng Zhang¹, Jiaqi Liu¹, Xinglin Zhang¹  
¹ Medical Image Insights · ² Shanghai Jiaotong University · ³ Xi’an Jiaotong University · ⁴ Tsinghua University

## Official results

| Track | Overall rank | Leaderboard |
|---|---:|---|
| Report Generation | **#1** | https://research.forithmus.com/challenges/ct-report-generation#leaderboard |
| Multi-Abnormality Classification | **#3** | https://research.forithmus.com/challenges/ct-abnormality-classification#leaderboard |

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

## Setup

Weights are in two Hugging Face repositories:

- Report generation:
  [`yinghemedical/VLM3D2025-Task1-ReportGeneration`](https://huggingface.co/yinghemedical/VLM3D2025-Task1-ReportGeneration)
  — final 8B MedM-VL checkpoint (4 shards).
- Multi-abnormality classification:
  [`yinghemedical/VLM3D2025-Task2-AbnormalityClassifications`](https://huggingface.co/yinghemedical/VLM3D2025-Task2-AbnormalityClassifications)
  — CT_CLS8 3-checkpoint ensemble + TotalSegmentator checkpoint.

Set local paths in the environment file:

```bash
cp pipeline/configs/pipeline.env.example pipeline/configs/pipeline.env
# edit VLM3D_ROOT, WEIGHTS_ROOT, CT_CLS_ROOT, CACHE_DIR, MEDMVL_ROOT
```

For CT_CLS8 inference, `CT_CLS_ROOT` must contain `process.py` and a local
`nnUNet_results` symlink pointing at the Dataset665 weights.

## Quick start

Run one case:

```bash
bash pipeline/run_case.sh /path/to/case.nii.gz CASE_ID pipeline/outputs
# -> pipeline/outputs/CASE_ID/case.json
```

Run the example cases:

```bash
bash pipeline/run_examples.sh
```

Run the full CT-RATE `valid_fixed` set (3,039 volumes):

```bash
bash pipeline/run_3039.sh
```

RadBERT/RoBERTa are public and are not included. Point the evaluation script to
the upstream RadBERT checkpoint for F1/CRG reproduction.

## Data and citation

Training and evaluation use the
[CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

If you use the CT-RATE dataset, please cite:

```bibtex
@article{hamamci2026generalist,
  title={Generalist foundation models from a multimodal dataset for 3D computed tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Dogan, Irem and Durugol, Omer Faruk and Hou, Benjamin and Shit, Suprosanna and others},
  journal={Nature Biomedical Engineering},
  pages={1--19},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

## License

Code is Apache-2.0. Released model weights are
**CC BY-NC 4.0 (non-commercial use only)**. Upstream base models, datasets, and
dependencies retain their own licenses.
