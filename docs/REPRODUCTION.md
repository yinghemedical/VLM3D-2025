# Reproduction Notes

## Environment

- Python 3.10, PyTorch 2.2, Transformers 4.51, MONAI 1.4, nnU-Net 2.4.1.
- Install with `pip install -r requirements.txt` or
  `conda env create -f environment.yml`.
- DeepSpeed ZeRO-3 for report training.
- Base models: SigLIP2-so400m-patch14-384 and Qwen3-8B. Download them from
  their upstream releases; see `docs/REFERENCES.md`.
- Final weights: `docs/MODEL_WEIGHTS.md`.
- Data access and layout: `docs/DATA.md`.

## Report-Generation Training

Training used 50,179 CT-RATE volume-report records.

1. **Pretrain.** Image encoder and LLM frozen; connector trained for one epoch.
   Per-device batch 8, gradient accumulation 2, learning rate 1e-3.
2. **Fine-tune.** Resume from the pretrain output; image encoder, connector, and
   LLM trained jointly for one epoch. Per-device batch 8, gradient
   accumulation 2, learning rate 2e-5.

The final fine-tuning file includes the CT-RATE train and valid splits, so
report metrics on `valid_fixed` are in-sample. Training manifests are generated
from CT-RATE; images and reports are not redistributed.

## Inference

The report model depends on the classification model:

1. Run CT_CLS8 to obtain 18-class probabilities.
2. Build the positive/negative finding lists at threshold 0.5.
3. Run MedM-VL 8B with the CT volume and prompt.

Per-case:

```bash
bash pipeline/run_case.sh /path/to/case.nii.gz CASE_ID pipeline/outputs
```

Batch:

```bash
bash pipeline/run_3039.sh
```

Cluster configs and run order are in `docs/SUBMISSION.md`.

## Evaluation

```bash
bash scripts/classification/run_ct_cls8_all.sh full
bash scripts/report_generation/run_ctrate_eval_8b.sh full_predcls_ctcls8
```

The report command reads the CT_CLS8 predictions and computes NLG and
RadBERT-based F1 metrics. See `scripts/report_generation/README.md` for path
overrides.

## Weights

- Report generation:
  https://huggingface.co/yinghemedical/VLM3D2025-Task1-ReportGeneration
- Classification:
  https://huggingface.co/yinghemedical/VLM3D2025-Task2-AbnormalityClassifications

Code repository: https://github.com/yinghemedical/VLM3D-2025
