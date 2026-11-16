# Data

## CT-RATE

Both tracks use the CT-RATE chest CT dataset:

- Dataset: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
- Paper: Hamamci et al., *Generalist foundation models from a multimodal
  dataset for 3D computed tomography*, Nature Biomedical Engineering, 2026.

CT-RATE images, reports, patient identifiers, and label CSVs are not
redistributed in this repository. Download the dataset under its upstream
terms.

## Expected Layout

The evaluation scripts take explicit paths for images, reports, and labels.
A typical local layout is:

```text
${CTRATE_ROOT}/
  dataset/valid_fixed/
    valid_<patient>/valid_<patient>_<study>/<series>.nii.gz
  radiology_text_reports/validation_reports.csv
  multi_abnormality_labels/valid_predicted_labels.csv
```

`validation_reports.csv` must contain `VolumeName`, `Findings_EN`, and
`Impressions_EN`. `valid_predicted_labels.csv` must contain `VolumeName` and
the 18 abnormality label columns. The column order is defined in
`scripts/report_generation/prepare_ctrate_eval.py`.

## Split

- Evaluation: CT-RATE `valid_fixed`, 3,039 CT volumes from 1,304 patients.
- Report training: 50,179 volume-report records. The final fine-tuning file
  includes the CT-RATE train and valid splits; report metrics on `valid_fixed`
  are therefore in-sample.
- Classification training: CT-RATE chest CT volumes with the 18
  multi-abnormality labels.

## Preprocessing

The public per-case pipeline accepts `.nii.gz`, `.nii`, and `.mha` CT volumes.
Model-specific code handles image loading and normalization.

### CT_CLS8

1. TotalSegmentator v2.0.1 segments the full CT volume.
2. Seven ROI crops are extracted: material, arterial, heart, lung, trachea,
   hiatal, and lymphadenopathy.
3. ROI-specific ResEncUNet classifiers predict the 18 labels.
4. Class masks restrict each ROI to its assigned findings.
5. Three checkpoints are averaged at the sigmoid level; threshold 0.5.

### Report Generation

The report prompt is built from CT_CLS8 predictions, not ground-truth labels:

```text
<image3d>
Generate radiology report for the CT.
This patient may have the following diseases: {positive findings}
This patient does not have the following diseases: {negative findings}
```

The model then generates `Findings: ... Impressions: ...` in one response.

## Configuration

Copy the environment template and set the local paths:

```bash
cp pipeline/configs/pipeline.env.example pipeline/configs/pipeline.env
```

The template uses placeholders only. Do not commit local paths, patient
identifiers, or CT-RATE data.
