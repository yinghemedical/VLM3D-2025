# Limits and Limitations

## Multi-Abnormality Classification

- The model covers only the 18 CT-RATE abnormality labels; findings outside
  this set are invisible.
- It depends on TotalSegmentator segmentation and fixed ROI-to-class masks.
  Segmentation errors, unusual anatomy, and small lesions can propagate to the
  final probabilities.
- The 0.5 threshold is not calibrated per finding. Class imbalance and
  report-derived label noise make performance uneven across findings.
- Evaluation is limited to CT-RATE `valid_fixed`. External cohorts, other
  scanners or acquisition protocols, non-chest anatomy, and the official hidden
  test set were not tested.
- F1 measures label agreement, not clinical utility.

## Report Generation

- Report generation depends on the classification model. Missed findings are
  usually omitted, false positives can be repeated, and the 8B model can still
  generate findings unsupported by the image.
- Training used CT-RATE reports, so the model may reproduce their style,
  terminology, and prevalence biases.
- CT-RATE `valid` was included in fine-tuning. The reported report-generation
  metrics are in-sample, not a held-out estimate.
- We did not test external cohorts, other scanners or acquisition protocols,
  non-English reports, alternative prompt formats, per-finding thresholds, or
  the official hidden test set.
- BLEU, ROUGE, CIDEr, and RadBERT-based F1 measure text or label overlap, not
  clinical correctness. The system has not been evaluated in a human reader
  study or clinical workflow.
