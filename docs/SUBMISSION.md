# Submission Artifacts

The report-generation submission consumes CT_CLS8 predictions, so the
classification step runs first.

| Track | Submitted Model | Config | Weights | Command |
|---|---|---|---|---|
| Multi-Abnormality Classification | CT_CLS8 three-checkpoint ROI ensemble | `configs/classification/ctrate_ct_cls8_full.yaml` | [Task 2 weights](https://huggingface.co/yinghemedical/VLM3D2025-Task2-AbnormalityClassifications): three `ct_cls8/*.ckpt` files + TotalSegmentator checkpoint | `bash scripts/classification/run_ct_cls8_all.sh full` |
| Report Generation | MedM-VL 8B (`SigLIP2-so400m-patch14-384` + `Qwen3-8B`), single checkpoint | `configs/report_generation/ctrate_report_8b_ctcls8.yaml` | [Task 1 weights](https://huggingface.co/yinghemedical/VLM3D2025-Task1-ReportGeneration): four safetensors shards + config/tokenizer | `bash scripts/report_generation/run_ctrate_eval_8b.sh full_predcls_ctcls8` |

The report command reads the classification output from the default run
directory. See `configs/report_generation/ctrate_report_8b_ctcls8.yaml` and
`scripts/report_generation/README.md` for path overrides.

## Per-Case Pipeline

For a single CT volume, run classification and report generation together:

```bash
bash pipeline/run_case.sh /path/to/case.nii.gz CASE_ID pipeline/outputs
```

The final result is `pipeline/outputs/CASE_ID/case.json`, containing the
18-class probabilities, prompt, and generated report.

## Final Weight Files

See `docs/MODEL_WEIGHTS.md` for the complete file lists and checksums. Weights
are hosted on Hugging Face and are not committed to GitHub.
