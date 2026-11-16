# Report-Generation Configs

Final submitted model: **MedM-VL 8B** (`SigLIP2-so400m-patch14-384` +
`Qwen3-8B`). The original pretrain and fine-tune scripts are kept for
provenance; update local paths before use.

## CT-RATE Evaluation

| YAML | Purpose | Devices |
|---|---|---:|
| `ctrate_report_8b_smoke.yaml` | 8-case end-to-end smoke test | 1 |
| `ctrate_report_8b_full.yaml` | full CT-RATE `valid_fixed` evaluation with GT-label prompt (3,039 volumes) | 8 |
| `ctrate_report_8b_predcls.yaml` | full evaluation with classification-model output as prompt labels | 8 |
| `ctrate_report_8b_ctcls8.yaml` | final evaluation with CT_CLS8 output as prompt labels | 4 |
| `ctrate_report_8b_f1.yaml` | RadBERT F1 + CRG on an existing run | 1 |

The scripts invoke `scripts/report_generation/run_ctrate_eval_8b.sh` and write
to `runs/report_generation/ctrate_valid_8b/<mode>/`.

```bash
batch_cmd run -i configs/report_generation/ctrate_report_8b_smoke.yaml -y
batch_cmd run -i configs/report_generation/ctrate_report_8b_ctcls8.yaml -y
```

The final report run depends on the CT_CLS8 predictions from the classification
config. See `docs/SUBMISSION.md` for the full run order.
