# Report-generation configs

`original_final_pretrain.sh`, `original_final_finetune.sh` and
`original_inference.sh` are copied from the 2025 Jiawei workspace for provenance.

They contain old absolute paths and must be sanitized before release:

- `/path/to/jiawei_workspace/...` -> public repo root
- `image3d_path /path/to/jiawei_database/ctrate_raw` -> local CT-RATE preprocessed path
- `MODEL_PATH` in inference -> `weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune`

Hyperparameters are summarized in `docs/REPRODUCTION.md`.

## CT-RATE local evaluation

| YAML | Purpose | GPUs |
|---|---|---:|
| `ctrate_report_8b_smoke.yaml` | 8-case end-to-end smoke test | 1 |
| `ctrate_report_8b_full.yaml` | full CT-RATE `valid_fixed` evaluation with GT-label prompt (3,039 records) | 8 |
| `ctrate_report_8b_predcls.yaml` | full evaluation with classification-model output as prompt labels | 8 |
| `ctrate_report_8b_f1.yaml` | RadBERT F1 + CRG on an existing run | 1 |

Both invoke `scripts/report_generation/run_ctrate_eval_8b.sh` (`smoke` or
`full`), write to `runs/report_generation/ctrate_valid_8b/<mode>/`, then merge
predictions and compute `metrics_nlg.json`.

```bash
batch_cmd run -i configs/report_generation/ctrate_report_8b_smoke.yaml -y
batch_cmd run -i configs/report_generation/ctrate_report_8b_full.yaml -y
```
