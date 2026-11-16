# Classification Configs

Final submitted classifier: **CT_CLS8**, a three-checkpoint ensemble
(`epoch=263`, `epoch=371`, `epoch=92`; sigmoid averaged).

## CT-RATE CT_CLS8 Inference

| YAML | Purpose | A100s |
|---|---|---:|
| `ctrate_ct_cls8_smoke.yaml` | one-case smoke test | 1 |
| `ctrate_ct_cls8_full.yaml` | 3,039-volume inference (4 shards) | 4 |

Both invoke `scripts/classification/run_ct_cls8_all.sh` and write to
`runs/classification/ctrate_valid_ct_cls8/<mode>/`.

```bash
batch_cmd run -i configs/classification/ctrate_ct_cls8_smoke.yaml -y
batch_cmd run -i configs/classification/ctrate_ct_cls8_full.yaml -y
```

The resulting `ct_cls8_predictions.csv` is used by
`run_ctrate_eval_8b.sh full_predcls_ctcls8` to condition the report prompt on
the classification model output.

See `docs/MODEL_WEIGHTS.md` for the final checkpoint files and
`docs/SUBMISSION.md` for the submission mapping.
