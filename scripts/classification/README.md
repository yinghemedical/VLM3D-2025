# CT_CLS8 classification inference

This directory runs the **actual CT_CLS8 ensemble** (not U-VLM classification)
on the 3,039 CT-RATE `valid_fixed` volumes, producing 18-class probabilities
for the report prompt.

## Files

| File | Purpose |
|---|---|
| `prepare_ct_cls8_shards.py` | Build flat shard dirs of NIfTI symlinks from `gt.csv`. |
| `run_ct_cls8_shard.py` | Run the original CT_CLS8 `process.py` on one shard; patches only input/output/checkpoint globals. |
| `process_ct_cls8.py` | Copy of the original process with Hydra 1.0/OmegaConf 2.0 compatibility fixes. |
| `parsing_utils_ct_cls8.py` | OmegaConf 2.0-compatible resolver registration. |
| `merge_ct_cls8_predictions.py` | Merge shard JSONs into `ct_cls8_predictions.csv`. |
| `run_ct_cls8_all.sh` | Smoke/full launcher (4 shards by default). |

## Why the wrapper

The original code is used unchanged from:

```text
/path/to/fangyunhan/docker5/
  process.py
  ckpts/                     # unreadable by current account
  nnUNet_results/            # readable
```

The wrapper runs from that directory so `nnUNet_results` resolves, then points
the three classifier checkpoints at the readable extracted copies in:

```text
/path/to/VLM3D/weights/classification/docker_ct_cls8/
```

The `process.py` copy only removes Hydra `version_base`, replaces
`register_new_resolver` with OmegaConf 2.0 `register_resolver`, converts a
`PosixPath` to `str`, and instantiates `data.module` rather than `data`.

## Run

```bash
bash /path/to/VLM3D/scripts/classification/run_ct_cls8_all.sh smoke
bash /path/to/VLM3D/scripts/classification/run_ct_cls8_all.sh full
```

Cluster YAMLs:

```bash
batch_cmd run -i /path/to/VLM3D/configs/classification/ctrate_ct_cls8_smoke.yaml -y
batch_cmd run -i /path/to/VLM3D/configs/classification/ctrate_ct_cls8_full.yaml -y
```

Output:

```text
runs/classification/ctrate_valid_ct_cls8/full/
  predictions.shard00.json ... predictions.shard03.json
  ct_cls8_predictions.csv        # series_id + 18 probabilities
```

The report pipeline can then use:

```bash
bash .../run_ctrate_eval_8b.sh full_predcls_ctcls8
```

which reads `ct_cls8_predictions.csv`, reruns the 8B report model on all 3,039
cases, and computes BLEU/F1.
