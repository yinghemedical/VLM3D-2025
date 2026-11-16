# Model Weights

Final weights are published in two Hugging Face repositories. Weights are not
stored in this GitHub repository.

## Report Generation

- Hugging Face:
  https://huggingface.co/yinghemedical/VLM3D2025-Task1-ReportGeneration
- Model: MedM-VL 8B (SigLIP2-so400m-patch14-384 + Qwen3-8B)
- Size: about 17 GB
- Files:
  - `model-00001-of-00004.safetensors`
  - `model-00002-of-00004.safetensors`
  - `model-00003-of-00004.safetensors`
  - `model-00004-of-00004.safetensors`
  - `config.json`
  - `model.safetensors.index.json`
  - `tokenizer_config.json`
  - `added_tokens.json`
  - `special_tokens_map.json`
  - `vocab.json`
  - `merges.txt`
  - `chat_template.jinja`
- `SHA256SUMS` is included in the repository.

## Multi-Abnormality Classification

- Hugging Face:
  https://huggingface.co/yinghemedical/VLM3D2025-Task2-AbnormalityClassifications
- Model: CT_CLS8 three-checkpoint ensemble with TotalSegmentator v2.0.1
- Files:
  - `ct_cls8/epoch=263-crop_resume-val_F1=0.00.ckpt`
  - `ct_cls8/epoch=371-crop_cls_head-val_F1=0.00.ckpt`
  - `ct_cls8/epoch=92-crop_cls_head-f-val_F1=0.00.ckpt`
  - `total_segmentator/fold_0/checkpoint_final.pth`
  - `total_segmentator/plans.json`
  - `total_segmentator/dataset.json`
  - `total_segmentator/dataset_fingerprint.json`
- `SHA256SUMS` is included in the repository.

## Base Models

- SigLIP2: `google/siglip2-so400m-patch14-384`
- Qwen3: `Qwen/Qwen3-8B`
- TotalSegmentator: v2.0.1, `Dataset665_Totalsegmentator_v201_Anatomy`
- RadBERT: used for report-label evaluation; weights are not included

See `docs/REFERENCES.md` for upstream links and terms.

## Usage Terms

- Code in this repository: Apache-2.0.
- Final report-generation weights: CC BY-NC 4.0.
- Final CT_CLS8 weights and TotalSegmentator bundle: CC BY-NC 4.0 for
  team-provided weights; upstream components retain their own terms.
- CT-RATE images and reports are not redistributed.
