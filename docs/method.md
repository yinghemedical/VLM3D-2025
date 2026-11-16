# Method

This repository covers the two VLM3D@MICCAI 2025 tracks submitted by
**Medical Image Insights**: Multi-Abnormality Classification and Report
Generation. Both tracks use CT-RATE chest CT.

## Multi-Abnormality Classification

The classification model is **CT_CLS8**, a three-checkpoint ROI ensemble.

1. **Anatomy segmentation.** TotalSegmentator v2.0.1
   (`Dataset665_Totalsegmentator_v201_Anatomy`) segments the full CT volume.
2. **ROI crops.** Seven anatomical regions are extracted: material, arterial,
   heart, lung, trachea, hiatal, and lymphadenopathy.
3. **ROI classifiers.** ROI-specific ResEncUNet classifiers predict the 18
   CT-RATE abnormality labels.
4. **Class masking and ensemble.** Each checkpoint's sigmoid outputs are
   restricted to the classes assigned to its ROI; the three masked probability
   maps are averaged.
5. **Output.** 18 class probabilities. The 0.5 threshold is used to build the
   report prompt and for evaluation.

The three checkpoints are combined only at the probability level. No language
model or report text is used in this track.

## Report Generation

The report model is **MedM-VL**:

- Image encoder: **SigLIP2-so400m-patch14-384**
- Language model: **Qwen3-8B**
- Connector: two-layer MLP with attention pooling

Training used 50,179 CT-RATE volume-report records:

1. **Pretrain.** Image encoder and LLM frozen; connector trained for one epoch.
2. **Fine-tune.** Image encoder, connector, and LLM trained jointly for one
   epoch.

At inference, report generation **depends on the classification model**:

1. Run CT_CLS8 on the CT volume.
2. Convert the 18 probabilities into positive and negative finding lists at a
   0.5 threshold.
3. Insert the lists into the prompt together with the CT volume and
   instruction.
4. Generate `Findings: ... Impressions: ...` with the 8B model.

Ground-truth labels are not used at inference. The report track uses a single
8B checkpoint and no ROI ensemble.

## Data and weights

CT-RATE is not redistributed. See `docs/DATA.md` for data access and
preprocessing, `docs/MODEL_WEIGHTS.md` for final weights, and
`docs/SUBMISSION.md` for the submission-to-config mapping.
