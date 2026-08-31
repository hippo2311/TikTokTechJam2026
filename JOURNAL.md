# Reelistic Experiment and Engineering Journal

This journal records why the current system looks the way it does: the hypotheses we formed, alternatives we considered, what we implemented, what evidence still needs to be inserted, and which claims must remain separate. It is intentionally more candid than the judge-facing summary in [README.md](README.md).

## 1. Problem framing

The target is not simply clean-image classification. Images on social platforms are commonly recompressed, resized, cropped, blurred, color-adjusted, or contaminated by noise before a detector sees them. A useful detector must therefore satisfy three goals at once:

1. retain sensitivity to synthetic-image traces;
2. avoid relying only on generator- or dataset-specific shortcuts;
3. operate as part of a reviewable product rather than as an isolated notebook.

The competition also emphasizes low-false-positive operation. We therefore select checkpoints using TPR at 1% FPR rather than accuracy alone and report both ranking and operating-point metrics.

## 2. Data decisions

### Development sources

The current DINOv3 configuration names three development sources:

- SID;
- CIFAKE;
- approved WildFake data.

Source and class balancing are enabled so that a large dataset cannot dominate every batch. The organizer demonstration/reference split is evaluation-only and is never added to training.

### Leakage controls

Our intended protocol is:

- freeze train, internal-validation, and test manifests before final training;
- deduplicate by image content rather than file path alone;
- keep generator families and source domains visible in reporting;
- never place final-test images or labels in initial training or feedback-based retraining;
- never use the final test result for augmentation design, checkpoint selection, hyperparameter tuning, calibration, or threshold selection;
- evaluate the final test only after the model and decision policy are frozen;
- keep the organizer reference set out of optimization.

**Non-negotiable rule:** the final test set never trains the model. This applies to the first training run and every future human-feedback retraining cycle.

The final README tables will only be filled from artifacts that identify the checkpoint, split manifest, sample count, threshold, and transformation parameters.

## 3. Earlier model direction: complementary forensic branches

The first Reelistic line of work combined texture, frequency, residual-noise, and semantic branches with a quality-aware fusion gate. It was useful for understanding several tensions:

- semantic evidence transfers well in some clean settings but can dominate fragile low-level evidence;
- blur and aggressive downsampling can erase forensic traces;
- source-aware validation can choose a different checkpoint from aggregate validation;
- calibration and low-FPR metrics matter when false accusations are costly.

That full history and its legacy measurements remain in [reelistic/JOURNAL.md](reelistic/JOURNAL.md). They are preserved as experiment history, not reported as measurements of the current DINOv3 checkpoint.

## 4. Why move to DINOv3?

We wanted one strong visual backbone that could expose features at multiple abstraction levels without maintaining several unrelated heavyweight encoders. DINOv3 ViT-B/16 provides a natural feature hierarchy and stays far below the 2B-parameter competition limit.

The working hypothesis became:

> A shared self-supervised backbone, tapped at low, middle, and high layers, can preserve forensic detail while still reasoning about global semantic consistency.

The deployed checkpoint contains 101,532,673 parameters.

## 5. Layer-tapping design

We tap layers 4, 8, and 12.

### Layer 4

Expected to retain comparatively local and low-level signals: edge behavior, high-frequency noise, interpolation artifacts, and pixel-grid irregularities.

### Layer 8

Expected to encode mid-range structure: texture repetition, material consistency, regional coherence, and relationships among nearby patches.

### Layer 12

Expected to carry higher-level scene information: composition, object relationships, lighting, physical plausibility, and anatomical consistency.

These descriptions are hypotheses about useful evidence, not standalone explanations for individual predictions. Final claims should be supported with ablation or probe results.

## 6. Forensic head

Each tapped hidden state feeds two branches:

- a patch branch reshapes patch tokens into a 14 x 14 map and applies a 3 x 3 convolution, GELU, global pooling, and flattening;
- a CLS branch applies a two-layer MLP to the global token.

Each output has 512 dimensions. Six outputs are concatenated into a 3,072-dimensional feature, normalized, and passed to the binary classifier. A separate projection head supports supervised contrastive learning during training.

This design keeps local patch evidence and global token evidence explicit until late fusion.

## 7. Training objective

The configured objective is:

```text
BCEWithLogits
+ 0.5 x binary Jensen-Shannon prediction consistency
+ 0.1 x supervised contrastive loss
```

Reasoning:

- BCE learns the primary binary decision;
- prediction consistency discourages large clean/transformed prediction shifts;
- supervised contrastive learning encourages a more stable class representation across sources and transformations.

The optimizer is AdamW with a lower backbone learning rate than the new forensic head. The configuration uses gradient clipping, cosine annealing, warm-up, EMA, mixed precision, and a ten-epoch schedule.

## 8. Transformation curriculum

We expose one transformation per augmented view. We deliberately avoid chaining every corruption together because a heavily chained sample may stop resembling a plausible platform transformation and makes failure attribution difficult.

### Stage 1: easier transformations, epochs 0-2

- JPEG quality 90;
- blur sigma 0.5 or 1.0;
- resize scale 0.5;
- noise sigma 0.02 or 0.05;
- color jitter 0.10;
- center crop 0.80.

### Stage 2: medium transformations, epochs 3-6

- JPEG quality 70 or 50;
- blur sigma 1.0 or 2.0;
- resize scale 0.25 or 0.5;
- noise sigma 0.05 or 0.10;
- color jitter 0.20;
- center crop 0.80.

### Stage 3: harder transformations, epochs 7-9

- JPEG quality 50 or 30;
- blur sigma 1.0 or 2.0;
- resize scale 0.25;
- noise sigma 0.05 or 0.10;
- color jitter 0.20;
- center crop 0.80.

Noise and blur have higher sampling weight in the current curriculum because they directly threaten high-frequency forensic evidence. This weighting should be revisited after the per-transformation validation table is available.

## 9. Checkpoint selection

The primary checkpoint monitor is internal-validation TPR at 1% FPR. We also track:

- TPR at 5% FPR;
- FPR at 95% TPR;
- FPR at 99% TPR;
- loss and accuracy;
- organizer-reference metrics for evaluation only.

The checkpoint currently deployed is named `best_competition_tpr_at_1_fpr.pt`. Its final clean and transformed measurements are intentionally not copied from unrelated runs. They will be added when the matching training, validation, and test result files are provided.

## 10. Deployment engineering

### Stable API boundary

The browser extension depends on a stable `/detect` response rather than model-specific code. This allowed the backend to move from the legacy branch ensemble to DINOv3 without changing the extension interaction.

### Google Cloud

The current prototype runs FastAPI on Google Compute Engine. Google Cloud Storage retains captured PNGs and event JSON. SQLAlchemy stores prediction and feedback records locally in SQLite and can switch to PostgreSQL through `DATABASE_URL`.

### ONNX

The DINOv3 detector was exported to ONNX opset 18 with dynamic batch size, a normalized `N x 3 x 224 x 224` input, and a single logit output. The generated graph is approximately 382 MB and passed `onnx.checker.check_model`.

Remaining ONNX work:

- publish the artifact with a permanent URL;
- record SHA-256;
- compare PyTorch and ONNX numerical outputs on a fixed image set;
- benchmark mean/P95 latency, throughput, and peak memory;
- optionally quantize only after measuring accuracy and low-FPR impact.

## 11. Human in the loop

The app stores successful predictions as reviewable records. Processing failures are also preserved as `UNREVIEWED`, with confidence zero, and are excluded from metrics.

Users can mark a result correct or wrong. A wrong response implies the opposite actual class. An administrator can later inspect the source image and metadata, revise the review status/actual label, and update the database. This creates an operational error-analysis queue and a future source of curated retraining examples.

We do not assume that user feedback is ground truth. Before retraining, feedback should be deduplicated, audited, and separated from evaluation data.

## 12. Result insertion checklist

When results are supplied, update the empty README tables only after verifying:

- exact checkpoint and epoch;
- data manifest and sample count;
- class mapping and positive class;
- fixed decision threshold;
- clean metrics;
- each transformation and parameter separately;
- confidence intervals where available;
- representative false positives and false negatives;
- no train/test overlap;
- matching PyTorch and ONNX outputs.

### Pending artifacts

| Artifact | Status | Notes |
|---|---|---|
| Training metrics | Pending | Add complete metric file and epoch history |
| Validation metrics | Pending | Include clean and every transformation parameter |
| Test metrics | Pending | Include untouched final split and error analysis |
| PyTorch/ONNX parity | Pending | Compare logits/probabilities on fixed samples |
| ONNX benchmark | Pending | Device, batch size, latency, throughput, memory |
| ONNX public download | Pending | Add permanent URL and SHA-256 |
| Qualitative errors | Pending | Add representative FP/FN examples with permission |

## 13. Known limitations and open questions

- Does each layer tap contribute under every transformation, or do some taps become harmful under severe corruption?
- How much does the contrastive term improve unseen-generator performance compared with consistency alone?
- Is the 0.5 decision threshold appropriate for deployment, or should it be calibrated to a target false-positive rate?
- How stable is performance across source domains where real and synthetic images do not share the same capture pipeline?
- Can ONNX quantization reduce cost without degrading TPR at 1% FPR?
- How should uncertain predictions be surfaced without encouraging overconfidence?

These questions should guide the next ablations and prevent a single aggregate accuracy value from hiding operational weaknesses.
