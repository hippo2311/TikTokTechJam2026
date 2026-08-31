# Reelistic — Remaining Decision Plan

This plan starts from the completed three-source, three-seed training runs. It
keeps the competition-grade data protocol and limits remaining work to evidence
that can change the final checkpoint or materially strengthen the demo.

## Fixed foundations

Do not rework these:

- 1,824,222 unique decodable development records;
- 64,540 content duplicates and 142 corrupt files excluded;
- disjoint train, validation, calibration, reserved birdy654 test, and external
  final-test paths;
- source → class → family balanced training;
- complete-family WildFake validation holdouts;
- frozen CLIP ViT-L/14 semantic backbone;
- four-branch quality-aware fusion with branch dropout;
- fixed manifests and confirmation seeds 42, 43, and 44;
- hard parameter-budget enforcement in aigc_detector/param_budget.py;
- isolated COCO/DALL-E final holdout, still unopened.

The selected disagreement model contains 315,129,796 parameters, safely below
two billion.

## Interpret the generalization evidence correctly

The seed-42 selected checkpoint reported:

| Development source | Validation AUC |
|---|---:|
| birdy654 | 0.9899 |
| SID | 0.9201 |
| WildFake held-family set | 0.6989 |

The WildFake gap is the main weakness. However, 0.6989 is an aggregate over a
held-family WildFake subset, not one generator family. The sources also differ
in content and collection pipeline, so the gap is strong evidence of
cross-domain/generalization weakness but not proof that generator novelty alone
caused it. The matched three-seed evaluation supersedes these provisional
numbers.

Do not assume the locked COCO/DALL-E result will equal WildFake performance.
COCO/DALL-E couples source with class and is therefore a separate, imperfect
external test.

## Gate 1 — Matched three-seed selection completed

Job 769531 completed successfully on xgph0 (A100 80 GB). It evaluated seed 42,
43, and 44 checkpoints on:

- the same 1,000 balanced examples per source;
- a fixed evaluation seed of 42;
- clean plus 14 corruption conditions;
- fusion and all four individual branches;
- mean gate weights for every source/condition.

Each JSON contains a SHA-256 fingerprint of the exact ordered sample identities.
cluster/compare_matched_robustness.py rejects seed, count, condition, or
fingerprint mismatches.

The provisional ranking rule is:

1. reject any candidate with more than 0.01 clean-AUC regression on any source
   versus seed 42;
2. maximize weakest-source mean corruption AUC;
3. break ties by overall source/condition mean AUC;
4. then break ties by clean mean AUC.

The matched result is:

| Candidate | Clean birdy654 | Clean SID | Clean WildFake | Weakest-source mean | Severe mean | Result |
|---|---:|---:|---:|---:|---:|---|
| seed 42 | 0.9913 | 0.9201 | 0.6718 | 0.6122 | 0.8065 | Eligible |
| **seed 43** | **0.9909** | 0.9103 | **0.6838** | **0.6272** | **0.8107** | **Best single** |
| seed 44 | 0.9908 | 0.9148 | 0.6485 | 0.5955 | 0.8024 | Rejected |

Seed 43 is now the selected single-checkpoint reference. Its improvement does
not close the WildFake gap, so the remaining work must still target shared
cross-domain bias rather than only seed variance.

### Gate 1B — Shared-CLIP logit ensemble rejected

Job 769895 completed successfully in 22:07 on the A100. The implementation:

- verifies that all frozen CLIP backbone tensors are identical;
- runs CLIP once per image;
- keeps each seed's semantic probe, texture, frequency, noise, and gate heads;
- averages raw two-class fusion logits, never probabilities or decisions;
- fits one new temperature on the disjoint calibration manifest;
- compares against seed 43 on the same sample fingerprints and selection rule.

Exact deployed parameter accounting is 339,029,813: 303,179,776 shared CLIP
parameters, 35,850,036 parameters across the three seed-specific paths, and one
ensemble temperature. The naive unshared count would be 945,389,364. Both are
below two billion, but only the unique shared-graph count represents deployment.

The ensemble's WildFake clean AUC was 0.6712 versus seed 43's 0.6838, a −0.0126
regression that exceeded the −0.01 guard. Its WildFake mean corruption AUC was
about 0.6136 versus 0.6272 for seed 43. It therefore failed both the clean guard
and weakest-source objective. This confirms that averaging reduced neither the
shared WildFake bias nor enough variance to justify the extra heads. Keep the
code as reproducible negative evidence, but do not package this candidate.

## Gate 2 — Add deployment operating-point metrics

Seed 43 remains selected. A100 job 770096, submitted from
`slurm/diagnose_seed43_a100.sbatch`, adds these to
aigc_detector.metrics.binary_classification_metrics:

- TPR at at most 1% FPR;
- FPR at at least 99% TPR;
- the corresponding thresholds;
- ROC-curve points;
- precision-recall curve points.

Report ROC-AUC, average precision, TPR@1%FPR, and FPR@99%TPR overall and by
source. These metrics are high value because moderation systems usually care
more about strict false-positive regions than average threshold performance.
Use a larger fixed evaluation sample than the 1,000-image robustness smoke when
possible: with only 500 REAL examples, 1% FPR represents five errors. Report
sample counts and a bootstrap confidence interval so the operating point is not
presented with false precision.

The diagnostic uses 5,000 balanced examples per source, 300 deterministic
stratified bootstrap repetitions, and includes full ROC/precision-recall points
for fusion. The external final holdout remains untouched.

## Gate 3 — Focused diagnosis before architecture work

Job 770096 completed successfully in 18:34. Deployment-oriented clean results
were:

| Source | Samples | ROC-AUC | TPR@1%FPR | FPR@99%TPR |
|---|---:|---:|---:|---:|
| birdy654 | 4,971 | 0.9889 | 0.8215 | 0.1704 |
| SID | 591 | 0.9103 | 0.5703 | 0.8450 |
| WildFake | 5,000 | 0.6964 | 0.1068 | 0.9868 |

This confirms that aggregate AUC understates the deployment weakness: at only
1% false positives, the detector catches about 10.7% of WildFake fakes in this
development sample.

Only one eligible held-out fake family had at least 100 examples in the fixed
WildFake validation manifest: VQDM. Against the deterministic opposite-class
reference pool, its clean AUC was 0.7218. Semantic-mask ablation showed:

| Condition | Full fusion | Semantic masked | Mask delta |
|---|---:|---:|---:|
| clean | 0.7218 | 0.6220 | −0.0998 |
| downsample 25% | 0.5768 | 0.6735 | +0.0967 |
| blur 2.0 | 0.6147 | 0.6560 | +0.0413 |
| JPEG 30 | 0.6327 | 0.6096 | −0.0231 |
| noise 0.10 | 0.5933 | 0.5777 | −0.0156 |

Semantic evidence is valuable on clean/JPEG/noise but misrouted under severe
downsampling and blur. Branch disagreement is also higher on WildFake errors
and correlates with error, while the existing gate only weakly reacts to it.

The completed bounded diagnostics were:

1. Clean per-family branch and fusion metrics for WildFake.
2. The five weakest families under only four severe conditions:
   downsample_25, blur_2.0, jpeg_30, and noise_0.10.
3. A semantic-mask inference ablation using the same examples. Because branch
   dropout trained the gate to tolerate missing experts, this is more
   informative than comparing fusion with semantic-only AUC.
4. Branch disagreement versus gate weights and errors.

Do not build a full family × 15-condition matrix initially. It is expensive and
most cells will not change the decision. Expand only around observed failures.

## Gate 4 — Explicit disagreement gate candidate, then full A100 seeds

Completed A100 job 770115 tested the one selected change. It adds a zero-initialized
eight-parameter correction from normalized branch-probability variance to the
four gate logits. Zero initialization preserves the seed-43 model exactly at
startup. All four evidence branches are frozen and only the fusion gate trains,
so this is a controlled routing experiment rather than an architecture-wide
fine-tune.

The bounded candidate used two 100,000-draw balanced epochs and passed the same
matched 1,000-image-per-source × 15-condition gate against seed 43:

| Metric | Seed 43 | Disagreement gate | Delta |
|---|---:|---:|---:|
| Clean mean AUC | 0.8617 | 0.8646 | +0.0029 |
| Weakest-source mean corruption AUC | 0.6272 | 0.6319 | +0.0047 |
| All source/condition mean AUC | 0.8377 | 0.8410 | +0.0033 |
| Severe-condition mean AUC | 0.8107 | 0.8157 | +0.0050 |
| Mean fusion minus semantic AUC | 0.0061 | 0.0094 | +0.0033 |

Every clean source improved, so the candidate advanced. Array job 770372
completed full confirmation training for seeds 42/43/44: five 100,000-draw
balanced epochs per seed, initialized from each seed's original checkpoint.
All tasks completed successfully in about 71 minutes each and accepted bounded
temperature calibration. Matched job 770373 then completed in 2:07:47.

| Matched candidate | Clean WildFake AUC | Weakest-source mean | Overall mean | Severe mean | Decision |
|---|---:|---:|---:|---:|---|
| Original seed 43 | 0.6838 | 0.6272 | 0.8377 | 0.8107 | Superseded |
| Disagreement seed 42 | 0.6800 | 0.6260 | 0.8404 | 0.8145 | Eligible, no weakest-source gain |
| **Disagreement seed 43** | **0.6974** | **0.6434** | **0.8455** | **0.8218** | **Selected** |
| Disagreement seed 44 | 0.6609 | 0.6097 | 0.8356 | 0.8130 | Rejected clean regression |

The selected checkpoint is
`cluster_outputs/disagreement_full_seed43_20260829/best_ensemble_calibrated.pt`.
The improvement is seed-dependent rather than uniform, so do not ensemble the
three improved seeds. Its larger diagnosis completed as job `771201`:

| Source | Samples | ROC-AUC | TPR@1%FPR | FPR@99%TPR |
|---|---:|---:|---:|---:|
| birdy654 | 4,971 | 0.9902 | 0.8321 | 0.1512 |
| SID | 591 | 0.9165 | 0.5652 | 0.8250 |
| WildFake | 5,000 | 0.7109 | 0.1156 | 0.9856 |

For VQDM, semantic masking changed AUC by −0.0716 on clean, +0.1376 on
downsampling 25%, +0.0753 on blur 2.0, +0.0168 on JPEG 30, and −0.0112 on
noise 0.10. The change therefore improves overall routing but does not solve
severe-transform routing. Freeze it and disclose that limitation; do not add a
second post-selection architecture change.

### If the gate misroutes high-disagreement examples

The gate already consumes all four branch logits, so disagreement is already
implicit. First verify that it fails to use that information. If it does, test
one explicit scalar such as normalized branch-probability variance or entropy.
Do not add a CLIP-distance feature first; it requires stable family centroids
and creates another distribution-specific dependency.

### If texture is consistently weak on unseen families

Test neighboring-pixel residual/NPR input using the existing ResNet-18 before
increasing capacity. Compare against raw-RGB texture on the same manifests and
matched robustness samples. Consider ResNet-34 only if NPR helps and capacity
still limits it.

### If fusion depends excessively on semantic evidence

Run the semantic-mask ablation first. Do not replace ViT-L/14 merely because it
holds most parameters; parameter share is not a performance objective. A frozen
ViT-B or larger frozen CLIP comparison is justified only if matched ablation
evidence shows a clear accuracy/latency trade-off.

### If real embeddings are diffuse

The texture branch already has a projection head and supervised contrastive
loss. A real-only compactness loss is a later controlled alternative, not a new
default. Test it only after inspecting real/fake embedding distributions.

Do not unfreeze CLIP. Do not attempt ViT-H/14, DIRE, a fifth branch, or several
simultaneous architecture changes during the remaining hackathon path.

Use the A100 for a bounded one-seed smoke and matched diagnostic comparison of
the chosen change first. Only if it beats the Gate 1 winner should the improved
configuration receive full seed-42/43/44 A100 training. Gate 4 is therefore the
right place for the user's end goal of full improved A100 seeds: earlier would
triple the cost before the architecture change has evidence; later would leave
insufficient time to confirm and freeze it.

## Gate 5 — Development package frozen

Before touching the external final holdout:

- choose one calibrated checkpoint and one fixed global threshold;
- record code/archive hash and all development-manifest SHA-256 hashes;
- preserve calibrated and uncalibrated checkpoint hashes;
- record arguments, seed, job IDs, logs, package versions, CUDA, and GPU;
- save matched robustness, operating-point, ablation, and calibration reports;
- remove non-selected resumable optimizer states only after selection;
- copy the frozen checkpoint and reports to the Mac.

Job `771273` created the final immutable package and hashes at
`cluster_outputs/frozen_disagreement_seed43_final_20260829`. It fixed the
selected checkpoint, uncalibrated checkpoint, threshold 0.5, development
manifests, reports, parameter count, runtime metadata, and the corrected
external-evaluation source snapshot before either external archive was opened.

Any model, threshold, calibration, or preprocessing change after opening the
final test invalidates its one-shot status.

## Gate 6 — Evaluate COCO/DALL-E exactly once

Only after Gate 5:

~~~bash
sbatch slurm/download_final_test_archives.sbatch
sbatch slurm/build_final_test_manifest.sbatch
cat Dataset/WildFake/final_test/manifests/audit.json
~~~

Require exactly 4,998 COCO REAL and 8,843 DALL-E Advanced FAKE images, successful
decoding, no internal duplicates, and no hash overlap with development data.

Then evaluate once with the frozen checkpoint. Report ROC-AUC, average
precision, balanced accuracy, class recalls, confusion counts, calibration,
TPR@1%FPR, FPR@99%TPR, branch metrics, gate weights, latency, memory, and hashes.
Disclose that source and label are coupled.

## Demo and judge-facing evidence

The model already exposes probability, branch scores, gate weights, and quality
features. Build the demo around those before adding heavier explainability:

1. calibrated FAKE probability and uncertainty;
2. branch-evidence bars and per-image gate weights;
3. quality indicators and transformation warnings;
4. selected texture patches and an FFT spectrum overlay;
5. optional Grad-CAM on the texture branch if it remains interpretable;
6. representative success and failure cases from development data only.

This creates a clear demo story: the detector does not return only a label; it
shows which forensic evidence contributed and when the input quality makes the
decision less reliable.

## A100 impact

The A100 environment and batch-64 smoke run completed successfully:

| Measurement | Titan V full runs | A100 smoke |
|---|---:|---:|
| Physical batch | 16 | 64 |
| Gradient accumulation | 4 | 1 |
| Effective batch | 64 | 64 |
| Observed step time | about 0.20 s | about 0.36 s |
| Approximate throughput | about 80 images/s | about 176 images/s |

The A100 delivered roughly 2.2× training throughput while preserving effective
batch size. A 500,000-draw training phase would therefore be approximately 47
minutes of training compute instead of 104 minutes, before validation,
checkpoint loading, and calibration. Treat this as an estimate until a matched
full A100 run is measured.

Robustness evaluation gains less because the large manifest and checkpoints
must still be loaded from the shared filesystem. Job 769895 removes two of
three CLIP forward passes, but still runs all forensic heads three times.

## Remaining checklist

- [x] Three full confirmation seeds completed.
- [x] A100 persistent environment and batch-64 smoke completed.
- [x] Matched three-seed robustness job completed and reviewed.
- [x] Seed 43 selected as the best single checkpoint.
- [x] Shared-CLIP raw-logit ensemble compared and rejected.
- [x] Low-FPR operating metrics added for the selected checkpoint.
- [x] Focused per-family/ablation diagnosis completed.
- [x] Explicit disagreement gate passed the bounded matched gate.
- [x] Full disagreement-gate seeds and matched confirmation completed.
- [x] Selected disagreement seed 43 passed large-sample operating diagnosis.
- [x] Development package frozen with hashes (job `771259`).
- [ ] Isolated COCO/DALL-E holdout audited and evaluated once.
- [ ] Demo and judge-facing report completed.
