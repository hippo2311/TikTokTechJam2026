# Reelistic Project Journal

This journal records implementation history, experiment decisions, cluster
operations, completed work, and deferred work. The judge-facing model evidence
and usage instructions live in [README.md](README.md).

## Completed work

- Built a four-branch detector with texture, frequency, forensic-noise, and
  frozen CLIP ViT-L/14 semantic evidence.
- Added a quality-aware soft gate, branch dropout, supervised contrastive
  texture loss, source-aware checkpoint selection, bounded temperature
  calibration, and an explicit branch-disagreement gate signal.
- Downloaded and retained 13 approved WildFake training archives on the SoC
  cluster while excluding the organizer-reserved DALL-E/COCO holdout from all
  development stages.
- Built immutable train, validation, calibration, and reserved-test manifests
  from birdy654, SID, and approved WildFake families.
- Audited 1,824,222 unique decodable development records, removed 64,540
  content duplicates, and excluded 142 corrupt files.
- Trained and compared full seeds 42, 43, and 44 on A100/Titan V resources.
- Selected disagreement-gate seed 43 after matched source/condition evaluation.
- Froze the selected checkpoint, manifests, arguments, source hashes, and
  environment information.
- Audited and evaluated the isolated 4,998-COCO/8,843-DALL-E external holdout
  once, including a supplementary unique-content analysis.
- Consolidated duplicated metric/device logic and re-verified retraining after
  Python cleanup.
- Updated the judge-facing workflow SVG with the selected validation,
  calibration, robustness, and final-test evidence.

## Data and cluster history

- Cluster project root: `~/TechJam`
- Authoritative dataset root: `~/TechJam/Dataset`
- Working Titan V node: `xgpd0`, `gpu:nv:1`, 12 GB VRAM
- Working A100 node: `xgph0`, `gpu:a100-80:1`, 80 GB VRAM
- Titan V container: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
- Persistent A100 environment: `~/TechJam/.envs/a100-cu121-py312`
- Full development manifests contain 1,589,846 training rows, 189,040
  validation rows, 25,374 calibration rows, and 19,962 reserved birdy654 test
  rows before bounded evaluation sampling.

The full manifest is intentionally read before hierarchical limiting so every
source/class/family remains eligible. This makes even a tiny training smoke
take several minutes to start; it is not a GPU stall.

## Model-selection history

### Initial three-source checkpoint

Job `768268` selected epoch 2 by the source-aware objective:

| Stage | Balanced accuracy | Overall AUC | WildFake AUC | birdy654 AUC | SID AUC |
|---|---:|---:|---:|---:|---:|
| Initialization | 0.6932 | 0.7944 | 0.6463 | 0.9891 | 0.9086 |
| Selected epoch 2 | 0.7262 | 0.8291 | 0.6989 | 0.9899 | 0.9201 |
| Epoch 5 | 0.7264 | 0.8313 | 0.6903 | 0.9916 | 0.9204 |

Epoch 5 had a slightly higher aggregate AUC but weaker held-family WildFake
performance, so the source-aware selector retained epoch 2.

### First matched seed comparison

Job `769531` compared the same 1,000 examples per source under 15 conditions:

| Candidate | Clean birdy654 | Clean SID | Clean WildFake | Weakest-source mean | Severe mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| Seed 42 | 0.9913 | 0.9201 | 0.6718 | 0.6122 | 0.8065 | Eligible |
| Seed 43 | 0.9909 | 0.9103 | 0.6838 | 0.6272 | 0.8107 | Best original seed |
| Seed 44 | 0.9908 | 0.9148 | 0.6485 | 0.5955 | 0.8024 | Rejected |

### Shared-CLIP logit ensemble

Job `769895` evaluated a three-seed raw-logit average with one shared frozen
CLIP pass. Unique deployed parameters were 339,029,813 rather than a naive
three-times count. It was rejected because WildFake clean AUC fell from 0.6838
to 0.6712 and WildFake mean corruption AUC fell from 0.6272 to about 0.6136.

### Operating-point and family diagnosis

Jobs `770096` and `771201` added ROC/PR curves, TPR@1%FPR,
FPR@99%TPR, bootstrap intervals, per-family evidence, branch disagreement,
and semantic-mask ablations. The diagnosis showed that semantic evidence is
helpful on clean VQDM but can be harmful under severe downsampling and blur.

### Disagreement-gate candidate and full seeds

Bounded job `770115` tested a zero-initialized eight-parameter disagreement
correction while freezing the evidence branches. Full A100 seed training ran
as array job `770372`; matched evaluation ran as job `770373`.

| Candidate | Clean WildFake AUC | Weakest-source mean | Overall mean | Severe mean | Decision |
|---|---:|---:|---:|---:|---|
| Original seed 43 | 0.6838 | 0.6272 | 0.8377 | 0.8107 | Superseded |
| Disagreement seed 42 | 0.6800 | 0.6260 | 0.8404 | 0.8145 | Eligible |
| Disagreement seed 43 | 0.6974 | 0.6434 | 0.8455 | 0.8218 | Selected |
| Disagreement seed 44 | 0.6609 | 0.6097 | 0.8356 | 0.8130 | Rejected |

The selected checkpoint is
`cluster_outputs/disagreement_full_seed43_20260829/best_ensemble_calibrated.pt`.

## Final-test history

- Final-data audit job: `771381`, 1:49, exit `0:0`
- Frozen evaluation job: `771382`, 7:21, exit `0:0`
- Official paths: 13,841
- Unique content hashes: 8,717
- Same-label duplicate DALL-E paths: 5,124
- Cross-development overlap: zero
- Conflicting REAL/FAKE duplicate hashes: zero

The official path-weighted result reached ROC-AUC 0.8096 and balanced accuracy
0.7064. The unique-content result reached ROC-AUC 0.8138 and balanced accuracy
0.7089. Source and label are coupled in this holdout—COCO is always REAL and
DALL-E is always FAKE—so the result cannot isolate generator cues from every
dataset-domain cue.

## Cleanup and retraining verification

Python cleanup centralized binary metrics in `aigc_detector/metrics.py` and
device selection in `aigc_detector/utils/device.py`. Cluster entry-point
filenames were retained because Slurm invokes them directly. The obsolete
`tests/test_cluster_metrics.py` was replaced by module-owned
`tests/test_metrics.py`; the unused runtime dependency on scikit-learn was
removed.

Titan V job `771488` re-verified the cleaned training path using 64 balanced
development samples:

- 494/494 compatible checkpoint tensors loaded;
- 315,129,788 parameters instantiated and budget-checked;
- eight optimizer steps completed in 4.9 seconds;
- validation, resumable-state saving, and bounded calibration succeeded;
- job exit code `0:0`.

The 3.7 GB smoke checkpoints were deleted after verification; the small Slurm
logs were retained.

## Deferred or optional work

- Product/demo packaging and a polished user interface.
- Optional local copies of the selected checkpoint and audited final-test
  images; the cluster remains the authoritative storage location.
- Judge-facing latency/memory measurements and representative visual error
  examples.
- Any future replacement-backbone experiment. Such a model must be selected
  only with development validation/robustness evidence. The existing
  COCO/DALL-E result remains attached to frozen seed 43; a replacement needs a
  new untouched external holdout for a new final-test claim.

## Operational notes

- On campus: `ssh bryanngu@xlogin.comp.nus.edu.sg`
- Off campus: connect NUS VPN, then use
  `ssh -J bryanngu@stujump.comp.nus.edu.sg bryanngu@xlogin.comp.nus.edu.sg`
- Never run heavy downloads, manifest construction, training, or evaluation on
  the login node; submit them through Slurm.
- Keep the Apptainer/model caches while cluster work continues.
- Never overwrite a selected checkpoint or reuse an output directory for a
  different candidate.
