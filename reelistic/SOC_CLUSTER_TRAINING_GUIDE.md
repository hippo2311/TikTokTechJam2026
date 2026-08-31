# SoC Cluster — Practical GPU Training Notes

This is an operational guide for an experienced user. It records what was
actually verified on the SoC cluster for Reelistic; it is not a general Linux,
Python, or project-setup tutorial.

Cluster configuration and node images can change. Recheck live state before
copying any resource request.

## Current Reelistic status

Reelistic is not yet using A100 for every historical job:

| Work | Hardware used |
|---|---|
| Dataset download and preparation | Existing Slurm jobs, mostly pinned to xgpd0 |
| Manifest hashing/deduplication | normal partition |
| Full seed-42/43/44 training | Titan V on xgpd0 |
| A100 environment bootstrap | A100 80 GB on xgph0 |
| A100 batch-64 migration smoke | A100 80 GB on xgph0 |
| Current matched three-seed robustness | A100 80 GB on xgph0 |

The A100 training path is proven, but the currently trained full checkpoints
were produced on Titan V. Future compute-heavy training and evaluation can use
the separate A100 scripts after their outputs are reviewed. Do not silently
reinterpret the Titan checkpoints as A100-trained models.

## Access and login-node discipline

On campus:

~~~bash
ssh YOUR_USER_ID@xlogin.comp.nus.edu.sg
~~~

Off campus, connect to NUS VPN first:

~~~bash
ssh -J YOUR_USER_ID@stujump.comp.nus.edu.sg \
  YOUR_USER_ID@xlogin.comp.nus.edu.sg
~~~

The login node is for:

- editing and syncing small files;
- inspecting storage;
- sbatch, squeue, sacct, sinfo, and scontrol;
- light validation such as bash -n.

Do not run training, dataset decoding, full hashing, or large package
installation directly on the login node.

Never place passwords, tokens, private keys, or credentials in source files,
shell history, Slurm scripts, or logs.

## Verified GPU/runtime matrix

Observed on 29 August 2026:

| Nodes | Slurm GRES | Hardware | Runtime observation |
|---|---|---|---|
| xgpd0 | gpu:nv:1 | Titan V, 12 GB | /usr/bin/apptainer works |
| xgpg0–7 | gpu:a100-40:1 | A100 40 GB | No Apptainer/Singularity/native PyTorch on probed node |
| xgph0–9 | gpu:a100-80:1 | A100 80 GB | No container runtime; persistent Python environment works |
| xgph10–15 | gpu:a100-40:1 or node-specific count | A100 40 GB | Verify exact GRES before use |
| xgpi nodes | gpu:h100-47:1 or gpu:h100-96:1 | H100 instances/NVL | No runtime/native PyTorch on probed node |

The A100 and H100 hardware can be allocated even when Apptainer is missing.
Container cache files do not provide the Apptainer executable.

Check live resources:

~~~bash
sinfo -N -o '%N|%P|%t|%G|%c|%m|%f' | grep -E '^(NODELIST|xgp)'
sinfo -o '%P|%a|%l|%D|%G|%N'
scontrol show node xgph0
~~~

Check whether a request is valid without submitting it:

~~~bash
sbatch --test-only \
  --partition=gpu \
  --nodelist=xgph0 \
  --time=00:02:00 \
  --cpus-per-task=1 \
  --mem=2G \
  --gres=gpu:a100-80:1 \
  --wrap='hostname'
~~~

In our tests, the normal gpu partition accepted A100/H100 requests. The test
partition returned an invalid-QoS error for this account. A valid request can
still wait because of priority, reservations, or current allocations.

## Two supported runtime paths

### Titan V: Apptainer

The tested image is:

~~~text
docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
~~~

Persistent caches:

~~~bash
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
~~~

Install only small missing Python packages into job-local storage. Do not
reinstall PyTorch inside every job.

### A100: persistent Python 3.12 environment

The A100 nodes had Python 3.12, pip, venv, and outbound package access, but no
container runtime. Reelistic therefore builds an isolated environment once:

~~~bash
sbatch slurm/bootstrap_a100_env.sbatch
~~~

Project environment:

~~~text
~/TechJam/.envs/a100-cu121-py312/
~~~

The bootstrap verifies:

- torch 2.4.1+cu121;
- torchvision 0.19.1+cu121;
- timm and scikit-learn imports;
- CUDA visibility;
- A100 device name and VRAM;
- a final .ready marker.

Training scripts refuse to start without the marker. Do not share another
user's environment or copy it between incompatible Python/node families.

For another project, copy the bootstrap pattern and pin compatible package
versions. Keep the environment outside Git and never overwrite an incomplete
environment automatically; inspect or remove the exact failed build directory
first.

## Storage model

Persistent shared storage:

~~~text
~/PROJECT/
├── Dataset/                 authoritative datasets
├── cluster_outputs/         checkpoints and metrics
├── .envs/                   persistent A100 Python environments
├── .cache/                  model/download caches
└── .apptainer-cache/        Titan container cache
~~~

SLURM_TMPDIR and node-local /tmp are ephemeral. They are suitable for package
layers, extracted scratch data, and temporary preprocessing—not the only copy
of a dataset or checkpoint.

Downloading through a GPU job does not automatically make data persistent or
give it more permanent capacity. Persistence depends on the destination path.
Reelistic downloads into ~/TechJam/Dataset, which is shared project storage.

Before large extraction:

~~~bash
quota -s
du -sh ~/PROJECT/Dataset ~/PROJECT/cluster_outputs 2>/dev/null
df -h ~/PROJECT
~~~

Large recursive du calls over millions of files can be slow. Stop the inspection
with Ctrl+C if necessary; that does not affect Slurm jobs.

Keep only:

- authoritative datasets and manifests;
- selected calibrated and uncalibrated checkpoints;
- protected baselines;
- reproducibility logs and reports;
- resumable state only while resume is still useful.

Delete superseded smoke checkpoints, failed temporary environments, redundant
optimizer states, and failed-job logs only after verifying the selected
artifacts.

## Sync code without damaging cluster data

From the local project:

~~~bash
rsync -avP --dry-run \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.envs' \
  --exclude '__pycache__' \
  --exclude '.cache' \
  --exclude '.apptainer-cache' \
  --exclude 'Dataset' \
  --exclude 'cluster_outputs' \
  --exclude '*.pt' \
  ./ YOUR_USER_ID@xlogin.comp.nus.edu.sg:~/PROJECT/
~~~

Remove --dry-run only after reviewing the targets. Off campus, add:

~~~bash
-e "ssh -J YOUR_USER_ID@stujump.comp.nus.edu.sg"
~~~

Avoid rsync --delete unless remote deletion is explicitly intended. A normal
code sync must not remove cluster-only datasets, environments, or checkpoints.

## Reelistic workload placement

| Workload | Preferred resource | Main files |
|---|---|---|
| Download approved WildFake archives | Existing download job; destination must be persistent | slurm/download_wildfake_archives.sbatch |
| Prepare SID subset | Existing preparation job | slurm/prepare_external_pilot.sbatch |
| Build development manifests | normal CPU partition | slurm/build_dataset_manifests.sbatch |
| Bootstrap A100 environment | A100 80 GB once | slurm/bootstrap_a100_env.sbatch |
| A100 smoke/full training | A100 80 GB | slurm/train_three_source_l14_a100.sbatch |
| Titan fallback training | Titan V | slurm/train_three_source_l14.sbatch |
| Matched robustness | A100 80 GB | slurm/evaluate_three_seeds_a100.sbatch |
| Locked final manifest/test | Only after model freeze | dedicated final-test scripts |

Some older data/final-test scripts request xgpd0 even when their work is
CPU/network-heavy. Do not infer that GPU acceleration helps downloading or
hashing. Those scripts are retained because the node/runtime combination is
known to work; resource requests should be refactored only after a smoke test.

## A100 smoke and full-run submission

The A100 training script defaults to a safe smoke:

~~~bash
sbatch --export=ALL,\
OUTPUT_DIR="$HOME/TechJam/cluster_outputs/a100_smoke_unique" \
  slurm/train_three_source_l14_a100.sbatch
~~~

It defaults to one epoch, 4,000 balanced draws, physical batch 64, and no
gradient accumulation.

For a full five-epoch run:

~~~bash
OUTPUT_DIR="$HOME/TechJam/cluster_outputs/a100_full_seed46"

sbatch --export=ALL,\
OUTPUT_DIR="$OUTPUT_DIR",\
EPOCHS=5,\
BATCH_SIZE=64,\
GRAD_ACCUM_STEPS=1,\
MAX_TRAIN_SAMPLES=100000,\
EPOCH_SAMPLES=100000,\
MAX_VAL_SAMPLES=10000,\
MAX_CALIBRATION_SAMPLES=10000,\
SEED=46 \
  slurm/train_three_source_l14_a100.sbatch
~~~

Always use a new output directory. The script refuses to reuse a directory
containing a checkpoint.

Important: physical batch 64 and physical batch 16 × accumulation 4 have the
same effective batch, but they are not mathematically identical when BatchNorm
or batch-dependent operations are present. For a strict Titan/A100 comparison,
first run A100 with batch 16 and accumulation 4. Treat batch 64 as a separate
training configuration.

## Monitoring without interruption

~~~bash
squeue -u "$USER" -o '%.18i %.24j %.9T %.10M %R'
squeue -j JOB_ID -o '%.18i %.24j %.9T %.10M %R'

tail -f slurm/logs/JOB_NAME-JOB_ID.out
tail -n 50 slurm/logs/JOB_NAME-JOB_ID.err

sacct -j JOB_ID \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,MaxRSS
~~~

Ctrl+C exits tail -f only. It does not cancel the job. Use scancel JOB_ID only
when cancellation is intentional.

Useful training filter:

~~~bash
grep -E '^\[(runtime|data|sampling|init-val|train|val|checkpoint|calibration|done)' \
  slurm/logs/JOB_NAME-JOB_ID.out | tail -n 80
~~~

Interpretation:

- PENDING (Resources/Priority): valid request waiting for allocation;
- PENDING (Dependency): waiting for an upstream job;
- DependencyNeverSatisfied: an upstream job failed;
- COMPLETED with 0:0: successful;
- OUT_OF_MEMORY: lower physical batch before changing the model;
- TIMEOUT: resume or use an eligible longer partition;
- exit 127 at startup: command/runtime missing on that node;
- MaxRSS: host RAM, not GPU VRAM;
- quiet logs during manifest parsing or validation: often CPU/shared-I/O work,
  not a hang.

## Performance expectations

Observed Reelistic training throughput:

| GPU/configuration | Throughput |
|---|---:|
| Titan V, batch 16 | about 80 images/s |
| A100 80 GB, batch 64 | about 176 images/s |

The A100 was roughly 2.2× faster for training compute. End-to-end speedup is
smaller when manifest parsing, checkpoint loading, validation, or shared storage
dominates. Measure identical workloads before quoting a production speedup.

## Checkpoint and evaluation discipline

- Save a resumable state every epoch.
- Select by source-aware validation, not aggregate accuracy alone.
- Calibrate only after checkpoint selection on a separate calibration split.
- Evaluate multiple candidates on the exact same sample fingerprint.
- Keep final-test data isolated until code, threshold, preprocessing,
  calibration, and checkpoint are frozen.
- A frozen CLIP backbone is still required at inference; frozen means its
  weights are not updated, not that it can be omitted.
- Never overwrite a protected checkpoint while testing another GPU or runtime.

## Recommended operational sequence

1. Sync code with Dataset, environments, caches, outputs, and checkpoints
   excluded.
2. Verify partition, node state, GRES label, time limit, and runtime.
3. Bootstrap the persistent environment once if using A100.
4. Run a bounded smoke in a fresh output directory.
5. Confirm CUDA device, memory stability, loss movement, validation,
   calibration, and checkpoint creation.
6. Run the full job only after the smoke succeeds.
7. Monitor through several logged steps, then disconnect safely.
8. Compare candidates on matched source/family robustness data.
9. Freeze artifacts and hashes.
10. Open the isolated final test exactly once.

If the same failure repeats without new evidence, stop resubmitting and diagnose
the first reproducible error.
