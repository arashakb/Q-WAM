# ImageWAM

Q-WAM on ImageWAM (FLUX.2 [klein] 4B image-editing expert + flow-matching action expert in a
Mixture-of-Transformers), evaluated on RoboTwin 2.0 (50 tasks x 100 episodes, clean and randomized).
This covers the paper configuration (Table 1), the bf16 reference, and the component ablation
(Table 3). Accuracy numbers use simulated W4A4 quantization (INT4 codes are dequantized and the
GEMM runs in bf16); no INT4 kernels are included.

## Upstream code and weights

| Component | Source | Version |
|---|---|---|
| ImageWAM | `github.com/yuyangalin/ImageWAM` | `00a4e7afe82a2245f77a95240730a572212edefd` |
| RoboTwin 2.0 | vendored in ImageWAM at `third_party/RoboTwin` | upstream `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| FLUX.2 source | `github.com/black-forest-labs/flux2` | `50fe5162777813d869182b139e83b10743caef15` |
| ImageWAM RoboTwin checkpoint | HF `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin` | `model.pt`, `dataset_stats.json` |
| FLUX.2 klein base 4B | HF `black-forest-labs/FLUX.2-klein-base-4B` | `flux-2-klein-base-4b.safetensors` |
| FLUX.2 autoencoder | HF `black-forest-labs/FLUX.2-dev` (gated) | `ae.safetensors` |
| Text encoder | HF `Qwen/Qwen3-4B` | |

Rebuilding the ASP subspaces additionally needs the RoboTwin 2.0 LeRobot demonstrations released
with FastWAM (HF dataset `yuanty/robotwin2.0-fastwam`) in a FastWAM checkout (`FASTWAM_ROOT`),
read through `qwam.robotwin_calib`.

## Environment

One conda environment holds the model and the simulator: Python 3.10, torch 2.9.0+cu126,
torchvision 0.24.0, transformers 4.56.1 (the FLUX.2 pin of ImageWAM), hydra-core 1.3.2, sapien
3.0.0b1, mplib 0.2.1, CuRobo and the other RoboTwin 2.0 dependencies, plus the ImageWAM package
(`pip install -e <ImageWAM> --no-deps`). Upstream recommends Python 3.11 and torch 2.7.1; the
numbers below were produced with the versions listed here. Evaluation used 8 NVIDIA L40S (48 GB),
one task per GPU; building the ASP subspaces needs one GPU with about 24 GB.

## Setup

1. Clone ImageWAM at the commit above and follow its README for the FLUX.2 source tree, the
   weights, and the RoboTwin environment. The RoboTwin assets (`background_texture`,
   `embodiments`, `objects`) come from `third_party/RoboTwin/assets/_download.py` (HF dataset
   `TianxingChen/RoboTwin2.0`), unzipped in that directory.
2. Apply the Q-WAM patch, link the RoboTwin policy and write `.env.local`:

   ```bash
   export IMAGEWAM_ROOT=/path/to/ImageWAM
   FLUX2_SRC=/path/to/flux2 PYTHON_BIN=$(which python) bash scripts/imagewam/setup_imagewam.sh
   ```

   Unset paths default to the upstream README layout under `$IMAGEWAM_ROOT/checkpoints`
   (`FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`, `CKPT_PATH`, `DATASET_STATS_PATH`).

The patch (`patches/imagewam/0001-robotwin-policy-qwam-hooks.patch`, 18 added lines in
`experiments/robotwin/imagewam_policy/deploy_policy.py`) runs after the bf16 weights are loaded:
`IW_CALIB_OUT` installs the activation-absmax hooks, `IW_QUANT_CKPT` replaces the MoT Linears with
the W4A4 modules of `qwam.imagewam.runtime`, and calibration shards are tagged with the task name.
The launchers put this repository on the workers' `PYTHONPATH`. The FLUX.2 paths reach the model as
Hydra overrides (upstream launcher) or `model_overrides` (subspace builder), so no config file is
changed.

## Pipeline

All commands run from the root of this repository with `IMAGEWAM_ROOT` exported.

**1. Smoothing calibration (optional).** The activation absmax used in the paper is shipped as
`artifacts/imagewam/imagewam_act_absmax_c50.pt` and is the default of the later steps. It was
collected from one bf16 closed-loop clean episode per task (the first evaluation episode of each
of the 50 tasks) with per-input-channel absmax hooks on the MoT Linears; the per-task shards are
merged by elementwise max over the 154 Linears that execute. To recompute it:

```bash
CALIB_DIR=work/imagewam/calib bash scripts/imagewam/calibrate_absmax.sh
# -> work/imagewam/calib/imagewam_act_absmax_c50.pt; pass it with --absmax or IW_ABSMAX
```

The rollouts draw unseeded sampler noise, so a recomputed absmax differs slightly from the shipped
one.

**2. ASP subspaces (optional).** The rank-32 basis used in the paper is shipped as
`artifacts/imagewam/imagewam_asp_subspaces_r32.pt` (65 action-expert Linears) and is the export
default. To recompute a basis:

```bash
# FastWAM environment (torchcodec): calibration frames, one random episode per task, seed 42
FASTWAM_ROOT=/path/to/FastWAM python scripts/common/dump_c50_frames.py --out-dir work/c50_frames
# ImageWAM environment, one GPU
set -a; source $IMAGEWAM_ROOT/.env.local; set +a
python scripts/imagewam/build_asp_subspaces.py --frames-dir work/c50_frames \
    --out work/imagewam/imagewam_asp_subspaces_r32.pt
```

What the builder computes: for every action-expert Linear l, the action-output Gram
G_l = E[J_l^T J_l] with J_l the Jacobian of the action expert's prediction with respect to the
layer input x_l, taken at the first denoising step of the action sampler (`action_step_idx` 0);
the video expert is prefilled once into its KV cache and is constant for this gradient. Each frame
adds one Hutchinson probe to a Nystrom sketch with 64 columns (rank 32 + 32 oversampling). The
sketch is mapped to the smoothed and rotated basis of the quantizer, H diag(s) G_l diag(s) H
(alpha 0.5, Hadamard block cap 1024), and V_l is its top-32 eigenbasis. Frames are 2 evenly spaced
frames per calibration episode (100 frames; `--per-ep 0` uses all frames). `action_encoder` and
`time_in.in_layer` receive no gradient and have no subspace. The shipped basis was built with these
settings by an earlier revision of the script whose probes and sampler noise were not seeded, so a
rebuild gives a different basis rather than the same file.

**3. Export.** Defaults are the paper configuration: W4A4, weight and activation group 32,
alpha 0.5, block Hadamard with block cap 1024, ASP rank 32 on the action expert, with the shipped
calibration absmax and subspace file.

```bash
python scripts/imagewam/export_qwam.py \
    --base-ckpt $IMAGEWAM_ROOT/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt \
    --out work/imagewam/imagewam_w4a4_qwam_r32_g32.pt
```

This quantizes 153 Linears (66 action expert, of which 65 carry the ASP branch, and 87 video
expert); `action_encoder` (14 -> 1024) stays in bf16 and the video `final_layer` Linears, which
never run, are excluded. BPW = (4 sum(in*out) + 16 sum(out*in/32) + 16 * 32 * sum_ASP(in+out)) /
sum(in*out) = 4.5789 (`bpw_with_asp_branch` in the checkpoint meta). The export runs on CPU in
about a minute. With the shipped files it reproduces the evaluated checkpoint byte for byte; the
fp32 CPU GEMMs follow the BLAS thread partitioning, and this held with the default torch thread
count (64 MKL threads on the machine used), while fewer than 32 threads changed a few fp16 values
of the 1024 -> 14 action head.

**4. Evaluation** (`IW_CACHE_WEIGHTS=1` keeps the dequantized weights resident while GPU memory
allows; the outputs are identical either way).

```bash
# Q-WAM: export if the checkpoint is missing, then clean and randomized runs
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/imagewam/run_qwam.sh
# bf16 reference
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/imagewam/run_bf16.sh
# one run directly
GPU_IDS=0,1,2,3,4,5,6,7 TASKS='' EPISODES=100 PHASES='[clean]' NUM_GPUS=8 MAX_TASKS_PER_GPU=1 \
    ARM=qwam_clean IW_CACHE_WEIGHTS=1 IW_QUANT_CKPT=work/imagewam/imagewam_w4a4_qwam_r32_g32.pt \
    bash scripts/imagewam/run_imagewam_robotwin.sh
```

Table 3 rows (same group size, rank 0):

```bash
IW_QUANT_CKPT=work/imagewam/imagewam_w4a4_smoothrot_g32.pt ARM_PREFIX=smoothrot \
    EXPORT_ARGS="--subspaces none" bash scripts/imagewam/run_qwam.sh
IW_QUANT_CKPT=work/imagewam/imagewam_w4a4_group_g32.pt ARM_PREFIX=group \
    EXPORT_ARGS="--no-smooth --no-rotate --subspaces none" bash scripts/imagewam/run_qwam.sh
```

A run of 50 tasks x 100 episodes took about 5-6 hours on 8 L40S. Results are written to
`$IMAGEWAM_ROOT/evaluate_results/robotwin/<checkpoint tag>/<timestamp>/`; an interrupted run
resumes with `IW_EXTRA_ARGS="EVALUATION.output_dir=<that run directory>"`.

**5. Results.** Success is the mean over the 50 tasks of the per-task success rate over 100
episodes. The per-episode video names (`episode<N>_randomized-<b>_success-<b>.mp4`) are the source
of truth, since `summary.csv` of a resumed run can miss tasks:

```bash
python scripts/imagewam/read_results.py <clean run dir> <randomized run dir>
```

## Expected results

| | BPW | Clean | Randomized | Avg. |
|---|---|---|---|---|
| bf16 | 16.00 | 92.82 | 93.70 | 93.26 |
| per-group W4A4 | 4.50 | 21.16 | 20.14 | 20.65 |
| + smoothing and rotation | 4.50 | 86.78 | 88.06 | 87.42 |
| + ASP (Q-WAM) | 4.58 | 93.00 | 92.94 | 92.97 |

The last row is the ImageWAM Q-WAM row of Table 1; the middle rows are Table 3. Each number is one
run of 5,000 episodes. The shipped files reproduce the evaluated checkpoints exactly, but the
action sampler draws unseeded noise, so a rerun of the evaluation differs by sampling variation.

## Files

| Path | Purpose |
|---|---|
| `qwam/imagewam/export.py` | target selection, per-Linear smoothing, rotation, ASP deflation and INT4 packing, checkpoint meta |
| `qwam/imagewam/runtime.py` | `W4A4PackedLinear` and `install_w4a4` (checkpoint checks, module swap) |
| `qwam/imagewam/calib.py` | absmax hooks and shard merge |
| `scripts/imagewam/setup_imagewam.sh` | patch, policy link and `.env.local` for an ImageWAM checkout |
| `scripts/imagewam/calibrate_absmax.sh`, `merge_absmax.py` | smoothing calibration |
| `scripts/imagewam/build_asp_subspaces.py` | action-output Gram and ASP subspaces |
| `scripts/imagewam/export_qwam.py` | W4A4 export (paper defaults, Table 3 switches) |
| `scripts/imagewam/run_imagewam_robotwin.sh` | RoboTwin launcher (bf16 or a checkpoint) |
| `scripts/imagewam/run_qwam.sh`, `run_bf16.sh` | clean and randomized runs of one configuration |
| `scripts/imagewam/read_results.py` | success rates from the episode videos |
| `artifacts/imagewam/imagewam_act_absmax_c50.pt` | calibration absmax used in the paper (154 Linears) |
| `artifacts/imagewam/imagewam_asp_subspaces_r32.pt` | rank-32 ASP subspaces used in the paper |
| `patches/imagewam/0001-robotwin-policy-qwam-hooks.patch` | Q-WAM hooks in the ImageWAM RoboTwin policy |
