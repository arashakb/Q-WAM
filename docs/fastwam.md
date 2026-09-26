# Q-WAM on Fast-WAM

Fast-WAM is a Mixture-of-Transformers world action model with a 5B video expert (Wan2.2-TI2V-5B)
and a 1B action expert. Q-WAM quantizes all 600 block linears of both experts (attention and
cross-attention q/k/v/o, FFN) to W4A4 with groups of 32, SmoothQuant smoothing (alpha 0.5) and a
block Hadamard rotation, and adds Action-Subspace Protection (ASP, rank 32) to the 300 linears of
the action expert, the expert with the larger action mass. Embeddings, heads and norms stay in bf16.

Quantization is simulated: weights and activations are rounded to the INT4 grid and dequantized,
and the GEMMs run in bf16. The code reproduces the accuracy results; it is not an INT4 kernel
implementation and its speed is not representative.

| Path | Purpose |
| --- | --- |
| `qwam/fastwam/asp_linear.py` | `ASPLinear`, the simulated W4A4 layer with smoothing, rotation and ASP |
| `qwam/fastwam/install.py` | `QWAMConfig` (paper defaults), `install_qwam`, `install_from_env` (policy hook) |
| `qwam/fastwam/harness.py` | loads the released checkpoint outside the simulator, for calibration |
| `patches/fastwam/deploy_policy.patch` | `QWAM_ENABLE` hook in FastWAM's RoboTwin policy |
| `scripts/fastwam/setup_fastwam.sh` | applies the patch and links the policy into RoboTwin |
| `scripts/fastwam/calibrate_absmax.py`, `merge_absmax.py` | activation absmax (smoothing) |
| `scripts/fastwam/estimate_aog.py`, `merge_aog.py` | Action Observability Gramians (AOGs) of the action expert |
| `scripts/fastwam/run_calibration.sh` | runs and merges all calibration shards on the GPUs of one machine |
| `scripts/fastwam/run_qwam.sh`, `run_bf16.sh`, `run_robotwin_eval.sh` | RoboTwin 2.0 evaluation |
| `scripts/fastwam/success_rate.py` | average success rate of a run |
| `scripts/fastwam/bpw.py` | bits per weight from the checkpoint's layer shapes |
| `scripts/fastwam/action_mass.py` | action mass per expert (optional) |

## Requirements

- FastWAM at commit `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
  (<https://github.com/yuantianyuan01/FastWAM>). It vendors RoboTwin 2.0 in `third_party/RoboTwin`
  (RoboTwin-Platform/RoboTwin `bf44be51cf5717a5595ce59447f2cf5263d2aa95`).
- FastWAM's environment (Python 3.10, PyTorch 2.7.1 with CUDA 12.8, `pip install -e .`), plus the
  RoboTwin simulator environment and assets installed as FastWAM's README describes.
- GPUs with 48 GB of memory. One evaluation worker or one calibration shard runs per GPU; an AOG
  shard peaks at about 43 GB.

## Setup

1. Follow FastWAM's README, from the FastWAM root:
   - *Environment Setup* and *Model Preparation*. The Wan2.2 T5 encoder and VAE are downloaded on
     first use into `$DIFFSYNTH_MODEL_BASE_PATH` (our scripts default it to `$FASTWAM_ROOT/checkpoints`;
     set `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` to download from Hugging Face instead of ModelScope).
     Our runs used `redirect_common_files: false` in `configs/model/fastwam.yaml`, which loads these
     two components from `Wan-AI/Wan2.2-TI2V-5B` rather than the converted copies FastWAM downloads
     by default.
   - *Inference with Released Checkpoints*: download `robotwin_uncond_3cam_384.pt` and
     `robotwin_uncond_3cam_384_dataset_stats.json` from <https://huggingface.co/yuanty/fastwam> into
     `checkpoints/fastwam_release/`, and install the RoboTwin environment and assets. RoboTwin's
     `task_config/` directory (the task list `_eval_step_limit.yml`, `demo_clean.yml`,
     `demo_randomized.yml` and the camera and embodiment configs) is not tracked in FastWAM's copy;
     copy it from the RoboTwin repository into `third_party/RoboTwin/task_config/`.
   - *Dataset Download / RoboTwin* (needed for calibration only): the RoboTwin 2.0 demonstrations
     from <https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam>, extracted to
     `data/robotwin2.0/robotwin2.0/`.
2. Apply the policy hook and link the policy into RoboTwin:

   ```bash
   export FASTWAM_ROOT=/path/to/FastWAM
   bash scripts/fastwam/setup_fastwam.sh
   ```

   The patch adds seven lines to `get_model()` in
   `experiments/robotwin/fastwam_policy/deploy_policy.py`: with `QWAM_ENABLE=1`, the bf16 model is
   quantized by `qwam.fastwam.install_from_env` after the checkpoint is loaded. Without it the
   policy is unchanged.

All commands below run from the root of this repository, in FastWAM's environment, with
`FASTWAM_ROOT` set. Calibration outputs go to `work/fastwam/` (`QWAM_WORK` overrides it).

## Calibration

The calibration set is one randomly chosen demonstration episode per RoboTwin task (50 tasks,
seed 42) with every frame, 10,919 frames in total (`qwam/robotwin_calib.py`). Observations are
built as the RoboTwin policy builds them, and every frame runs Fast-WAM's inference (10 denoising
steps, action horizon 32).

```bash
bash scripts/fastwam/run_calibration.sh          # both stages below; GPUS="0 1 ..." selects GPUs
```

1. **Activation absmax** (8 shards, about 1.3 s per frame): the per-input-channel maximum of |x| of
   every block linear, which sets the smoothing factor.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/fastwam/calibrate_absmax.py --shard 0 --num-shards 8 \
       --out-dir work/fastwam/absmax_shards          # one process per shard 0..7
   python scripts/fastwam/merge_absmax.py --shard-dir work/fastwam/absmax_shards --out work/fastwam/absmax.pt
   ```

2. **AOGs of the action expert** (11 shards, about 25 s per frame, 10 GB per shard file): for each
   of the 300 action-expert linears, the dense Gramian G = E[J^T J] of the generated action with
   respect to the layer input, estimated with 12 Gaussian probes per frame through the unrolled
   denoising loop. ASP protects the top-32 eigenvectors of H diag(s) G diag(s) H.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/fastwam/estimate_aog.py --shard 0 --num-shards 11 \
       --out-dir work/fastwam/aog_shards             # one process per shard 0..10
   python scripts/fastwam/merge_aog.py --shard-dir work/fastwam/aog_shards --out work/fastwam/aog_action.pt
   ```

   Probe seeds depend on the shard index, so `--num-shards 11` reproduces the probes of the released
   AOG; GPU kernels are not bitwise deterministic, so a recomputed AOG agrees with it up to
   floating-point noise.

## Evaluation

RoboTwin 2.0, 50 tasks, 100 episodes per task, unseen instructions (FastWAM's defaults), in the
clean and randomized settings. `run_robotwin_eval.sh` runs one task per GPU at a time (`GPUS`
selects the GPUs), writes a log per task, and skips tasks that already have a complete result when
restarted. A full setting takes about 8 to 10 hours on 8 GPUs.

```bash
# Q-WAM (paper configuration)
bash scripts/fastwam/run_qwam.sh clean
bash scripts/fastwam/run_qwam.sh randomized
# bf16 reference
bash scripts/fastwam/run_bf16.sh clean
bash scripts/fastwam/run_bf16.sh randomized
# component ablation (Table 3)
QWAM_RANK=0 bash scripts/fastwam/run_qwam.sh clean                              # + smoothing and rotation
QWAM_RANK=0 QWAM_ROTATE=0 QWAM_SMOOTH=0 bash scripts/fastwam/run_qwam.sh clean  # per-group W4A4
```

`run_qwam.sh` reads `work/fastwam/absmax.pt` and `work/fastwam/aog_action.pt` (or `QWAM_ABSMAX`,
`QWAM_AOG`). The switches `QWAM_RANK` (default 32; 0 removes ASP), `QWAM_ROTATE` and `QWAM_SMOOTH`
(default 1) are the only settings; everything else is fixed to the paper configuration in
`qwam.fastwam.QWAMConfig`. The `[qwam]` lines of each task log record the installed configuration.

### Reading the results

Each run writes to
`$FASTWAM_ROOT/evaluate_results/robotwin/robotwin_uncond_3cam_384/<tag>/`, with tag `qwam_clean`,
`qwam_randomized`, `bf16_clean`, ..., `qwam_r0_clean` and `qwam_r0_norot_nosmooth_clean` for the
ablation. RoboTwin writes one file per task, `<task>/_result_clean.txt` or
`<task>/_result_random.txt`, whose last line is the task's success fraction over its 100 episodes.
The success rate of a setting is the mean over the 50 tasks, and the average in the tables is the
mean of the clean and randomized rates:

```bash
python scripts/fastwam/success_rate.py $FASTWAM_ROOT/evaluate_results/robotwin/robotwin_uncond_3cam_384/qwam_clean
```

### Expected results

| Configuration | BPW | Clean | Randomized | Avg. |
| --- | --- | --- | --- | --- |
| bf16 | 16.00 | 92.28 | 91.34 | 91.81 |
| per-group W4A4 | 4.50 | 82.60 | 81.80 | 82.20 |
| + smoothing and rotation | 4.50 | 87.78 | 87.32 | 87.55 |
| **Q-WAM** (+ ASP) | **4.62** | **91.62** | **89.94** | **90.78** |

Closed-loop success rates vary by a few tenths of a point between repeated runs of the same
configuration.

## Bits per weight

```bash
python scripts/fastwam/bpw.py      # BPW = 4.6220
```

Over the 600 quantized linears (5,913,968,640 weights) Q-WAM stores INT4 weights (4 bits), one
16-bit scale per group of 32 weights (0.5), a 16-bit smoothing vector and bias per layer (0.0103),
and, on the 300 action-expert linears, the 16-bit ASP basis V and W~V, 32 (d_in + d_out) values per
layer (0.1117).

## Action mass (optional)

The action mass of an expert, mu_E = sum over its layers of tr(H diag(s) G diag(s) H), decides
which expert receives ASP. It needs the trace and diagonal of the video-expert AOGs as well, which
the quantizer does not use:

```bash
bash scripts/fastwam/run_calibration.sh video-mass     # 8 shards, 8 probes per frame
python scripts/fastwam/action_mass.py --aog work/fastwam/aog_action.pt \
    --video-mass work/fastwam/aog_video_mass.pt --absmax work/fastwam/absmax.pt \
    --ckpt $FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```
