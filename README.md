# Q-WAM: 4-Bit Quantization of World Action Models

Q-WAM is a post-training W4A4 quantization method for World Action Models (WAMs), models that
generate robot actions together with future video through iterative denoising. This repository
contains the code to reproduce the Q-WAM results of the paper on three WAMs in the RoboTwin 2.0
simulator: **Fast-WAM**, **ImageWAM** and **LingBot-VA**.

## Method in brief

* **Action Observability Gramian (AOG).** For every linear layer, the AOG
  `G_l = E[ sum_i J_{l,i}^T J_{l,i} ]` (with `J` the Jacobian of the generated action with respect to
  the layer input) predicts how much a rounding error in each weighted combination of the layer's
  input channels changes the final action. It is estimated without labels from random probes
  `u ~ N(0, I)` in action space: one backward pass of `<a, u>` returns `J^T u` for every layer.
* **Action-Subspace Protection (ASP).** Each layer is smoothed (SmoothQuant, alpha = 0.5) and
  rotated with a block Hadamard transform. The AOG is expressed in these coordinates,
  `G~ = H diag(s) G diag(s) H`, and its top `r = 32` eigenvectors `V` span the layer's action
  subspace. The layer output is computed as a small 16-bit branch plus a deflated 4-bit path:

      W^T x = (V^T W~)^T (V^T x~) + Q4(W~_perp)^T Q4(x~_perp),   W~_perp = (I - VV^T) W~,  x~_perp = (I - VV^T) x~

  Weights and activations are quantized symmetrically in groups of 32 (activations per token at
  run time).
* **Where to protect.** In a Mixture-of-Transformers WAM, ASP is applied to the expert with the
  largest action mass `mu_E = sum_{l in E} tr(G~_l)`: the action expert of Fast-WAM and ImageWAM.
  LingBot-VA has one shared backbone, so ASP covers all of its layers. All other layers use the
  same smoothing, rotation and W4A4 quantization without the branch.

## Results (RoboTwin 2.0, 50 tasks, success rate in %)

| Model | Method | BPW | Clean | Randomized | Avg. | Mem (GB) |
|---|---|---|---|---|---|---|
| Fast-WAM | bf16 | 16.00 | 92.28 | 91.34 | 91.81 | 11.85 |
| Fast-WAM | Q-WAM (W4A4) | 4.62 | 91.62 | 89.94 | 90.78 | 3.44 |
| ImageWAM | bf16 | 16.00 | 92.82 | 93.70 | 93.26 | 9.11 |
| ImageWAM | Q-WAM (W4A4) | 4.58 | 93.00 | 92.94 | 92.97 | 2.69 |
| LingBot-VA | bf16 | 16.00 | 90.84 | 90.20 | 90.52 | 10.16 |
| LingBot-VA | Q-WAM (W4A4) | 4.76 | 90.20 | 88.92 | 89.56 | 3.27 |

BPW counts the 4-bit weights, their group scales and the 16-bit ASP branch over the quantized
transformer blocks; Mem is the memory of those blocks. Success rates in RoboTwin vary by about one
point between repeated evaluations of the same checkpoint, because the simulator and the GPU are
not fully deterministic.

## Repository layout

```
qwam/                       Python package
  quant.py                  symmetric group-wise quantization, SmoothQuant factor
  hadamard.py               block Hadamard rotation (fast Walsh-Hadamard transform)
  robotwin_calib.py         the RoboTwin 2.0 calibration set shared by all models
  fastwam/                  Fast-WAM: ASP linear layer and its installation into the policy
  imagewam/                 ImageWAM: calibration hooks, checkpoint export, W4A4 runtime
  lingbot_va/               LingBot-VA: calibration, AOG sketch, checkpoint export, W4A4 runtime
scripts/
  common/dump_c50_frames.py raw calibration frames for ImageWAM and LingBot-VA
  fastwam/  imagewam/  lingbot_va/
                            setup, calibration, AOG estimation, export, RoboTwin evaluation, results
patches/                    small patches that add the Q-WAM hooks to each upstream repository
artifacts/                  calibration results used for the paper (ImageWAM and LingBot-VA:
                            activation absmax and rank-32 ASP subspaces)
docs/                       detailed instructions for each model
```

The models themselves are not part of this repository. Each is used from its own upstream
repository, pinned to the commit the results were produced with; the setup scripts check out or
verify that commit and apply the patches in `patches/`.

| Model | Upstream repository | Commit |
|---|---|---|
| Fast-WAM | https://github.com/yuantianyuan01/FastWAM | `45d8e14` |
| ImageWAM | https://github.com/yuyangalin/ImageWAM | `00a4e7a` |
| LingBot-VA | https://github.com/Robbyant/lingbot-va | `58c2ae5` |

## Requirements

* Linux with NVIDIA GPUs. The evaluations run one RoboTwin task per GPU (48 GB GPUs such as the
  L40S were used). AOG estimation needs more memory: about 43 GB per shard for Fast-WAM and about
  60 GB for LingBot-VA (an 80 GB GPU).
* One Python environment per model, created as described in that model's upstream README
  (Python 3.10; the results were produced with PyTorch 2.9.0 and CUDA 12.6), plus RoboTwin 2.0 and
  its assets as described in the RoboTwin documentation.
* This repository on `PYTHONPATH`, or run the scripts from its root (they add it themselves).

### Calibration data (all models)

All three models are calibrated on the same frames: one randomly chosen demonstration episode per
RoboTwin task (50 tasks, numpy seed 42) and every frame of each episode, 10,919 frames in total.
The frames come from the RoboTwin 2.0 LeRobot dataset released with Fast-WAM. Download it and the
Fast-WAM release checkpoint as described in the Fast-WAM README, so that the checkout contains

```
$FASTWAM_ROOT/data/robotwin2.0/robotwin2.0/                         (dataset; or set ROBOTWIN_DATA_DIR)
$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

`FASTWAM_ROOT` must be set for every calibration step, also for ImageWAM and LingBot-VA. Those two
models cannot decode the dataset videos in their own environments, so the frames are first dumped
to disk once, in the Fast-WAM environment:

```bash
FASTWAM_ROOT=/path/to/FastWAM python scripts/common/dump_c50_frames.py --out-dir work/c50_frames
```

## Fast-WAM

Details: [docs/fastwam.md](docs/fastwam.md).

```bash
export FASTWAM_ROOT=/path/to/FastWAM          # checkout at 45d8e14, with the release checkpoint and dataset
bash scripts/fastwam/setup_fastwam.sh         # applies the policy hook, links the policy into RoboTwin

# 1. Calibration: activation absmax (8 shards) and action-expert AOGs (11 shards, 12 probes per frame),
#    one shard per GPU, then merged into work/fastwam/{absmax.pt,aog_action.pt}
GPUS="0 1 2 3 4 5 6 7" bash scripts/fastwam/run_calibration.sh all

# 2. Evaluation of Q-WAM (paper configuration), 50 tasks x 100 episodes per condition.
#    The quantizer is installed when the policy loads; the ASP subspaces are computed from the AOGs.
bash scripts/fastwam/run_qwam.sh clean all 100
bash scripts/fastwam/run_qwam.sh randomized all 100

# bf16 reference
bash scripts/fastwam/run_bf16.sh clean all 100
bash scripts/fastwam/run_bf16.sh randomized all 100

# Success rates (one result directory per run; bf16 runs are tagged bf16_<condition>)
python scripts/fastwam/success_rate.py $FASTWAM_ROOT/evaluate_results/robotwin/robotwin_uncond_3cam_384/qwam_clean
python scripts/fastwam/success_rate.py $FASTWAM_ROOT/evaluate_results/robotwin/robotwin_uncond_3cam_384/qwam_randomized
# Bits per weight
python scripts/fastwam/bpw.py --ckpt $FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

Component ablation (paper Table 3): `QWAM_RANK=0` gives "+ smoothing and rotation", and
`QWAM_RANK=0 QWAM_ROTATE=0 QWAM_SMOOTH=0` gives "per-group W4A4". The action mass of the two
experts can be measured with `scripts/fastwam/run_calibration.sh video-mass` followed by
`scripts/fastwam/action_mass.py`.

## ImageWAM

Details: [docs/imagewam.md](docs/imagewam.md).

```bash
export IMAGEWAM_ROOT=/path/to/ImageWAM        # checkout at 00a4e7a; FLUX.2 source, weights, RoboTwin assets per its README
FLUX2_SRC=/path/to/flux2 PYTHON_BIN=$(which python) \
  bash scripts/imagewam/setup_imagewam.sh     # policy patch, RoboTwin policy link, .env.local

# 1. Calibration (optional): the activation absmax and the rank-32 ASP subspace file used for the
#    paper are shipped in artifacts/imagewam/ and used by default. To recompute them, see
#    docs/imagewam.md (calibrate_absmax.sh, dump_c50_frames.py, build_asp_subspaces.py).

# 2. Export the W4A4 checkpoint and evaluate it, clean and randomized (50 tasks x 100 episodes)
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/imagewam/run_qwam.sh

# bf16 reference
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/imagewam/run_bf16.sh

# Success rates from the per-episode records (the clean and the randomized run directory)
python scripts/imagewam/read_results.py <clean run dir> <randomized run dir>
```

Component ablation (Table 3), each with its own checkpoint and run label:

```bash
IW_QUANT_CKPT=work/imagewam/imagewam_w4a4_smoothrot_g32.pt ARM_PREFIX=smoothrot \
  EXPORT_ARGS="--subspaces none" bash scripts/imagewam/run_qwam.sh
IW_QUANT_CKPT=work/imagewam/imagewam_w4a4_group_g32.pt ARM_PREFIX=group \
  EXPORT_ARGS="--no-smooth --no-rotate --subspaces none" bash scripts/imagewam/run_qwam.sh
```

## LingBot-VA

Details: [docs/lingbot_va.md](docs/lingbot_va.md).

```bash
export LINGBOT_ROOT=/path/to/lingbot-va
bash scripts/lingbot_va/setup_lingbot.sh      # pinned lingbot-va + RoboTwin, patches, post-trained weights

# 1. Activation absmax: shipped in artifacts/lingbot_va/lingbot_va_act_absmax.pt. To recompute it:
#    for i in 0..6: python scripts/lingbot_va/calibrate_absmax.py --raw-dir work/c50_frames --nshard 7 --shard $i
#    python scripts/lingbot_va/merge_absmax.py --expect 7

# 2. ASP subspaces: the rank-32 subspace file used for the paper is shipped in
#    artifacts/lingbot_va/lingbot_va_asp_subspaces_r32.pt and used by default. To recompute one
#    from the AOG sketch (one 80 GB GPU, about one hour):
#    python scripts/lingbot_va/build_asp_subspaces.py --raw-dir work/c50_frames

# 3. Export the packed W4A4 checkpoint, evaluate it clean and randomized, print the success rates
bash scripts/lingbot_va/run_qwam.sh

# bf16 reference
bash scripts/lingbot_va/run_bf16.sh

# Success rates can be re-read at any time
python scripts/lingbot_va/read_results.py lingbot_qwam_clean lingbot_qwam_randomized
```

Component ablation (Table 3): `VARIANT=smoothrot` or `VARIANT=pergroup` for `run_qwam.sh`.

## Notes

* The run scripts default to the paper configuration: W4A4 with weight and activation group 32,
  SmoothQuant alpha 0.5, block Hadamard with block size up to 1024, and ASP rank 32.
* Evaluation launchers run one task per GPU and resume unfinished runs; see each script's header
  for the GPU selection variables.
* The AOG is estimated from random probes. `artifacts/` holds the activation absmax and the ASP
  subspace files of ImageWAM and LingBot-VA used for the paper; with them, the ImageWAM and
  LingBot-VA exporters reproduce the evaluated checkpoints byte for byte. Recomputed subspaces are close to, but
  not bit-identical with, the shipped ones. For Fast-WAM the AOGs are recomputed with
  `run_calibration.sh`, whose probes are seeded per shard and frame.

## Acknowledgements

This code builds on [Fast-WAM](https://github.com/yuantianyuan01/FastWAM),
[ImageWAM](https://github.com/yuyangalin/ImageWAM),
[LingBot-VA](https://github.com/Robbyant/lingbot-va) and
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).
