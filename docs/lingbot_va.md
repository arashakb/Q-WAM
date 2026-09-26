# LingBot-VA

Q-WAM on the released RoboTwin 2.0 LingBot-VA policy. LingBot-VA uses one shared transformer for
video and action, so ASP protects all 300 block Linears (30 blocks x {self-attention q, k, v, out;
cross-attention q, k, v, out; two FFN projections}; 4,907,335,680 weights).

## Upstream versions

| Component | Source | Version |
|---|---|---|
| LingBot-VA code | https://github.com/Robbyant/lingbot-va | `58c2ae5bac46bd8114065bea9d7d256eb67c16c3` |
| Weights | https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin | `8c9dea8abbc5c91cc9e18bc3264b8915083bbe70` |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin (nested at `$LINGBOT_ROOT/RoboTwin`) | `2eeec322d95799f537cbfe5f291a8220d965ccb8` |
| cuRobo | https://github.com/NVlabs/curobo (installed by RoboTwin's `script/_install.sh`) | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |

## Environment

One environment runs the inference server, the RoboTwin client and the Q-WAM scripts (the LingBot-VA
README setup): Python 3.10, torch 2.9.0+cu126, diffusers 0.36.0, transformers 4.55.2.

```bash
conda create -n lingbotva python=3.10.16 -y && conda activate lingbotva
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python \
    matplotlib ftfy easydict safetensors
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools      # RoboTwin rendering
```

The calibration frames are decoded once in the Fast-WAM environment (the RoboTwin 2.0 demonstrations
are AV1 videos), see step 0 below.

## Setup

```bash
export LINGBOT_ROOT=/path/to/lingbot-va          # created if absent
bash scripts/lingbot_va/setup_lingbot.sh          # pinned code + patches, RoboTwin + install edits, weights
cd $LINGBOT_ROOT/RoboTwin
bash script/_install.sh                           # RoboTwin requirements, pytorch3d, sapien/mplib fixes, cuRobo
bash script/_download_assets.sh
git -C envs/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb \
    && pip install -e envs/curobo --no-build-isolation   # optional: the cuRobo commit of the paper runs
```

`setup_lingbot.sh` applies the patches in `patches/lingbot_va/` (diffs against the pinned blobs):

| Patch | Change |
|---|---|
| `wan_va_server.patch` | after `_configure_model`, installs the checkpoint named by `LB_QUANT_CKPT` with `qwam.lingbot_va.runtime.install_w4a4` (Q-WAM repository on `PYTHONPATH`); unset, the server is unchanged |
| `va_robotwin_cfg.patch` | weight directory from `$LINGBOT_CKPT`, default `$LINGBOT_ROOT/checkpoints/lingbot-va-posttrain-robotwin` |
| `eval_polict_client_openpi.patch` | RoboTwin location from `$ROBOTWIN_ROOT`, default `$LINGBOT_ROOT/RoboTwin` |
| `robotwin_install.patch` | the two RoboTwin install edits prescribed by the LingBot-VA README (no torch pin, `huggingface_hub==0.36.2`, pytorch3d with `--no-build-isolation`) |

## Pipeline

Commands run from the Q-WAM repository root with `LINGBOT_ROOT` exported and the environment active.

**0. Calibration frames** (shared with ImageWAM; Fast-WAM environment): 50 episodes, one random
episode per task (seed 42), all 10,919 frames, each camera at 320x240.

```bash
export C50_RAW=work/c50_frames
FASTWAM_ROOT=/path/to/FastWAM python scripts/common/dump_c50_frames.py --stride 1 --wh 320x240 --out-dir $C50_RAW
```

**1. Activation absmax** for the smoothing factors. Shipped as
`artifacts/lingbot_va/lingbot_va_act_absmax.pt` (the file used for the paper). To recompute: 7 shards,
one GPU each, then merge (per-channel maximum over all tokens, denoising steps and frames).

```bash
for i in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES=$i python scripts/lingbot_va/calibrate_absmax.py --raw-dir $C50_RAW --nshard 7 --shard $i &
done; wait
python scripts/lingbot_va/merge_absmax.py --expect 7   # -> work/lingbot_va/lingbot_va_act_absmax.pt
```

Later steps read the shipped file by default; pass `--absmax work/lingbot_va/lingbot_va_act_absmax.pt`
(or `ABSMAX=...` for `run_qwam.sh`) to use a recomputed one. Rerunning a shard keeps an existing output
unless `--overwrite` is given; an interrupted shard leaves only a `.partial.pt` snapshot.

**2. ASP subspaces** (rank 32). The file used for the paper is shipped as
`artifacts/lingbot_va/lingbot_va_asp_subspaces_r32.pt` (81 MB) and is the default of the later steps.
To recompute one (one 80 GB GPU, about 1 h and 59 GB peak memory):

```bash
python scripts/lingbot_va/build_asp_subspaces.py --raw-dir $C50_RAW
# -> work/lingbot_va/lingbot_va_asp_subspaces_r32.pt
```

The estimator (`qwam/lingbot_va/aog.py`) runs 4 evenly spaced frames of each calibration episode
(200 frames). The server's sampler runs without gradients, so every action-mode transformer call of a
frame (the action denoising steps and the final cache-update call) is differentiated on its own: one
Gaussian probe of the call's output is backpropagated to the input of every block Linear, and the
contributions of all calls and frames are summed into a Nystrom sketch with 64 columns. The basis is
formed in the smoothed and rotated coordinates of the export (alpha 0.5, Hadamard block <= 1024) and
records the digest of the absmax it was built with; the exporter refuses a subspace file whose alpha,
Hadamard cap or absmax digest differ from the export settings. The probes come from the default CUDA
generator and the forward/backward passes run on the GPU, so a rebuild is close to, but not bit-identical
with, the subspace file of the paper.

**3. Export** (CPU, about 15 GB of RAM, one to two minutes):

```bash
python scripts/lingbot_va/export_qwam.py --out work/lingbot_va/lingbot_va_w4a4_qwam.pt
```

Defaults are the paper configuration: W4A4, weight and activation group 32, smoothing with alpha 0.5,
block Hadamard rotation (block = largest power of two dividing the input width, at most 1024), ASP
rank 32 from `artifacts/lingbot_va/lingbot_va_asp_subspaces_r32.pt` (`--subspaces` selects another file). The Table 3 rows switch components off:
`--subspaces none` (smoothing and rotation) and `--subspaces none --no-smooth --no-rotate` (per-group
W4A4). With the shipped absmax and the paper's subspace file the exporter reproduces the evaluated
checkpoints bit for bit.

**4. Evaluation** on RoboTwin 2.0 (8 GPUs, 50 tasks, clean and randomized; about a day per condition
on 8 L40S GPUs). `run_qwam.sh` exports the checkpoint if it does not exist, then runs both conditions:

```bash
bash scripts/lingbot_va/run_qwam.sh                       # Q-WAM (Table 1)
VARIANT=smoothrot bash scripts/lingbot_va/run_qwam.sh     # + smoothing and rotation (Table 3)
VARIANT=pergroup  bash scripts/lingbot_va/run_qwam.sh     # per-group W4A4 (Table 3)
bash scripts/lingbot_va/run_bf16.sh                       # bf16 reference
```

Both call `scripts/lingbot_va/run_8gpu_queue.sh` (one inference server per GPU, a work queue over the
50 tasks, `GPU_IDS`/`NGPU` select the GPUs). Episodes are written to
`$LINGBOT_ROOT/RoboTwin/results_<tag>/stseed-10000/visualization/<task>/<idx>_<instruction>_<True|False>.mp4`.
Rerunning a command resumes: complete tasks are skipped, partial ones restarted. The launcher only
ever stops processes it started itself.

**5. Success rates**:

```bash
python scripts/lingbot_va/read_results.py -n 50 lingbot_qwam_clean lingbot_qwam_randomized
```

## Expected results

| Method | BPW | Clean | Randomized | Avg. |
|---|---|---|---|---|
| bf16 | 16.00 | 90.84 | 90.20 | 90.52 |
| per-group W4A4 | 4.50 | 83.44 | 78.24 | 80.84 |
| + smoothing and rotation | 4.50 | 87.52 | 82.64 | 85.08 |
| + ASP (Q-WAM) | 4.76 | 90.20 | 88.92 | 89.56 |

BPW counts the 4-bit codes, one 16-bit scale per group of 32 weights and the 16-bit ASP factors
over the 300 block Linears:
BPW = [sum(in x out x 4 + (in / 32) x out x 16) + sum(32 x (in + out) x 16)] / 4,907,335,680 = 4.7628
(4.50 without ASP). The exporter writes it to the checkpoint metadata (`bpw_with_asp_branch`).

## Runtime

`qwam/lingbot_va/runtime.py` replaces each block Linear by `W4A4PackedLinear`, which keeps the packed
INT4 weight and its fp16 group scales resident and computes, with `x_r = FWHT(x / s)`,
`y = Q_a(x_r - p V^T) W_q^T + p (W_r V)^T + b` where `p = x_r V`, `W_q` is the dequantized INT4 weight
(the smoothed, rotated and deflated weight) and `Q_a` is symmetric per-token activation quantization
with group 32. The ASP branch stays in the model dtype (bf16) and never passes through `Q_a`. The GEMM
itself runs in bf16 on the dequantized operands (simulated quantization).
