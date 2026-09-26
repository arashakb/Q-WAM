"""RoboTwin 2.0 calibration set shared by all three models.

Calibration uses one randomly chosen demonstration episode per task (50 tasks, numpy seed 42) and
every frame of each episode (10,919 frames in total). The demonstrations are the RoboTwin 2.0
LeRobot dataset released with FastWAM, read from a FastWAM checkout:

    $FASTWAM_ROOT/data/robotwin2.0/robotwin2.0          (override with ROBOTWIN_DATA_DIR)
    $FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json

Observations are built exactly as FastWAM's RoboTwin policy builds them at evaluation time:
  image   : head camera resized to 320x256, wrist cameras to 160x128 (PIL bilinear),
            stacked as [head ; left | right] -> [1, 3, 384, 320], scaled to [-1, 1]
  proprio : z-scored with the released dataset statistics, clamped to [-5, 5]
  prompt  : FastWAM's DEFAULT_PROMPT filled with the episode instruction

Episodes are stored contiguously, 550 per task in alphabetical task order, so
task_id = episode_index // 550.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _fastwam_root() -> Path:
    root = os.environ.get("FASTWAM_ROOT")
    if not root:
        raise EnvironmentError("Set FASTWAM_ROOT to a FastWAM checkout that holds the RoboTwin 2.0 dataset.")
    return Path(root).resolve()


PROJECT_ROOT = _fastwam_root() if os.environ.get("FASTWAM_ROOT") else None
DATA_DIR = Path(os.environ["ROBOTWIN_DATA_DIR"]) if os.environ.get("ROBOTWIN_DATA_DIR") else (
    PROJECT_ROOT / "data/robotwin2.0/robotwin2.0" if PROJECT_ROOT else None)
META_DIR = DATA_DIR / "meta" if DATA_DIR else None
RELEASED_STATS = (PROJECT_ROOT / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
                  if PROJECT_ROOT else None)
FALLBACK_STATS = PROJECT_ROOT / "data/robotwin2.0/dataset_stats.json" if PROJECT_ROOT else None

EPISODES_PER_TASK = 550
NUM_TASKS = 50
CHUNK_SIZE = 1000
VIDEO_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

if PROJECT_ROOT is not None and str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
except Exception:  # pragma: no cover - same string as FastWAM's DEFAULT_PROMPT
    DEFAULT_PROMPT = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )

# The calibration protocol: one random episode per task, all 50 tasks, fixed seed.
CALIB_SEED = 42
CALIB_PER_TASK = 1
CALIB_NUM_TASKS = NUM_TASKS


def _require_data():
    if DATA_DIR is None:
        _fastwam_root()   # raises with a clear message


def _chunk(ep: int) -> int:
    return ep // CHUNK_SIZE


def _parquet(ep: int) -> Path:
    _require_data()
    return DATA_DIR / f"data/chunk-{_chunk(ep):03d}/episode_{ep:06d}.parquet"


def _video(ep: int, key: str) -> Path:
    _require_data()
    return DATA_DIR / f"videos/chunk-{_chunk(ep):03d}/{key}/episode_{ep:06d}.mp4"


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Identical to the FastWAM RoboTwin policy's resize (PIL, RGB, bilinear)."""
    from PIL import Image

    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def _decode_frame(path: Path, frame_idx: int) -> np.ndarray:
    """Decode one RGB frame with torchcodec (the dataset videos are AV1-encoded)."""
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(path))
    n = dec.metadata.num_frames or (frame_idx + 1)
    tgt = max(0, min(frame_idx, n - 1))
    data = dec.get_frame_at(tgt).data  # [C,H,W] uint8 RGB
    return data.permute(1, 2, 0).contiguous().numpy()  # [H,W,3] uint8 RGB


def _build_image(head_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray) -> torch.Tensor:
    head = _resize_rgb(head_rgb, (320, 256))          # [256,320,3]
    left = _resize_rgb(left_rgb, (160, 128))          # [128,160,3]
    right = _resize_rgb(right_rgb, (160, 128))        # [128,160,3]
    bottom = np.concatenate([left, right], axis=1)    # [128,320,3]
    image = np.concatenate([head, bottom], axis=0)    # [384,320,3]
    t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,384,320]
    return t * (2.0 / 255.0) - 1.0                    # [-1,1], CPU float32


def _load_state_stats():
    _require_data()
    path = RELEASED_STATS if RELEASED_STATS.exists() else FALLBACK_STATS
    if not path.exists():
        raise FileNotFoundError(f"No dataset stats found at {RELEASED_STATS} or {FALLBACK_STATS}.")
    stats = json.load(open(path))
    s = stats["state"]["default"]

    def pick(*names):
        for n in names:
            if n in s:
                return np.asarray(s[n], dtype=np.float32)
        raise KeyError(f"none of {names} in state stats keys {list(s)}")

    return pick("global_mean", "mean"), pick("global_std", "std"), str(path)


def _zscore(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    # FastWAM's z-score normalizer: (x - mean) / (std + 1e-8), clamped to [-5, 5]
    x = (state.astype(np.float32) - mean) / (std + 1e-8)
    x = np.clip(x, -5.0, 5.0)
    return torch.from_numpy(x).float().unsqueeze(0)  # [1,14]


def select_episodes_random(per_task: int = CALIB_PER_TASK, num_tasks: int = CALIB_NUM_TASKS,
                           seed: int = CALIB_SEED) -> list[int]:
    """Sample `per_task` episodes from each task with a dedicated numpy Generator seeded by `seed`.

    The returned list is a pure function of (per_task, num_tasks, seed).
    """
    rng = np.random.default_rng(seed)
    eps: list[int] = []
    for t in range(num_tasks):
        base = t * EPISODES_PER_TASK
        offs = rng.choice(EPISODES_PER_TASK, size=per_task, replace=False)
        eps.extend(int(base + o) for o in sorted(offs))
    return eps


def select_episodes(per_task: int = 5, num_tasks: int = NUM_TASKS, spread: bool = True) -> list[int]:
    """Deterministic, evenly spread episode choice (used only by the per-layer damage analysis)."""
    eps: list[int] = []
    for t in range(num_tasks):
        base = t * EPISODES_PER_TASK
        if spread and per_task > 1:
            offs = [int(round(i * (EPISODES_PER_TASK - 1) / (per_task - 1))) for i in range(per_task)]
        else:
            offs = list(range(per_task))
        eps.extend(base + o for o in offs)
    return eps


_INSTR_CACHE: dict | None = None


def _load_instructions() -> dict:
    global _INSTR_CACHE
    if _INSTR_CACHE is None:
        _require_data()
        d: dict = {}
        with open(META_DIR / "episodes.jsonl") as f:
            for line in f:
                e = json.loads(line)
                d[e["episode_index"]] = (e.get("tasks") or ["Do the manipulation task."])[0]
        _INSTR_CACHE = d
    return _INSTR_CACHE


def build_observations(episodes: list[int], frame_frac: float = 0.3, verbose: bool = True) -> list[dict]:
    """One observation per episode, at `frame_frac` of its length."""
    import pandas as pd

    mean, std, stats_path = _load_state_stats()
    instr = _load_instructions()
    out: list[dict] = []
    for k, ep in enumerate(episodes):
        df = pd.read_parquet(_parquet(ep), columns=["observation.state"])
        n = len(df)
        fidx = int(frame_frac * (n - 1))
        state = np.asarray(df["observation.state"].iloc[fidx], dtype=np.float32)[:14]
        cams = [_decode_frame(_video(ep, key), fidx) for key in VIDEO_KEYS]
        out.append(dict(
            image=_build_image(*cams),
            proprio=_zscore(state, mean, std),
            prompt=DEFAULT_PROMPT.format(task=instr.get(ep, "Do the manipulation task.")),
            task_id=ep // EPISODES_PER_TASK, episode=ep, frame=fidx,
        ))
        if verbose and (k % 25 == 0 or k == len(episodes) - 1):
            print(f"[calib] {k + 1}/{len(episodes)} ep={ep} task={ep // EPISODES_PER_TASK} frame={fidx}/{n}")
    if verbose:
        print(f"[calib] state stats from {stats_path}; built {len(out)} observations")
    return out


def build_episode_allframes(ep: int, stride: int = 1, mean=None, std=None) -> list:
    """All (strided) frames of one episode as evaluation-faithful observation dicts."""
    import pandas as pd
    from torchcodec.decoders import VideoDecoder
    if mean is None:
        mean, std, _ = _load_state_stats()
    instr = _load_instructions()
    df = pd.read_parquet(_parquet(ep), columns=["observation.state"])
    states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
    n = len(states)
    idx = list(range(0, n, max(1, stride)))
    cams = []
    for key in VIDEO_KEYS:
        dec = VideoDecoder(str(_video(ep, key)))
        nf = dec.metadata.num_frames or n
        ii = [min(i, nf - 1) for i in idx]
        try:
            batch = dec.get_frames_at(ii).data  # [k,C,H,W] uint8
            cams.append([f.permute(1, 2, 0).contiguous().numpy() for f in batch])
        except Exception:
            cams.append([dec.get_frame_at(i).data.permute(1, 2, 0).contiguous().numpy() for i in ii])
    prompt = DEFAULT_PROMPT.format(task=instr.get(ep, "Do the manipulation task."))
    return [
        dict(image=_build_image(cams[0][j], cams[1][j], cams[2][j]),
             proprio=_zscore(states[min(fi, n - 1)], mean, std),
             prompt=prompt, task_id=ep // EPISODES_PER_TASK, episode=ep, frame=fi)
        for j, fi in enumerate(idx)
    ]
