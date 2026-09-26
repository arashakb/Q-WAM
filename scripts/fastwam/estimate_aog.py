"""Action Observability Gramians (AOGs) of Fast-WAM's action-expert linears (one shard).

For layer l with input x_l, the AOG is G_l = E_o[ sum_i J_{l,i}^T J_{l,i} ], where J_{l,i} is the
Jacobian of the generated action chunk a with respect to x_l at token-step pair i, taken through
all remaining denoising steps. Per calibration frame (the 10,919 frames of qwam.robotwin_calib):

  1. run inference once without grad, recording the text/proprio context and the video KV cache;
  2. re-run the 10 action denoising steps with autograd and the video KV cache held fixed, tapping
     the input of every action-expert linear at every step; each tap is a clone, so a tensor read
     by several layers (e.g. the context read by every cross-attention k/v) gives each layer the
     gradient through that layer only;
  3. for each of P probes u ~ N(0, I) in action space, one backward pass of <a, u> returns J^T u for
     every layer and token-step pair, and sum_i (J_i^T u)(J_i^T u)^T is accumulated in float64.

Since E[u u^T] = I, the probe average converges to G_l. Each shard stores the sum over its frames
and probes; merge_aog.py adds the shards and divides by the number of frames, so the stored AOG is
P times the estimate of the paper (the scale does not change the protected subspace). Probes are
seeded by (shard index, frame index within the shard), so the estimate is reproducible for a fixed
--num-shards; the released numbers used --num-shards 11.

--mode video-mass attaches the video prefill to the graph (the video tokens and the text context
are made leaves), taps the video-expert linears and keeps only the trace and the diagonal of their
AOGs. It is used by action_mass.py for the per-expert action mass and never by the quantizer.

  FASTWAM_ROOT=/path/to/FastWAM CUDA_VISIBLE_DEVICES=0 \
  python scripts/fastwam/estimate_aog.py --shard 0 --num-shards 11 --out-dir work/fastwam/aog_shards

Action mode needs about 43 GB of GPU memory and 25 s per frame at P=12 on an L40S.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam import robotwin_calib as C  # noqa: E402
from qwam.fastwam import target_linears  # noqa: E402
from qwam.fastwam.harness import ACTION_HORIZON, NUM_INFERENCE_STEPS  # noqa: E402

DEFAULT_PROBES = {"action": 12, "video-mass": 8}


def tap_hook(store: dict, name: str):
    """Forward pre-hook routing a layer's input through a clone that retains its gradient."""
    def hook(module, inputs):
        x = inputs[0]
        if x.requires_grad:
            x = x.clone()
            x.retain_grad()
            store[name] = x
            return (x,) + tuple(inputs[1:])
    return hook


def capture(model, obs, device, num_steps, action_horizon) -> dict:
    """One no-grad inference; records the prefill inputs, the action-step context and the KV cache."""
    cap = {}
    prefill, predict = model.mot.prefill_video_cache, model._predict_action_noise_with_cache

    def spy_prefill(**kw):
        if "prefill" not in cap:
            cap["prefill"] = dict(kw)
        return prefill(**kw)

    def spy_predict(**kw):
        if "ctx" not in cap:
            cap.update(ctx=kw["context"], ctx_mask=kw["context_mask"], attn=kw["attention_mask"],
                       vseq=kw["video_seq_len"],
                       kv=[{k: v.detach().clone() for k, v in layer.items()} for layer in kw["video_kv_cache"]])
        return predict(**kw)

    model.mot.prefill_video_cache, model._predict_action_noise_with_cache = spy_prefill, spy_predict
    try:
        with torch.no_grad():
            model.infer_action(prompt=obs["prompt"], input_image=obs["image"].to(device),
                               action_horizon=action_horizon, proprio=obs["proprio"].to(device),
                               num_inference_steps=num_steps, seed=0, rand_device="cpu")
    finally:
        model.mot.prefill_video_cache, model._predict_action_noise_with_cache = prefill, predict
    return cap


def unroll(model, cap, kv_cache, tapped: dict, device, num_steps, action_horizon):
    """The action denoising loop under autograd; returns the final chunk and per-step input taps.

    The initial noise is the one inference draws with seed 0.
    """
    predict = model._predict_action_noise_with_cache.__func__.__wrapped__   # without no_grad
    sched = model.infer_action_scheduler
    g = torch.Generator(device="cpu").manual_seed(0)
    latents = torch.randn((1, action_horizon, int(model.action_expert.action_dim)), generator=g,
                          device="cpu").to(device, model.torch_dtype).requires_grad_(True)
    timesteps, deltas = sched.build_inference_schedule(num_steps, device, dtype=latents.dtype)
    taps, cur = [], latents
    for t, delta in zip(timesteps, deltas):
        step_taps = {}
        hooks = [m.register_forward_pre_hook(tap_hook(step_taps, n)) for n, m in tapped.items()]
        tt = t.unsqueeze(0).to(dtype=cur.dtype, device=device)
        pred = predict(model, latents_action=cur, timestep_action=tt, context=cap["ctx"],
                       context_mask=cap["ctx_mask"], video_kv_cache=kv_cache,
                       attention_mask=cap["attn"], video_seq_len=cap["vseq"])
        for h in hooks:
            h.remove()
        taps.append(step_taps)
        cur = sched.step(pred, delta, cur)
    return cur[0], taps


def accumulate_action(model, cap, gen, nprobe, aog, action_layers, device, num_steps, action_horizon):
    """Adds sum_p sum_i (J_i^T u_p)(J_i^T u_p)^T of one frame to aog[layer]."""
    kv = [{k: v.detach() for k, v in layer.items()} for layer in cap["kv"]]
    with torch.enable_grad():
        a_final, taps = unroll(model, cap, kv, action_layers, device, num_steps, action_horizon)
    for p in range(nprobe):
        for step in taps:
            for x in step.values():
                x.grad = None
        u = torch.randn(a_final.shape, generator=gen, device=device, dtype=torch.float32)
        (a_final.float() * u).sum().backward(retain_graph=(p < nprobe - 1))
        for step in taps:
            for n, x in step.items():
                if x.grad is not None:
                    gx = x.grad.double().reshape(-1, x.shape[-1])
                    aog[n] += gx.t() @ gx
    del taps, a_final, kv


def accumulate_video_mass(model, cap, gen, nprobe, tr, diag, video_layers, device, num_steps,
                          action_horizon):
    """Adds the trace and diagonal of one frame's video-expert AOGs to tr[layer] and diag[layer]."""
    taps = {}
    with torch.enable_grad():
        kw = dict(cap["prefill"])
        kw["video_tokens"] = kw["video_tokens"].detach().clone().requires_grad_(True)
        # The video cross-attention k/v read the text context, so it must be a leaf as well.
        payload = kw.get("video_context_payload")
        if not isinstance(payload, dict):
            raise TypeError(f"expected a dict video_context_payload, got {type(payload).__name__}")
        payload = {k: (v.detach().clone().requires_grad_(True)
                       if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in payload.items()}
        if not any(torch.is_tensor(v) and v.requires_grad for v in payload.values()):
            raise RuntimeError("video_context_payload holds no floating-point tensor")
        kw["video_context_payload"] = payload
        hooks = [m.register_forward_pre_hook(tap_hook(taps, n)) for n, m in video_layers.items()]
        kv = model.mot.prefill_video_cache(**kw)
        for h in hooks:
            h.remove()
        a_final, _ = unroll(model, cap, kv, {}, device, num_steps, action_horizon)
    for p in range(nprobe):
        for x in taps.values():
            x.grad = None
        u = torch.randn(a_final.shape, generator=gen, device=device, dtype=torch.float32)
        (a_final.float() * u).sum().backward(retain_graph=(p < nprobe - 1))
        for n, x in taps.items():
            if x.grad is not None:
                sq = x.grad.double().reshape(-1, x.shape[-1]).pow(2)
                diag[n] += sq.sum(0)
                tr[n] += sq.sum()
    del taps, a_final, kv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=sorted(DEFAULT_PROBES), default="action")
    ap.add_argument("--nprobe", type=int, default=None, help="probes per frame (12 action, 8 video-mass)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None, help="checkpoint (default: the released RoboTwin checkpoint)")
    ap.add_argument("--seed", type=int, default=C.CALIB_SEED, help="episode-sampling seed")
    ap.add_argument("--per-task", type=int, default=C.CALIB_PER_TASK, help="episodes per task")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (1 = every frame)")
    ap.add_argument("--num-steps", type=int, default=NUM_INFERENCE_STEPS, help="denoising steps")
    ap.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    args = ap.parse_args()
    nprobe = args.nprobe or DEFAULT_PROBES[args.mode]
    dev = args.device
    S, AH = args.num_steps, args.action_horizon

    from qwam.fastwam.harness import load_model
    os.makedirs(args.out_dir, exist_ok=True)
    eps = C.select_episodes_random(per_task=args.per_task, num_tasks=C.NUM_TASKS, seed=args.seed)
    mine = eps[args.shard::args.num_shards]
    tag = f"[aog {args.mode} {args.shard}/{args.num_shards}]"
    print(f"{tag} {len(eps)} episodes (seed {args.seed}), {len(mine)} on this shard: {mine}", flush=True)

    model = load_model(device=dev, ckpt=args.ckpt)
    targets = target_linears(model)
    action = {n: m for n, m in targets.items() if n.startswith("action_expert")}
    video = {n: m for n, m in targets.items() if not n.startswith("action_expert")}
    if args.mode == "action":
        aog = {n: torch.zeros((m.in_features, m.in_features), dtype=torch.float64, device=dev)
               for n, m in action.items()}
    else:
        tr = {n: torch.zeros((), dtype=torch.float64, device=dev) for n in video}
        diag = {n: torch.zeros(m.in_features, dtype=torch.float64, device=dev) for n, m in video.items()}
    mean, std, _ = C._load_state_stats()

    t0, n_frames = time.time(), 0
    for k, ep in enumerate(mine):
        for o in C.build_episode_allframes(ep, stride=args.stride, mean=mean, std=std):
            cap = capture(model, o, dev, S, AH)
            gen = torch.Generator(device=dev).manual_seed(1_000_003 * args.shard + n_frames)
            if args.mode == "action":
                accumulate_action(model, cap, gen, nprobe, aog, action, dev, S, AH)
            else:
                accumulate_video_mass(model, cap, gen, nprobe, tr, diag, video, dev, S, AH)
            del cap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n_frames += 1
        el = time.time() - t0
        peak = f", peak {torch.cuda.max_memory_allocated(dev) / 1e9:.1f} GB" if dev.startswith("cuda") else ""
        print(f"{tag} {k + 1}/{len(mine)} episode {ep} (task {ep // C.EPISODES_PER_TASK}), "
              f"{n_frames} frames ({el:.0f}s, {el / max(n_frames, 1):.1f} s/frame{peak})", flush=True)

    out = {"mode": args.mode, "n_frames": n_frames,
           "protocol": {"seed": args.seed, "per_task": args.per_task, "num_tasks": C.NUM_TASKS,
                        "stride": args.stride, "num_steps": S, "action_horizon": AH, "nprobe": nprobe,
                        "shard": args.shard, "num_shards": args.num_shards, "episodes": mine,
                        "all_episodes": eps}}
    if args.mode == "action":
        out["aog_sum"] = {n: g.cpu() for n, g in aog.items()}
    else:
        out["tr_sum"] = {n: float(v) for n, v in tr.items()}
        out["diag_sum"] = {n: v.cpu() for n, v in diag.items()}
    path = os.path.join(args.out_dir, f"aog_shard{args.shard}.pt")
    torch.save(out, path)
    print(f"{tag} done: {n_frames} frames -> {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
