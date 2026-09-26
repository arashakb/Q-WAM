"""Success rates of LingBot-VA RoboTwin 2.0 runs from the per-episode records.

  python scripts/lingbot_va/read_results.py [-n 50] [--root <RoboTwin>] [--per-task] ARM [ARM ...]

ARM is a run tag (read from <root>/results_<ARM>) or a results directory. The RoboTwin client writes
one <episode index>_<instruction>_<True|False>.mp4 per episode under stseed-*/visualization/<task>/.
The success rate of a task is the fraction of successes among episodes 0..n-1; the arm's rate is the
mean over the 50 tasks. Tasks with fewer than n episodes, a gap in the indices, or duplicated indices
are reported and left out of the mean. Arms named <x>_clean and <x>_randomized (or <x>_rand) are also
paired into the average of the two conditions.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

EP_RE = re.compile(r"^(\d+)_.*_(True|False)\.mp4$")
N_TASKS = 50


def scan(results_dir: Path, n: int):
    """-> (sr {task: rate}, partial {task: episodes}, bad {task: reason}, episodes run)."""
    by_task, dup = defaultdict(dict), defaultdict(set)
    for vis in sorted(results_dir.glob("stseed-*/visualization")):
        for tdir in sorted(p for p in vis.iterdir() if p.is_dir()):
            slot = by_task[tdir.name]
            for f in tdir.glob("*.mp4"):
                m = EP_RE.match(f.name)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx in slot:
                    dup[tdir.name].add(idx)
                slot[idx] = m.group(2) == "True"
    sr, partial, bad = {}, {}, {}
    for task, slot in sorted(by_task.items()):
        if dup[task]:
            bad[task] = f"{len(dup[task])} duplicated episode indices"
            continue
        missing = [i for i in range(min(len(slot), n)) if i not in slot]
        if missing:
            bad[task] = f"missing episode indices {missing[:4]}"
            continue
        if len(slot) < n:
            partial[task] = len(slot)
            continue
        sr[task] = sum(slot[i] for i in range(n)) / n
    return sr, partial, bad, sum(len(v) for v in by_task.values())


def condition(root: Path, tag: str):
    """demo_clean / demo_randomized as recorded by RoboTwin's own eval_result tree, if unambiguous."""
    found = {c for c in ("demo_clean", "demo_randomized")
             if any((root / "eval_result").glob(f"*/*/{c}/{tag}"))}
    return found.pop() if len(found) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("-n", type=int, default=50, help="episodes per task")
    ap.add_argument("--root", default=None,
                    help="RoboTwin checkout holding results_<ARM> "
                         "(default $ROBOTWIN_ROOT, else $LINGBOT_ROOT/RoboTwin)")
    ap.add_argument("--per-task", action="store_true", help="print the per-task table")
    ap.add_argument("--json", default=None, help="also write the results to this file")
    args = ap.parse_args()
    root = args.root or os.environ.get("ROBOTWIN_ROOT") or (
        str(Path(os.environ["LINGBOT_ROOT"]) / "RoboTwin") if os.environ.get("LINGBOT_ROOT") else ".")
    root = Path(root)

    out, rows = {}, {}
    for arm in args.arms:
        d = Path(arm) if Path(arm).is_dir() else root / f"results_{arm}"
        tag = d.name[len("results_"):] if d.name.startswith("results_") else d.name
        if not d.is_dir():
            print(f"{tag}: no results directory at {d}")
            continue
        sr, partial, bad, total = scan(d, args.n)
        cond = condition(d.parent, tag)
        mean = 100 * sum(sr.values()) / len(sr) if sr else float("nan")
        complete = len(sr) == N_TASKS
        print(f"{tag:32s} {cond or '(condition not recorded)':18s} {len(sr):2d}/{N_TASKS} tasks complete  "
              f"SR {mean:6.2f}" + ("" if complete else "  (INCOMPLETE: not a final number)"))
        for t, c in sorted(partial.items()):
            print(f"    partial: {t} has {c}/{args.n} episodes")
        for t, why in sorted(bad.items()):
            print(f"    excluded: {t}: {why}")
        out[tag] = {"condition": cond, "n": args.n, "complete_tasks": len(sr), "episodes": total,
                    "success_rate": mean, "per_task": {t: 100 * v for t, v in sr.items()},
                    "partial": partial, "excluded": bad}
        rows[tag] = sr

    for tag in list(out):
        rand = next((f"{tag[:-6]}{s}" for s in ("_randomized", "_rand") if f"{tag[:-6]}{s}" in out), None)
        if tag.endswith("_clean") and rand:
            c, r = out[tag]["success_rate"], out[rand]["success_rate"]
            print(f"{tag[:-6]:32s} clean {c:.2f}  randomized {r:.2f}  avg {(c + r) / 2:.2f}")
    if args.per_task and rows:
        tags = list(rows)
        print("\n" + f"{'task':28s}" + "".join(f"{t[-14:]:>15s}" for t in tags))
        for task in sorted(set().union(*rows.values())):
            print(f"{task:28s}" + "".join(
                f"{100 * rows[t][task]:15.0f}" if task in rows[t] else f"{'-':>15s}" for t in tags))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
