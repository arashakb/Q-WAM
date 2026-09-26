"""RoboTwin 2.0 success rates of ImageWAM runs, read from the per-episode videos.

Every finished episode leaves <run>/<task>/episode<N>_randomized-<true|false>_success-<true|false>.mp4.
Those names are the source of truth: summary.csv of a resumed run can miss tasks. Task directories
named *.partial_* (set aside when a run was restarted) are ignored.

Success of a condition = mean over the 50 tasks of (successful episodes / episodes per task).

  python scripts/imagewam/read_results.py <run dir> [<run dir> ...] [--per-task]

Each run directory is $IMAGEWAM_ROOT/evaluate_results/robotwin/<ckpt tag>/<timestamp>; clean and
randomized episodes are told apart by the file name, so the two runs of one method can be passed
together.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

EP = re.compile(r"^episode(\d+)_randomized-(true|false)_success-(true|false)\.mp4$")
NUM_TASKS = 50


def scan(run_dirs):
    """{condition: {task: {episode: success}}}; raises on conflicting duplicate episodes."""
    res = defaultdict(lambda: defaultdict(dict))
    for run in run_dirs:
        if not os.path.isdir(run):
            raise SystemExit(f"not a directory: {run}")
        for task in sorted(os.listdir(run)):
            tdir = os.path.join(run, task)
            if not os.path.isdir(tdir) or ".partial_" in task:
                continue
            for f in os.listdir(tdir):
                m = EP.match(f)
                if not m:
                    continue
                cond = "randomized" if m.group(2) == "true" else "clean"
                ep, ok = int(m.group(1)), m.group(3) == "true"
                prev = res[cond][task].get(ep)
                if prev is not None and prev != ok:
                    raise SystemExit(f"{tdir}: episode {ep} ({cond}) has both outcomes")
                res[cond][task][ep] = ok
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--episodes", type=int, default=100, help="episodes per task (protocol)")
    ap.add_argument("--per-task", action="store_true", help="print per-task success")
    a = ap.parse_args()

    res = scan(a.runs)
    if not res:
        raise SystemExit("no episode videos found")
    means, complete = {}, True
    for cond in ("clean", "randomized"):
        tasks = res.get(cond)
        if not tasks:
            continue
        full = {t: eps for t, eps in tasks.items() if sorted(eps) == list(range(a.episodes))}
        partial = sorted(set(tasks) - set(full))
        rates = {t: 100.0 * sum(eps.values()) / a.episodes for t, eps in full.items()}
        n_eps = sum(len(e) for e in tasks.values())
        print(f"{cond:10s} tasks {len(tasks):2d} (complete {len(full):2d}/{NUM_TASKS}), "
              f"episodes {n_eps}", end="")
        if len(full) == NUM_TASKS and not partial:
            means[cond] = sum(rates.values()) / NUM_TASKS
            print(f", success {means[cond]:.2f}")
        else:
            complete = False
            shown = ", ".join(f"{t}({len(tasks[t])})" for t in partial[:6])
            print(f"  INCOMPLETE{': ' + shown if shown else ''}"
                  + (f"; mean over complete tasks {sum(rates.values()) / len(rates):.2f}" if rates else ""))
        if a.per_task:
            for t in sorted(tasks):
                r = f"{100.0 * sum(tasks[t].values()) / len(tasks[t]):6.1f}" if tasks[t] else "   n/a"
                print(f"    {t:32s} {r}  ({sum(tasks[t].values())}/{len(tasks[t])})")
    if "clean" in means and "randomized" in means:
        print(f"average    {(means['clean'] + means['randomized']) / 2:.2f}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
