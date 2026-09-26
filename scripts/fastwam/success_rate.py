"""Average RoboTwin success rate of one or more Fast-WAM run directories.

RoboTwin writes one file per task, <run dir>/<task>/_result_<clean|random>.txt, whose last line is
that task's success fraction over its episodes. The reported success rate is the mean of these
fractions over the 50 tasks, in percent.

  python scripts/fastwam/success_rate.py $FASTWAM_ROOT/evaluate_results/robotwin/robotwin_uncond_3cam_384/qwam_clean
"""
import argparse
import glob
import os

NUM_TASKS = 50


def task_rates(run_dir: str) -> dict[str, float]:
    rates = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "_result_*.txt"))):
        lines = [ln.strip() for ln in open(path).read().splitlines() if ln.strip()]
        task = os.path.basename(os.path.dirname(path))
        if lines:
            if task in rates:
                raise SystemExit(f"{run_dir}: {task} has results for both settings; use one run dir per setting")
            rates[task] = float(lines[-1])
    return rates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--per-task", action="store_true", help="also print every task's success rate")
    args = ap.parse_args()
    for run_dir in args.run_dirs:
        rates = task_rates(run_dir)
        if args.per_task:
            for task, r in rates.items():
                print(f"  {task:32s} {100 * r:6.1f}")
        if not rates:
            print(f"{run_dir}: no results yet")
            continue
        note = "" if len(rates) == NUM_TASKS else f"  (incomplete: {len(rates)}/{NUM_TASKS} tasks)"
        print(f"{run_dir}: {100 * sum(rates.values()) / len(rates):.2f}% over {len(rates)} tasks{note}")


if __name__ == "__main__":
    main()
