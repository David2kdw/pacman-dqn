import argparse
import os
import shutil
import subprocess
import sys

from config import MAX_EPISODE_TIME, MAX_STEPS_PER_EPISODE, REWARD_PROFILES


SUMMARY_HEADER = (
    "profile\tseed\tcheckpoint_dir\tepisode\tcount\tavg_reward\tavg_dots\t"
    "avg_steps\tavg_wall_bumps\tdots_per_100_steps\twall_bumps_per_100_steps\t"
    "avg_death_after_dots\tavg_death_after_steps\tdeath_count\tclear_count\t"
    "step_cap_count\ttime_cap_count\n"
)


def _parse_eval_line(line):
    values = {}
    for part in line.strip().split("\t"):
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return values


def _latest_eval_metrics(eval_log):
    if not os.path.exists(eval_log):
        return None
    with open(eval_log, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if len(rows) <= 1:
        return None
    return _parse_eval_line(rows[-1])


def _write_summary(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(SUMMARY_HEADER)
        for row in rows:
            f.write(
                f"{row.get('profile', '')}\t"
                f"{row.get('seed', '')}\t"
                f"{row.get('checkpoint_dir', '')}\t"
                f"{row.get('episode', '')}\t"
                f"{row.get('count', '')}\t"
                f"{row.get('avg_reward', '')}\t"
                f"{row.get('avg_dots', '')}\t"
                f"{row.get('avg_steps', '')}\t"
                f"{row.get('avg_wall_bumps', '')}\t"
                f"{row.get('dots_per_100_steps', '')}\t"
                f"{row.get('wall_bumps_per_100_steps', '')}\t"
                f"{row.get('avg_death_after_dots', '')}\t"
                f"{row.get('avg_death_after_steps', '')}\t"
                f"{row.get('death_count', '')}\t"
                f"{row.get('clear_count', '')}\t"
                f"{row.get('step_cap_count', '')}\t"
                f"{row.get('time_cap_count', '')}\n"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run reward profile sweeps.")
    parser.add_argument("--profiles", nargs="+", choices=sorted(REWARD_PROFILES), default=sorted(REWARD_PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--base-dir", default=os.path.join("checkpoints", "sweeps"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--max-seconds", type=float, default=MAX_EPISODE_TIME)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete each profile/seed checkpoint directory before training it.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.base_dir, exist_ok=True)

    summary_rows = []
    for profile in args.profiles:
        for seed in args.seeds:
            checkpoint_dir = os.path.join(args.base_dir, f"{profile}_seed{seed}")
            if args.fresh and os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

            cmd = [
                sys.executable,
                "train.py",
                "--reward-profile",
                profile,
                "--seed",
                str(seed),
                "--checkpoint-dir",
                checkpoint_dir,
                "--episodes",
                str(args.episodes),
                "--eval-every",
                str(args.eval_every),
                "--eval-episodes",
                str(args.eval_episodes),
                "--max-steps",
                str(args.max_steps),
                "--max-seconds",
                str(args.max_seconds),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            metrics = _latest_eval_metrics(os.path.join(checkpoint_dir, "eval_log.tsv"))
            if metrics is not None:
                metrics["checkpoint_dir"] = checkpoint_dir
                summary_rows.append(metrics)

    summary_path = os.path.join(args.base_dir, "reward_sweep_summary.tsv")
    _write_summary(summary_path, summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
