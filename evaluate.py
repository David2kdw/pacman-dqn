import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch

from agent import Agent
from config import (
    CHECKPOINT_DIR,
    INPUT_SIZE,
    MAX_EPISODE_TIME,
    MAX_STEPS_PER_EPISODE,
    REWARD_PROFILES,
)
from environment import Environment


EVAL_LOG_HEADER = (
    "timestamp\tprofile\tseed\tepisode\tcount\tavg_reward\tavg_dots\tavg_steps\t"
    "avg_wall_bumps\tdots_per_100_steps\twall_bumps_per_100_steps\t"
    "avg_death_after_dots\tavg_death_after_steps\tdeath_count\tclear_count\t"
    "step_cap_count\ttime_cap_count\n"
)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _reason_from_env(env, done, steps, elapsed, max_steps, max_seconds):
    if done:
        reason = getattr(env, "last_event", {}).get("terminal_reason")
        if reason:
            return reason
        return "clear" if len(env.dots) == 0 else "death"
    if steps >= max_steps:
        return "step limit reached"
    if elapsed >= max_seconds:
        return "time limit reached"
    return "unknown"


def run_greedy_eval(
    agent,
    reward_config=None,
    episodes: int = 30,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    max_seconds: float = MAX_EPISODE_TIME,
    seed: int | None = None,
):
    """
    Run greedy evaluation without training, replay writes, or optimizer updates.
    The agent's epsilon, step counter, frame stack, and model mode are restored.
    """
    set_seed(seed)

    old_epsilon = agent.epsilon
    old_steps_done = agent.steps_done
    old_state_buf = deque((t.clone() for t in agent.state_buf), maxlen=agent.state_buf.maxlen)
    old_training = agent.policy_net.training

    records = []
    agent.epsilon = 0.0
    agent.policy_net.eval()

    try:
        for _ in range(episodes):
            env = Environment(reward_config=reward_config)
            state = agent.reset_episode(env)
            total_reward = 0.0
            steps = 0
            wall_bumps = 0
            dots_eaten = 0
            done = False
            start = time.time()

            while (
                not done
                and steps < max_steps
                and (time.time() - start) < max_seconds
            ):
                action = agent.select_action(state)
                _, _, reward, next_state, done = agent.step(env, action)
                state = next_state
                total_reward += reward
                steps += 1

                last_event = getattr(env, "last_event", {})
                if last_event.get("wall_bump"):
                    wall_bumps += 1
                if last_event.get("ate_dot"):
                    dots_eaten += 1

            elapsed = time.time() - start
            reason = _reason_from_env(env, done, steps, elapsed, max_steps, max_seconds)
            records.append({
                "reward": total_reward,
                "steps": steps,
                "wall_bumps": wall_bumps,
                "dots_eaten": dots_eaten,
                "reason": reason,
            })
    finally:
        agent.epsilon = old_epsilon
        agent.steps_done = old_steps_done
        agent.state_buf = old_state_buf
        agent.policy_net.train(old_training)

    total_steps = sum(r["steps"] for r in records)
    total_dots = sum(r["dots_eaten"] for r in records)
    total_walls = sum(r["wall_bumps"] for r in records)
    death_records = [r for r in records if r["reason"] == "death"]

    return {
        "count": len(records),
        "avg_reward": _avg([r["reward"] for r in records]),
        "avg_dots": _avg([r["dots_eaten"] for r in records]),
        "avg_steps": _avg([r["steps"] for r in records]),
        "avg_wall_bumps": _avg([r["wall_bumps"] for r in records]),
        "dots_per_100_steps": (100.0 * total_dots / total_steps) if total_steps else 0.0,
        "wall_bumps_per_100_steps": (100.0 * total_walls / total_steps) if total_steps else 0.0,
        "avg_death_after_dots": _avg([r["dots_eaten"] for r in death_records]),
        "avg_death_after_steps": _avg([r["steps"] for r in death_records]),
        "death_count": sum(1 for r in records if r["reason"] == "death"),
        "clear_count": sum(1 for r in records if r["reason"] == "clear"),
        "step_cap_count": sum(1 for r in records if r["reason"] == "step limit reached"),
        "time_cap_count": sum(1 for r in records if r["reason"] == "time limit reached"),
    }


def append_eval_log(path, profile, seed, episode, metrics):
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"profile={profile}\t"
        f"seed={seed if seed is not None else ''}\t"
        f"episode={episode}\t"
        f"count={metrics['count']}\t"
        f"avg_reward={metrics['avg_reward']:.3f}\t"
        f"avg_dots={metrics['avg_dots']:.3f}\t"
        f"avg_steps={metrics['avg_steps']:.1f}\t"
        f"avg_wall_bumps={metrics['avg_wall_bumps']:.3f}\t"
        f"dots_per_100_steps={metrics['dots_per_100_steps']:.3f}\t"
        f"wall_bumps_per_100_steps={metrics['wall_bumps_per_100_steps']:.3f}\t"
        f"avg_death_after_dots={metrics['avg_death_after_dots']:.3f}\t"
        f"avg_death_after_steps={metrics['avg_death_after_steps']:.1f}\t"
        f"death_count={metrics['death_count']}\t"
        f"clear_count={metrics['clear_count']}\t"
        f"step_cap_count={metrics['step_cap_count']}\t"
        f"time_cap_count={metrics['time_cap_count']}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(EVAL_LOG_HEADER)
        f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Run greedy Pac-Man DQN evaluation.")
    parser.add_argument("--model", "-m", default=os.path.join(CHECKPOINT_DIR, "latest_model.pth"))
    parser.add_argument("--reward-profile", choices=sorted(REWARD_PROFILES), default="baseline")
    parser.add_argument("--episodes", "-e", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--max-seconds", type=float, default=MAX_EPISODE_TIME)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"No checkpoint found at {args.model}")

    agent = Agent(input_dim=INPUT_SIZE)
    agent.load(args.model)
    metrics = run_greedy_eval(
        agent,
        reward_config=args.reward_profile,
        episodes=args.episodes,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        seed=args.seed,
    )
    print(metrics)


if __name__ == "__main__":
    main()
