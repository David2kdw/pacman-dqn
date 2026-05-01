import os
import pickle
import time
import json
import argparse
from config import (
    BASE_FEAT_DIM,
    INPUT_SIZE,
    NUM_EPISODES,
    TARGET_UPDATE_FREQ,
    CHECKPOINT_DIR,
    MAX_STEPS_PER_EPISODE,
    MAX_EPISODE_TIME,
    VISUALIZE_TRAINING,
    REWARD_PROFILES,
)
from environment import Environment
from renderer import Renderer
from agent import Agent
from evaluate import append_eval_log, run_greedy_eval, set_seed
import learning
import pygame, sys

LOG_EVERY_EPISODES = 50
TRAINING_LOG_HEADER = (
    "timestamp\tepisodes\tcount\tavg_reward\tmin_reward\t"
    "max_reward\tavg_steps\tavg_duration_sec\tclear\tdeath\t"
    "step_cap\ttime_cap\tinterrupted\twall_bumps\tdots_eaten\tepsilon\tmemory\ttrain_step\t"
    "latest_loss\tavg_loss_last100\tlatest_q\tavg_q_last100\n"
)


def checkpoint_paths(checkpoint_dir):
    return {
        "latest_model": os.path.join(checkpoint_dir, "latest_model.pth"),
        "latest_meta": os.path.join(checkpoint_dir, "latest_meta.json"),
        "full_template": os.path.join(checkpoint_dir, "policy_ep{ep}.pth"),
        "training_log": os.path.join(checkpoint_dir, "training_log.txt"),
        "eval_log": os.path.join(checkpoint_dir, "eval_log.tsv"),
        "memory": os.path.join(checkpoint_dir, "memory.pkl"),
    }


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def append_training_log(path, episode_records, agent):
    """
    Append a compact rolling summary for the latest training window.
    """
    if not episode_records:
        return

    rewards = [rec['reward'] for rec in episode_records]
    steps = [rec['steps'] for rec in episode_records]
    durations = [rec['duration'] for rec in episode_records]
    reasons = [rec['reason'] for rec in episode_records]
    wall_bumps = sum(rec.get('wall_bumps', 0) for rec in episode_records)
    dots_eaten = sum(rec.get('dots_eaten', 0) for rec in episode_records)

    clear_count = reasons.count('clear')
    death_count = reasons.count('death')
    step_cap_count = reasons.count('step limit reached')
    time_cap_count = reasons.count('time limit reached')
    interrupted_count = reasons.count('interrupted')

    recent_losses = learning.losses[-100:]
    recent_q_values = learning.q_value_logs[-100:]
    latest_loss = learning.losses[-1] if learning.losses else 0.0
    latest_q = learning.q_value_logs[-1] if learning.q_value_logs else 0.0

    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not needs_header:
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline()
            needs_header = "wall_bumps" not in header or "interrupted" not in header

    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"episodes={episode_records[0]['episode']}-{episode_records[-1]['episode']}\t"
        f"count={len(episode_records)}\t"
        f"avg_reward={_avg(rewards):.3f}\t"
        f"min_reward={min(rewards):.3f}\t"
        f"max_reward={max(rewards):.3f}\t"
        f"avg_steps={_avg(steps):.1f}\t"
        f"avg_duration_sec={_avg(durations):.2f}\t"
        f"clear={clear_count}\t"
        f"death={death_count}\t"
        f"step_cap={step_cap_count}\t"
        f"time_cap={time_cap_count}\t"
        f"interrupted={interrupted_count}\t"
        f"wall_bumps={wall_bumps}\t"
        f"dots_eaten={dots_eaten}\t"
        f"epsilon={agent.epsilon:.6f}\t"
        f"memory={len(agent.memory)}\t"
        f"train_step={learning.train_step}\t"
        f"latest_loss={latest_loss:.6f}\t"
        f"avg_loss_last100={_avg(recent_losses):.6f}\t"
        f"latest_q={latest_q:.6f}\t"
        f"avg_q_last100={_avg(recent_q_values):.6f}\n"
    )

    with open(path, 'a', encoding='utf-8') as f:
        if needs_header:
            f.write(TRAINING_LOG_HEADER)
        f.write(line)


def save_latest_checkpoint(paths, agent, episode, args):
    agent.save(paths["latest_model"], paths["memory"])
    with open(paths["latest_meta"], 'w') as meta_file:
        json.dump({
            'episode': episode,
            'epsilon': agent.epsilon,
            'reward_profile': args.reward_profile,
            'seed': args.seed,
            'base_feat_dim': BASE_FEAT_DIM,
            'input_size': INPUT_SIZE,
        }, meta_file)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Pac-Man DQN.")
    parser.add_argument("--reward-profile", choices=sorted(REWARD_PROFILES), default="baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--max-seconds", type=float, default=MAX_EPISODE_TIME)
    return parser.parse_args(argv)


def main(argv=None):
    """
    Main training loop with resume and checkpointing:
      - Resume from latest_model + metadata if present
      - Run episodes with optional step/time limits
      - Save rolling latest_model + metadata each episode
      - Save replay memory and full snapshots every 100 episodes
      - Append training summaries to training_log.txt every 50 episodes
    """
    args = parse_args(argv)
    set_seed(args.seed)

    paths = checkpoint_paths(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize environment and agent
    env = Environment(reward_config=args.reward_profile)
    init_state = env.reset()
    _, state_dim = init_state.shape
    agent = Agent(input_dim=INPUT_SIZE)

    # Resume from latest checkpoint if available and state encoding is compatible.
    last_ep = 0
    can_load_memory = True
    if os.path.exists(paths["latest_model"]) and os.path.exists(paths["latest_meta"]):
        with open(paths["latest_meta"], 'r') as f:
            meta = json.load(f)
        if meta.get("input_size") == INPUT_SIZE:
            last_ep = meta.get('episode', 0)
            agent.epsilon = meta.get('epsilon', agent.epsilon)
            agent.load(paths["latest_model"])
            print(f"Resumed training from episode {last_ep}, epsilon={agent.epsilon:.3f}")
        else:
            can_load_memory = False
            print(
                "[resume] checkpoint input_size is incompatible with current state encoding; "
                "starting a fresh model in this checkpoint directory."
            )

    # Load memory
    if can_load_memory:
        try:
            with open(paths["memory"], "rb") as f:
                loaded = pickle.load(f)
                agent.memory = loaded
                print("Memory loaded")
        except Exception as e:
            print(f"[memory] failed to load: {e}")
    else:
        print("[memory] skipped incompatible replay memory")


    start_ep = last_ep + 1

    paused = False
    heatmap_cache = None
    renderer = Renderer() if VISUALIZE_TRAINING else None
    episode_records = []
    last_completed_ep = last_ep
    current_partial_record = None

    # Training loop
    try:
        for ep in range(start_ep, args.episodes + 1):
            state = agent.reset_episode(env)  # returns stacked [1, feat*K]
            total_reward = 0.0
            done = False
            step = 0
            end_reason = None
            wall_bumps = 0
            dots_eaten = 0
            start_time = time.time()
            paused = False
            heatmap_cache = None
            current_partial_record = {
                'episode': ep,
                'reward': total_reward,
                'steps': step,
                'duration': 0.0,
                'reason': 'interrupted',
                'wall_bumps': wall_bumps,
                'dots_eaten': dots_eaten,
            }
            if renderer is not None:
                renderer.render(env,
                                heatmap=None,
                                stats={'episode': ep,
                                       'reward': total_reward,
                                       'epsilon': agent.epsilon})

            # Loop until terminal, step limit, or time limit
            while (not done
                   and step < args.max_steps
                   and (time.time() - start_time) < args.max_seconds):
                if renderer is not None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                        if event.type == pygame.KEYUP and event.key == pygame.K_h:
                            # toggle pause & heatmap
                            print("h pressed")
                            paused = not paused
                            if paused:
                                # compute once
                                heatmap_cache = agent.compute_heatmap(env)
                            else:
                                heatmap_cache = None
            
                if paused:
                    renderer.render(env,
                                    heatmap=heatmap_cache,
                                    stats={'episode': ep,
                                           'reward': total_reward,
                                           'epsilon': agent.epsilon})
                    pygame.time.delay(100)
                    continue   # back to top of while, still paused
            

                action = agent.select_action()
                s, a, r, s_next, done = agent.step(env, action)
                last_event = getattr(env, "last_event", {})
                if last_event.get("wall_bump"):
                    wall_bumps += 1
                if last_event.get("ate_dot"):
                    dots_eaten += 1
                step += 1
                current_partial_record.update({
                    'reward': total_reward + r,
                    'steps': step,
                    'duration': time.time() - start_time,
                    'wall_bumps': wall_bumps,
                    'dots_eaten': dots_eaten,
                })

                if not done:
                    if step >= args.max_steps:
                        done = True
                        end_reason = 'step limit reached'
                        r += env.reward_config.R_TIMEOUT
                    elif (time.time() - start_time) >= args.max_seconds:
                        done = True
                        end_reason = 'time limit reached'
                        r += env.reward_config.R_TIMEOUT

                agent.store_transition(s, a, r, s_next, done)
                agent.optimize_model()
                total_reward += r
                state = s_next

                if renderer is not None:
                    renderer.render(env,
                        heatmap=None,
                        stats={
                        'episode':  ep,
                        'reward':   total_reward,
                        'epsilon':  agent.epsilon
                        }
                    )

            # If episode finished by cap, the final transition was stored as terminal.
            if end_reason is not None:
                print(f"→ Episode {ep} ended early ({end_reason})")
            elif len(env.dots) == 0:
                end_reason = 'clear'
            else:
                end_reason = 'death'
        

            # Episode-end updates
            agent.decay_epsilon()
            if ep % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            print(f"Episode {ep}/{args.episodes}"
                  f" | Steps: {step}"
                  f" | Reward: {total_reward:.2f}"
                  f" | Epsilon: {agent.epsilon:.3f}")

            episode_records.append({
                'episode': ep,
                'reward': total_reward,
                'steps': step,
                'duration': time.time() - start_time,
                'reason': end_reason,
                'wall_bumps': wall_bumps,
                'dots_eaten': dots_eaten,
            })
            last_completed_ep = ep
            current_partial_record = None

            if ep % LOG_EVERY_EPISODES == 0:
                append_training_log(paths["training_log"], episode_records[-LOG_EVERY_EPISODES:], agent)

            if args.eval_every > 0 and ep % args.eval_every == 0:
                metrics = run_greedy_eval(
                    agent,
                    reward_config=args.reward_profile,
                    episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    max_seconds=args.max_seconds,
                    seed=args.seed,
                )
                append_eval_log(paths["eval_log"], args.reward_profile, args.seed, ep, metrics)
                print(
                    f"[Eval {ep}] dots/100={metrics['dots_per_100_steps']:.2f}"
                    f" | walls/100={metrics['wall_bumps_per_100_steps']:.2f}"
                    f" | death_after_dots={metrics['avg_death_after_dots']:.2f}"
                )

            # Save latest model and metadata.
            save_latest_checkpoint(paths, agent, ep, args)

            # Full snapshot every 100 episodes
            if ep % 100 == 0:
                agent.save(paths["full_template"].format(ep=ep), paths["memory"])
                agent.save_memory(paths["memory"])

    except KeyboardInterrupt:
        print("\nInterrupted. Saving latest checkpoint and replay memory...")
        if last_completed_ep > 0:
            save_latest_checkpoint(paths, agent, last_completed_ep, args)
            agent.save_memory(paths["memory"])
            unwritten = len(episode_records) % LOG_EVERY_EPISODES
            if unwritten:
                append_training_log(paths["training_log"], episode_records[-unwritten:], agent)
            if current_partial_record is not None and current_partial_record.get('steps', 0) > 0:
                append_training_log(paths["training_log"], [current_partial_record], agent)
            print(f"Saved interrupt checkpoint at episode {last_completed_ep}.")
        else:
            agent.save_memory(paths["memory"])
            if current_partial_record is not None and current_partial_record.get('steps', 0) > 0:
                append_training_log(paths["training_log"], [current_partial_record], agent)
            print("Saved replay memory, but no completed episode was available for metadata.")
        if renderer is not None:
            pygame.quit()
        return
        

    print("Training complete.")


if __name__ == '__main__':
    main()
