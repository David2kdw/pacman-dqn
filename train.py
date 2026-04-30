import os
import pickle
import time
import json
from config import (
    INPUT_SIZE,
    NUM_EPISODES,
    TARGET_UPDATE_FREQ,
    CHECKPOINT_DIR,
    MEMORY_PATH,
    MAX_STEPS_PER_EPISODE,
    MAX_EPISODE_TIME,
    ACTION_MASK_UNTIL_EPISODE,
    VISUALIZE_TRAINING
)
from environment import Environment
from environment import R_TIMEOUT
from renderer import Renderer
from agent import Agent
import learning
import pygame, sys

# File paths for checkpoints and metadata
LATEST_MODEL = os.path.join(CHECKPOINT_DIR, 'latest_model.pth')
LATEST_META  = os.path.join(CHECKPOINT_DIR, 'latest_meta.json')
FULL_TEMPLATE = os.path.join(CHECKPOINT_DIR, 'policy_ep{ep}.pth')
TRAINING_LOG = os.path.join(CHECKPOINT_DIR, 'training_log.txt')
LOG_EVERY_EPISODES = 50
TRAINING_LOG_HEADER = (
    "timestamp\tepisodes\tcount\tavg_reward\tmin_reward\t"
    "max_reward\tavg_steps\tavg_duration_sec\tclear\tdeath\t"
    "step_cap\ttime_cap\tmask_episodes\twall_bumps\tdots_eaten\tepsilon\tmemory\ttrain_step\t"
    "latest_loss\tavg_loss_last100\tlatest_q\tavg_q_last100\n"
)


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
    mask_episodes = sum(1 for rec in episode_records if rec.get('action_masked'))
    wall_bumps = sum(rec.get('wall_bumps', 0) for rec in episode_records)
    dots_eaten = sum(rec.get('dots_eaten', 0) for rec in episode_records)

    clear_count = reasons.count('clear')
    death_count = reasons.count('death')
    step_cap_count = reasons.count('step limit reached')
    time_cap_count = reasons.count('time limit reached')

    recent_losses = learning.losses[-100:]
    recent_q_values = learning.q_value_logs[-100:]
    latest_loss = learning.losses[-1] if learning.losses else 0.0
    latest_q = learning.q_value_logs[-1] if learning.q_value_logs else 0.0

    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not needs_header:
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline()
            needs_header = "wall_bumps" not in header or "mask_episodes" not in header

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
        f"mask_episodes={mask_episodes}\t"
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


def main():
    """
    Main training loop with resume and checkpointing:
      - Resume from latest_model + metadata if present
      - Run episodes with optional step/time limits
      - Save rolling latest_model + metadata each episode
      - Save replay memory and full snapshots every 100 episodes
      - Append training summaries to training_log.txt every 50 episodes
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize environment and agent
    env = Environment()
    init_state = env.reset()
    _, state_dim = init_state.shape
    agent = Agent(input_dim=INPUT_SIZE)

    # Resume from latest checkpoint if available
    last_ep = 0
    if os.path.exists(LATEST_MODEL) and os.path.exists(LATEST_META):
        with open(LATEST_META, 'r') as f:
            meta = json.load(f)
        last_ep = meta.get('episode', 0)
        agent.epsilon = meta.get('epsilon', agent.epsilon)
        agent.load(LATEST_MODEL)
        print(f"Resumed training from episode {last_ep}, epsilon={agent.epsilon:.3f}")

    # Load memory
    try:
        with open(MEMORY_PATH, "rb") as f:
            loaded = pickle.load(f)
            agent.memory = loaded
            print("Memory loaded")
    except Exception as e:
        print(f"[memory] failed to load: {e}")


    start_ep = last_ep + 1

    paused = False
    heatmap_cache = None
    renderer = Renderer() if VISUALIZE_TRAINING else None
    episode_records = []

    # Training loop
    for ep in range(start_ep, NUM_EPISODES + 1):
        state = agent.reset_episode(env)  # returns stacked [1, feat*K]
        total_reward = 0.0
        done = False
        step = 0
        end_reason = None
        action_masked = ep <= ACTION_MASK_UNTIL_EPISODE
        wall_bumps = 0
        dots_eaten = 0
        start_time = time.time()
        paused = False
        heatmap_cache = None
        if renderer is not None:
            renderer.render(env,
                            heatmap=None,
                            stats={'episode': ep,
                                   'reward': total_reward,
                                   'epsilon': agent.epsilon})

        # Loop until terminal, step limit, or time limit
        while (not done
               and step < MAX_STEPS_PER_EPISODE
               and (time.time() - start_time) < MAX_EPISODE_TIME):
            if renderer is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
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
            

            valid_actions = env.valid_actions() if action_masked else None
            action = agent.select_action(valid_actions=valid_actions)
            s, a, r, s_next, done = agent.step(env, action)
            last_event = getattr(env, "last_event", {})
            if last_event.get("wall_bump"):
                wall_bumps += 1
            if last_event.get("ate_dot"):
                dots_eaten += 1
            step += 1

            if not done:
                if step >= MAX_STEPS_PER_EPISODE:
                    done = True
                    end_reason = 'step limit reached'
                    r += R_TIMEOUT
                elif (time.time() - start_time) >= MAX_EPISODE_TIME:
                    done = True
                    end_reason = 'time limit reached'
                    r += R_TIMEOUT

            next_valid_actions = env.valid_actions() if action_masked else None
            agent.store_transition(s, a, r, s_next, done, next_valid_actions)
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

        print(f"Episode {ep}/{NUM_EPISODES}"
              f" | Steps: {step}"
              f" | Reward: {total_reward:.2f}"
              f" | Epsilon: {agent.epsilon:.3f}")

        episode_records.append({
            'episode': ep,
            'reward': total_reward,
            'steps': step,
            'duration': time.time() - start_time,
            'reason': end_reason,
            'action_masked': action_masked,
            'wall_bumps': wall_bumps,
            'dots_eaten': dots_eaten,
        })

        if ep % LOG_EVERY_EPISODES == 0:
            append_training_log(TRAINING_LOG, episode_records[-LOG_EVERY_EPISODES:], agent)

        # Save latest model and metadata.
        agent.save(LATEST_MODEL, MEMORY_PATH)
        with open(LATEST_META, 'w') as meta_file:
            json.dump({'episode': ep, 'epsilon': agent.epsilon}, meta_file)

        # Full snapshot every 100 episodes
        if ep % 100 == 0:
            agent.save(FULL_TEMPLATE.format(ep=ep), MEMORY_PATH)
            agent.save_memory(MEMORY_PATH)
        

    print("Training complete.")


if __name__ == '__main__':
    main()
