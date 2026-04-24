import os
import pickle
import time
import json
import torch
from config import (
    INPUT_SIZE,
    NUM_EPISODES,
    TARGET_UPDATE_FREQ,
    CHECKPOINT_DIR,
    MEMORY_PATH,
    MAX_STEPS_PER_EPISODE,
    MAX_EPISODE_TIME
)
from environment import Environment
from environment import R_TIMEOUT
from renderer import Renderer
from agent import Agent
import pygame, sys

# File paths for checkpoints and metadata
LATEST_MODEL = os.path.join(CHECKPOINT_DIR, 'latest_model.pth')
LATEST_META  = os.path.join(CHECKPOINT_DIR, 'latest_meta.json')
FULL_TEMPLATE = os.path.join(CHECKPOINT_DIR, 'policy_ep{ep}.pth')


def main():
    """
    Main training loop with resume and checkpointing:
      - Resume from latest_model + metadata if present
      - Run episodes with optional step/time limits
      - Save rolling latest_model + metadata each episode
      - Save replay memory and full snapshots every 100 episodes
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
    renderer = Renderer()

    # Training loop
    for ep in range(start_ep, NUM_EPISODES + 1):
        state = agent.reset_episode(env)  # returns stacked [1, feat*K]
        total_reward = 0.0
        done = False
        step = 0
        end_reason = None
        start_time = time.time()
        paused = False
        heatmap_cache = None
        renderer.render(env,
                        heatmap=None,
                        stats={'episode': ep,
                               'reward': total_reward,
                               'epsilon': agent.epsilon})

        # Loop until terminal, step limit, or time limit
        while (not done
               and step < MAX_STEPS_PER_EPISODE
               and (time.time() - start_time) < MAX_EPISODE_TIME):
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
            

            action = agent.select_action()
            s, a, r, s_next, done = agent.step(env, action)
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

            agent.store_transition(s, a, r, s_next, done)
            agent.optimize_model()
            total_reward += r
            state = s_next

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
        

        # Episode-end updates
        agent.decay_epsilon()
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        print(f"Episode {ep}/{NUM_EPISODES}"
              f" | Steps: {step}"
              f" | Reward: {total_reward:.2f}"
              f" | Epsilon: {agent.epsilon:.3f}")

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
