import os
import sys
import argparse
import torch
import pygame
from config import CHECKPOINT_DIR
from environment import Environment
from renderer import Renderer
from agent import Agent


def main():
    parser = argparse.ArgumentParser(
        description='Run a trained Pac-Man DQN model with rendering for testing.'
    )
    parser.add_argument(
        '--model', '-m', type=str,
        help='Path to a .pth model checkpoint. If omitted, uses CHECKPOINT_DIR/latest_model.pth.'
    )
    parser.add_argument(
        '--episodes', '-e', type=int, default=5,
        help='Number of test episodes to run (default: 5).'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Rendering frames per second (default: 30).'
    )
    args = parser.parse_args()

    # Determine checkpoint to load: argument or latest_model.pth
    ckpt = args.model or os.path.join(CHECKPOINT_DIR, 'latest_model.pth')
    if not ckpt or not os.path.isfile(ckpt):
        print(f"Error: No valid checkpoint found at {ckpt}")
        sys.exit(1)

    # Initialize environment, agent, renderer
    env = Environment()
    init_state = env.reset()
    _, state_dim = init_state.shape
    agent = Agent(input_dim=state_dim)
    agent.load(ckpt)
    agent.epsilon = 0.0  # greedy policy

    pygame.init()
    renderer = Renderer()
    # Set target FPS
    renderer.clock.tick(args.fps)

    print(f"Loaded checkpoint: {ckpt}")
    print(f"Running {args.episodes} test episodes...")

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Handle OS events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Select & apply action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            # Render frame
            renderer.render(env)

        print(f"[Test {ep}] Steps: {steps}, Total Reward: {total_reward:.2f}")

    pygame.quit()


if __name__ == '__main__':
    main()
