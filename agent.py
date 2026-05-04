from collections import deque

import torch
import pickle
import numpy as np
from learning import DQN, DuelingDQN, DuelingCNN, train_dqn, select_action as select_action_fn
from replayMemory import ReplayMemory
from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    MODEL_TYPE,
    MEMORY_CAPACITY,
    MEMORY_PATH,
    LR,
    LR_MIN,
    LR_DECAY,
    GAMMA,
    N_STEP_RETURN,
    BATCH_SIZE,
    EPSILON_START,
    EPSILON_DECAY,
    EPSILON_MIN,
    GRID_SIZE
)


class Agent:
    """
    DQN-based agent that manages policy and target networks, replay memory,
    epsilon-greedy action selection, and learning updates.
    """

    def __init__(self, input_dim, output_dim=OUTPUT_SIZE, k_frames=3):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.model_type = MODEL_TYPE

        # Networks
        self.policy_net = self._build_network(input_dim, output_dim).to(self.device)
        self.target_net = self._build_network(input_dim, output_dim).to(self.device)
        self.update_target()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR, weight_decay=1e-5
        )

        # Replay memory
        self.memory = ReplayMemory(MEMORY_CAPACITY, min_terminal_samples=5)
        self.n_step = max(1, int(N_STEP_RETURN))
        self.n_step_buffer = deque()

        # Epsilon-greedy parameters
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.steps_done = 0

        self.k = k_frames
        self.state_buf = deque(maxlen=self.k)

    def _build_network(self, input_dim, output_dim):
        if self.model_type == "cnn":
            return DuelingCNN(input_dim, HIDDEN_SIZE, output_dim)
        if self.model_type == "mlp":
            return DuelingDQN(input_dim, HIDDEN_SIZE, output_dim)
        raise ValueError(f"Unsupported MODEL_TYPE: {self.model_type}")

    def _stacked(self):
        assert len(self.state_buf) == self.k
        frames = []
        for f in self.state_buf:
            if f.dim() == 1:
                f = f.unsqueeze(0)
            frames.append(f)
        return torch.cat(frames, dim=1)

    def reset_episode(self, env):
        s0 = env.reset().to(self.device)
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        self.state_buf.clear()
        self.n_step_buffer.clear()
        for _ in range(self.k):
            self.state_buf.append(s0.clone())
        return self._stacked()

    def step(self, env, action):
        """
        Interact one step and return (stacked_state, action, reward, stacked_next, done).
        """
        s_t = self._stacked().detach().cpu()

        next_state, reward, done = env.step(action)

        s1 = next_state.to(self.device)
        if s1.dim() == 1:
            s1 = s1.unsqueeze(0)
        self.state_buf.append(s1)
        next_stacked = self._stacked().detach().cpu()

        if getattr(self, "_stack_debug", False):
            K = self.k
            feat = next_stacked.shape[1] // K

            left = next_stacked[:, : (K - 1) * feat]
            right = s_t[:, feat: K * feat]
            if not torch.allclose(left, right):
                print("⚠️ Agent stack debug: shift mismatch",
                      (left - right).abs().max().item())
            else:
                print("Agent stack debug: match!")

            if self.steps_done > 50:
                self._stack_debug = False
        return s_t, action, float(reward), next_stacked, bool(done)

    def update_target(self):
        """
        Copy policy network weights to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, *args):
        """
        Save a transition into replay memory.

        """
        state, action, reward, next_state, done = args
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if self.n_step == 1:
            self.memory.push(state, action, reward, next_state, done)
            self.n_step_buffer.clear()
            return

        if len(self.n_step_buffer) >= self.n_step:
            self._push_n_step_transition(self.n_step)

        if done:
            while self.n_step_buffer:
                self._push_n_step_transition(len(self.n_step_buffer))

    def _push_n_step_transition(self, horizon):
        reward_sum = 0.0
        next_state = None
        done = False

        for i, (_, _, reward, candidate_next_state, candidate_done) in enumerate(
            list(self.n_step_buffer)[:horizon]
        ):
            reward_sum += (GAMMA ** i) * float(reward)
            next_state = candidate_next_state
            done = bool(candidate_done)
            if done:
                break

        state, action, _, _, _ = self.n_step_buffer[0]
        bootstrap_discount = GAMMA ** horizon
        self.memory.push(state, action, reward_sum, next_state, done, bootstrap_discount)
        self.n_step_buffer.popleft()

    def select_action(self, state=None):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state (torch.Tensor): current state tensor of shape [1, state_dim]
        Returns:
            int: chosen action index
        """
        if state is None:
            state = self._stacked()
        state = state.to(self.device)

        eps = self.epsilon
        action = select_action_fn(self.policy_net, state.to(self.device), eps)
        self.steps_done += 1
        return action

    def optimize_model(self):
        """
        Perform a single DQN training step using replay memory.
        """
        train_dqn(
            self.policy_net,
            self.target_net,
            self.memory,
            self.optimizer,
            BATCH_SIZE,
            GAMMA
        )

    def decay_epsilon(self):
        """
        Decay epsilon after each episode to reduce exploration over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, learning_rate):
        lr = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def scheduled_learning_rate(self, completed_episodes):
        lr = LR * (LR_DECAY ** max(0, int(completed_episodes)))
        return max(LR_MIN, lr)

    def decay_learning_rate(self):
        self.set_learning_rate(max(LR_MIN, self.get_learning_rate() * LR_DECAY))

    def save(self, model_path: str, memory_path: str):
        """
        Save the policy network weights to `model_path`.

        `memory_path` is kept for backward-compatible call sites. Replay memory
        is large, so training saves it explicitly at snapshot intervals.
        """
        torch.save(self.policy_net.state_dict(), model_path)

    def save_memory(self, memory_path: str):
        """
        Persist replay memory separately from the model checkpoint.
        """
        with open(memory_path, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, path: str):
        """
        Load policy network weights from disk and update target network.

        Args:
            path (str): filepath to load the model from
        """
        try:
            self.policy_net.load_state_dict(torch.load(path))
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint. The model architecture, MODEL_TYPE, "
                "or INPUT_SIZE does not match the current state encoder; train a "
                "fresh checkpoint or use the code version that created this checkpoint."
            ) from exc
        self.update_target()

    @torch.no_grad()
    def compute_heatmap(self, env):
        gw = env.width // GRID_SIZE
        gh = env.height // GRID_SIZE

        # 1) gather states
        orig = tuple(env.pacman_pos)
        states = []
        for i in range(gw):
            for j in range(gh):
                env.pacman_pos = [i * GRID_SIZE, j * GRID_SIZE]
                s = env.get_state()[0]
                states.append(s)
        env.pacman_pos = list(orig)

        # 2) batch forward  —— stack single frame K times along feature-dim
        was_training = self.policy_net.training
        self.policy_net.eval()
        batch = torch.stack(states, dim=0).to(self.device)  # [N, BASE_FEAT_DIM]
        # K-frame approximate stacking: [N, BASE_FEAT_DIM*K]
        batch = torch.cat([batch] * self.k, dim=1)
        qvals = self.policy_net(batch)  # [N, A]
        self.policy_net.train(was_training)

        # 3) reshape back to grid
        max_q = qvals.max(dim=1).values  # [gw*gh]
        heatmap = max_q.view(gw, gh).cpu().numpy()
        return heatmap
