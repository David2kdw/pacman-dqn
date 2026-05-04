import random
from replayMemory import ReplayMemory
import torch
import torch.nn as nn
from config import (
    GRID_W,
    GRID_H,
    K_FRAMES,
    STATE_GRID_CHANNELS,
    STATE_EXTRA_FEATURES,
)

train_step = 0  # Global counter for tracking updates
losses = []  # Store loss history
q_value_logs = []  # Store Q-value history


def _unpack_transition_batch(batch):
    """
    Support old 1-step replay items and newer n-step replay items.
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    discounts = []

    for item in batch:
        if len(item) == 6:
            s, a, r, s_next, discount, done = item
        else:
            s, a, r, s_next, done = item
            discount = None
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)
        dones.append(done)
        discounts.append(discount)

    return states, actions, rewards, next_states, dones, discounts

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.LeakyReLU(negative_slope=0.01)

        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.act4 = nn.LeakyReLU(negative_slope=0.01)

        self.output_layer = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # Flatten to (batch, -1)
        x = x.view(x.shape[0], -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return self.output_layer(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # ---- 共享干路，与原来保持一致 ----
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.LeakyReLU(negative_slope=0.01)

        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.act4 = nn.LeakyReLU(negative_slope=0.01)

        # ---- Dueling 头：Value 与 Advantage ----
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, 1)           # 标量 V(s)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, output_size) # 各动作的 A(s,a)
        )

        # 小技巧：最后一层置 0，开局 Q≈0 更稳
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.zeros_(self.adv_head[-1].weight)
        nn.init.zeros_(self.adv_head[-1].bias)

    def forward(self, x):
        # Flatten to (batch, -1)
        x = x.view(x.shape[0], -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))

        V = self.value_head(x)                 # (B, 1)
        A = self.adv_head(x)                   # (B, n_actions)
        A = A - A.mean(dim=1, keepdim=True)    # mean 归一，解决不唯一性
        Q = V + A                              # (B, n_actions)
        return Q


class DuelingCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = int(input_size)
        self.k_frames = K_FRAMES
        self.grid_w = GRID_W
        self.grid_h = GRID_H
        self.grid_channels = STATE_GRID_CHANNELS
        self.extra_per_frame = 4 + STATE_EXTRA_FEATURES
        self.grid_len = self.grid_w * self.grid_h * self.grid_channels
        self.frame_dim = self.grid_len + self.extra_per_frame

        expected_input = self.frame_dim * self.k_frames
        if self.input_size != expected_input:
            raise ValueError(
                f"DuelingCNN expected input_size={expected_input}, got {self.input_size}"
            )

        self.conv = nn.Sequential(
            nn.Conv2d(self.grid_channels * self.k_frames, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.grid_channels * self.k_frames, self.grid_h, self.grid_w)
            conv_out = self.conv(dummy).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(conv_out + self.extra_per_frame * self.k_frames, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, output_size)
        )

        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.zeros_(self.adv_head[-1].weight)
        nn.init.zeros_(self.adv_head[-1].bias)

    def _split_state(self, x):
        x = x.view(x.shape[0], self.k_frames, self.frame_dim)
        grid = x[:, :, :self.grid_len]
        extra = x[:, :, self.grid_len:]

        grid = grid.view(
            x.shape[0],
            self.k_frames,
            self.grid_channels,
            self.grid_w,
            self.grid_h,
        )
        grid = grid.permute(0, 1, 2, 4, 3).contiguous()
        grid = grid.view(
            x.shape[0],
            self.k_frames * self.grid_channels,
            self.grid_h,
            self.grid_w,
        )
        extra = extra.reshape(x.shape[0], self.k_frames * self.extra_per_frame)
        return grid, extra

    def forward(self, x):
        grid, extra = self._split_state(x)
        conv_features = self.conv(grid)
        features = torch.cat((conv_features, extra), dim=1)
        shared = self.shared(features)

        V = self.value_head(shared)
        A = self.adv_head(shared)
        A = A - A.mean(dim=1, keepdim=True)
        return V + A



    
def train_dqn(
    online_model: nn.Module,
    target_model: nn.Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32,
    gamma: float = 0.99,
    beta: float = 0.3,
    max_grad_norm: float | None = 5.0,
) -> None:
    """
    One optimization step of Double DQN with PER.

    Improvements vs. baseline:
      - Deterministic target computation: temporarily switches models to eval()
      - Gradient clipping (optional): avoids occasional exploding gradients
      - Robust logging: uses tensor ops to count terminals
    """
    global train_step, losses, q_value_logs

    # 1) Need enough samples
    if len(memory) < batch_size:
        return

    # 2) Sample from replay
    batch, indices, weights = memory.sample(batch_size, beta=beta)
    states, actions, rewards, next_states, dones, discounts = _unpack_transition_batch(batch)

    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states, dim=0)
    actions = torch.as_tensor(actions, dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.bool)
    discounts = torch.as_tensor(
        [gamma if d is None else float(d) for d in discounts],
        dtype=torch.float32,
    )

    device = next(online_model.parameters()).device
    states, next_states = states.to(device), next_states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    discounts = discounts.to(device)
    weights = weights.to(device)

    # 3) Q(s,a) for actions taken
    q_values = online_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 4) Double DQN target (temporarily eval for stability)
    online_mode, target_mode = online_model.training, target_model.training
    with torch.no_grad():
        online_model.eval()
        target_model.eval()
        next_actions = online_model(next_states).argmax(dim=1, keepdim=True)
        next_target_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + discounts * next_target_q * (~dones)
    online_model.train(online_mode)
    target_model.train(target_mode)

    # 5) IS-weighted loss
    loss_fn = nn.SmoothL1Loss(reduction='none')
    per_sample = loss_fn(q_values, target_q_values)
    loss = (per_sample * weights.detach()).mean()

    # 6) Optimize + (optional) gradient clipping
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(online_model.parameters(), max_grad_norm)
    optimizer.step()

    # 7) Update PER priorities
    td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors)

    # 8) Logging
    losses.append(loss.item())
    q_value_logs.append(q_values.mean().item())
    train_step += 1

    if train_step % 100 == 0:
        import numpy as _np
        last = slice(-100, None)
        n_term = int(dones.sum().item())
        q_np = q_values.detach().cpu().numpy()
        target_np = target_q_values.detach().cpu().numpy()
        td_np = td_errors
        reward_np = rewards.detach().cpu().numpy()
        bootstrap_np = next_target_q.detach().cpu().numpy()
        action_counts = torch.bincount(actions.detach().cpu(), minlength=4).tolist()
        latest_loss = losses[-1]
        avg_loss = _np.mean(losses[last])

        print(f"\n🔹 DQN UPDATE {train_step} | loss={latest_loss:.5f} avg100={avg_loss:.5f}")
        print(
            f"   Q taken mean/min/max: "
            f"{_np.mean(q_np):.3f} / {_np.min(q_np):.3f} / {_np.max(q_np):.3f}"
        )
        print(
            f"   Target mean/min/max: "
            f"{_np.mean(target_np):.3f} / {_np.min(target_np):.3f} / {_np.max(target_np):.3f}"
        )
        print(
            f"   TD error mean/p95/max: "
            f"{_np.mean(td_np):.3f} / {_np.percentile(td_np, 95):.3f} / {_np.max(td_np):.3f}"
        )
        print(
            f"   Reward mean/min/max: "
            f"{_np.mean(reward_np):.3f} / {_np.min(reward_np):.3f} / {_np.max(reward_np):.3f}"
        )
        print(
            f"   Bootstrap Q mean/max: "
            f"{_np.mean(bootstrap_np):.3f} / {_np.max(bootstrap_np):.3f}"
        )
        print(f"   Batch terminal={n_term}/{batch_size} | actions={action_counts}")
        print('-' * 50)


def train_sarsa_n_step(
    policy_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions,
    gamma: float = 0.99,
    max_grad_norm: float | None = 5.0,
) -> None:
    """
    One on-policy n-step Deep SARSA update.

    `transitions` is an ordered episode-local sequence of
    (state, action, reward, next_state, next_action, done).
    The bootstrap action must be the action actually selected by the behavior
    policy, not an argmax computed during training.
    """
    global train_step, losses, q_value_logs

    if not transitions:
        return

    state, action, _, _, _, _ = transitions[0]
    reward_sum = 0.0
    bootstrap_state = None
    bootstrap_action = None
    terminal = False
    horizon = 0

    for i, (_, _, reward, next_state, next_action, done) in enumerate(transitions):
        reward_sum += (gamma ** i) * float(reward)
        bootstrap_state = next_state
        bootstrap_action = next_action
        terminal = bool(done)
        horizon = i + 1
        if terminal:
            break

    device = next(policy_model.parameters()).device
    state = state.to(device)
    action_tensor = torch.as_tensor([action], dtype=torch.long, device=device)
    target_value = torch.as_tensor([reward_sum], dtype=torch.float32, device=device)

    policy_mode, target_mode = policy_model.training, target_model.training
    with torch.no_grad():
        target_model.eval()
        if not terminal and bootstrap_state is not None and bootstrap_action is not None:
            bootstrap_state = bootstrap_state.to(device)
            bootstrap_action_tensor = torch.as_tensor(
                [[bootstrap_action]], dtype=torch.long, device=device
            )
            bootstrap_q = target_model(bootstrap_state).gather(1, bootstrap_action_tensor).squeeze(1)
            target_value = target_value + (gamma ** horizon) * bootstrap_q
    target_model.train(target_mode)

    policy_model.train(policy_mode)
    q_value = policy_model(state).gather(1, action_tensor.view(1, 1)).squeeze(1)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(q_value, target_value)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
    optimizer.step()

    losses.append(loss.item())
    q_value_logs.append(q_value.mean().item())
    train_step += 1

    if train_step % 100 == 0:
        import numpy as _np
        last = slice(-100, None)
        print(f"\n🔹 SARSA UPDATE {train_step}")
        print(f"   loss={loss.item():.5f} avg100={_np.mean(losses[last]):.5f}")
        print(
            f"   Q avg100/min/max: {_np.mean(q_value_logs[last]):.3f} / "
            f"{_np.min(q_value_logs[last]):.3f} / {_np.max(q_value_logs[last]):.3f}"
        )
        print(
            f"   Current sample: Q={q_value.item():.3f} | "
            f"target={target_value.item():.3f} | "
            f"td_error={abs(q_value.item() - target_value.item()):.3f}"
        )
        print(f"   Horizon={horizon} | Terminal={terminal} | Action={action}")
        print('-' * 50)

def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        with torch.no_grad():
            model.eval()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = model(state)
            model.train()
            return q_values.argmax(dim=1).item()



    

