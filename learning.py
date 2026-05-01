import random
from replayMemory import ReplayMemory
import torch
import torch.nn as nn

train_step = 0  # Global counter for tracking updates
losses = []  # Store loss history
q_value_logs = []  # Store Q-value history


def _unpack_transition_batch(batch):
    """
    Support both normal 5-field replay items and older 6-field items.
    Any stored legacy metadata is ignored.
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for item in batch:
        if len(item) == 6:
            s, a, r, s_next, _ignored_metadata, done = item
        else:
            s, a, r, s_next, done = item
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)
        dones.append(done)

    return states, actions, rewards, next_states, dones

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
    states, actions, rewards, next_states, dones = _unpack_transition_batch(batch)

    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states, dim=0)
    actions = torch.as_tensor(actions, dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.bool)

    device = next(online_model.parameters()).device
    states, next_states = states.to(device), next_states.to(device)
    actions, rewards, dones = actions.to(device), rewards.to(device), dones.to(device)
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
        target_q_values = rewards + gamma * next_target_q * (~dones)
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
        print(f"\n🔹 TRAINING UPDATE {train_step}")
        print(f"   ✅ Average Loss (Last 100): {_np.mean(losses[last]):.5f}")
        print(f"   ✅ Average Q: {_np.mean(q_value_logs[last]):.3f} "
              f"(Min: {_np.min(q_value_logs[last]):.3f}, Max: {_np.max(q_value_logs[last]):.3f})")
        print(f"   ✅ Termination Samples in batch: {n_term}")

        # Debug examples (one ongoing and one terminal if present)
        if n_term < batch_size:
            # pick a non-terminal
            idx = int((~dones).nonzero(as_tuple=True)[0][0].item())
            print(f"   ✅ Ongoing Sample: Q={q_values[idx].item():.3f} | "
                  f"T={target_q_values[idx].item():.3f}")
        if n_term > 0:
            idx = int(dones.nonzero(as_tuple=True)[0][0].item())
            print(f"   ❌ Terminal Sample: Q={q_values[idx].item():.3f} | "
                  f"T={target_q_values[idx].item():.3f}")
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



    

