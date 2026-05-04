from typing import Optional, Sequence, Tuple, List
import numpy as np
import torch

class ReplayMemory:
    """
    A general-purpose replay buffer with optional Prioritized Experience Replay (PER).

    Compatibility:
        - sample(batch_size, beta=...) -> (batch, indices, weights_tensor)
        - update_priorities(indices, td_errors)

    Notes:
        * When alpha == 0.0 the buffer reduces to uniform sampling (no prioritization).
        * New transitions receive the current maximum priority by default to ensure
          they can be sampled at least once (standard PER practice).
        * Terminal-enforcement is optional via `min_terminal_samples` at init-time
          (set None to disable). This is domain-agnostic and simply relies on the
          `done` flag stored with each transition.
    """
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        min_terminal_samples: Optional[int] = None
    ) -> None:
        assert capacity > 0, "capacity must be positive"
        assert alpha >= 0.0, "alpha must be non-negative"
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.min_terminal_samples = (int(min_terminal_samples)
                                     if min_terminal_samples is not None else None)

        # Ring buffer storage
        self.storage: List[Optional[Tuple]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

        self.next_idx: int = 0   # position to write next transition
        self.size: int = 0       # number of valid entries

    # ----- properties -----
    def __len__(self) -> int:
        return self.size

    def is_full(self) -> bool:
        return self.size == self.capacity

    # ----- core ops -----
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        bootstrap_discount: Optional[float] = None,
        priority_override: Optional[float] = None,
    ) -> None:
        """
        Insert a transition. If `priority_override` is None, the new item will receive
        the current maximum priority (or 1.0 if buffer is empty).
        """
        if bootstrap_discount is None:
            self.storage[self.next_idx] = (state, action, reward, next_state, done)
        else:
            self.storage[self.next_idx] = (
                state,
                action,
                reward,
                next_state,
                float(bootstrap_discount),
                done,
            )

        if priority_override is None:
            max_pr = self.priorities[:self.size].max() if self.size > 0 else 1.0
            pr = max(1.0, float(max_pr))
        else:
            pr = float(priority_override)

        self.priorities[self.next_idx] = pr

        # Advance ring pointer
        self.next_idx = (self.next_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def _valid_indices(self) -> np.ndarray:
        """Return numpy array of valid absolute indices [0, size)."""
        return np.arange(self.size, dtype=np.int64)

    def _terminal_mask(self) -> np.ndarray:
        """Boolean mask of terminal transitions within the valid range."""
        mask = np.zeros(self.size, dtype=bool)
        for i in range(self.size):
            item = self.storage[i]
            if item is not None and item[-1]:  # done flag
                mask[i] = True
        return mask

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[List[Tuple], np.ndarray, torch.Tensor]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: number of samples to draw.
            beta: strength of importance-sampling correction in [0, 1].
                  (beta=0 -> no correction; beta=1 -> full correction)

        Returns:
            samples: list of transitions (state, action, reward, next_state, done)
            indices: numpy array of absolute indices in the buffer
            weights: torch tensor of importance-sampling weights (shape [batch])
        """
        assert self.size > 0, "Cannot sample from an empty buffer"
        assert batch_size > 0, "batch_size must be positive"

        n = self.size
        batch_size = min(batch_size, n)  # safeguard

        valid_idx = self._valid_indices()

        # Compute sampling probabilities
        if self.alpha == 0.0:
            probs_all = np.ones(n, dtype=np.float32) / n
        else:
            # Priorities for valid range (clip for numerical safety)
            pr = np.clip(self.priorities[:n], a_min=self.epsilon, a_max=None)
            probs_all = pr ** self.alpha
            probs_all /= probs_all.sum()

        # (Optional) ensure a minimum number of terminal transitions
        chosen_indices: List[int] = []
        if self.min_terminal_samples is not None and self.min_terminal_samples > 0:
            term_mask = self._terminal_mask()
            term_indices = valid_idx[term_mask]
            num_term = min(len(term_indices), self.min_terminal_samples, batch_size)
            if num_term > 0:
                # Uniform within terminals (simple and unbiased within subset)
                term_pick = np.random.choice(term_indices, size=num_term, replace=False)
                chosen_indices.extend(term_pick.tolist())

                # Exclude chosen terminals for the remaining picks
                remain_mask = np.ones(n, dtype=bool)
                remain_mask[term_pick] = False
                valid_remain = valid_idx[remain_mask]
                probs_remain = probs_all[remain_mask]
                # Normalize in case some mass was removed
                probs_remain /= probs_remain.sum()

                rest = batch_size - num_term
                if rest > 0:
                    replace_flag = rest > len(valid_remain)
                    rest_pick = np.random.choice(valid_remain, size=rest, replace=replace_flag, p=probs_remain)
                    chosen_indices.extend(rest_pick.tolist())
            else:
                # Not enough terminals; fall back to pure prioritized sampling
                replace_flag = batch_size > n
                chosen_indices = np.random.choice(valid_idx, size=batch_size, replace=replace_flag, p=probs_all).tolist()
        else:
            # No terminal enforcement
            replace_flag = batch_size > n
            chosen_indices = np.random.choice(valid_idx, size=batch_size, replace=replace_flag, p=probs_all).tolist()

        indices = np.asarray(chosen_indices, dtype=np.int64)

        # Gather samples
        samples = [self.storage[i] for i in indices]

        # Compute IS weights (normalize by max to keep scale ~1)
        if beta is None:
            beta = 0.0
        beta = float(beta)
        if beta <= 0.0:
            weights = np.ones(len(indices), dtype=np.float32)
        else:
            chosen_probs = probs_all[indices]
            # Avoid division by zero (should be safe due to epsilon+clip)
            chosen_probs = np.clip(chosen_probs, a_min=self.epsilon, a_max=None)
            weights = (1.0 / (n * chosen_probs)) ** beta
            weights /= weights.max()  # normalize

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return samples, indices, weights_tensor

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        """
        Update priorities using TD errors. Standard rule:
            p_i = (|δ_i| + ε) ** 1   (then raised to alpha during sampling)
        We store "raw" priorities here; alpha is applied in `sample`.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return
        td_errors = np.asarray(td_errors, dtype=np.float32)
        raw_pr = np.abs(td_errors) + self.epsilon
        # Bound for numerical safety
        raw_pr = np.clip(raw_pr, a_min=self.epsilon, a_max=1e6)
        # Write back
        for idx, pr in zip(indices, raw_pr):
            if 0 <= idx < self.size:
                self.priorities[idx] = float(pr)

    # ----- utility -----
    def clear(self) -> None:
        self.storage = [None] * self.capacity
        self.priorities[:] = 0.0
        self.next_idx = 0
        self.size = 0
