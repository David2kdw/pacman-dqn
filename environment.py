from collections import deque

import pygame
import random
import numpy as np
import torch
from copy import deepcopy
from config import MAZE, GRID_SIZE, WALL_VAL, DOT_VAL, ENEMY_VAL, EMPTY_VAL, PACMAN_VAL

# ====== Tunable rewards / shaping weights ======
R_DEATH = -15.0  # collision with an enemy
R_CLEAR = +25.0  # all dots cleared
R_TIMEOUT = -1.0  # episode ended by time/step limit
R_DOT = +1.0  # eat a dot
LIVING_COST = -0.01  # per-step cost
WALL_BUMP = -1.0  # bump into a wall (no movement)
BACKTRACK = -0.05  # reverse direction (helps reduce oscillation)
GAMMA_SHAPING = 0.99  # discount factor used in potential-based shaping
LAMBDA_DOT = 0.10  # weight for dot shaping
LAMBDA_ENEMY = 0.06  # weight for enemy shaping
ADJ_ENEMY_PEN = -2  # small penalty when adjacent to an enemy (optional)

ACTION_DELTAS = {
    0: (-GRID_SIZE, 0),
    1: (GRID_SIZE, 0),
    2: (0, -GRID_SIZE),
    3: (0, GRID_SIZE),
}



class Environment:
    """
    Pac-Man environment for DQN training.
    Encapsulates maze layout, entity positions, state encoding, and reward logic.
    """

    def __init__(self):
        """
        Build the static maze and record initial dot locations.
        Fix Pac-Man and enemy spawn points (center + four corners).
        Finally, call reset() to initialize the episode state.
        """
        self.maze = MAZE.splitlines()
        self.grid_w = len(self.maze[0])
        self.grid_h = len(self.maze)

        self.walls, self.init_dots, self.width, self.height = self.build_maze(MAZE, GRID_SIZE)

        self.pacman_start = self._find_fixed_center_spawn()  # [px, py]
        self.enemy_count = 4
        self.enemy_starts = self._find_corner_spawns()  # [[ex, ey, dx, dy], ...] 长度应为4
        if len(self.enemy_starts) != self.enemy_count:
            self.enemy_starts = self.enemy_starts[:self.enemy_count]

        self.reset()

    def build_maze(self, maze_layout: str, grid: int):
        """
        Parse an ASCII maze into wall rectangles and dot locations.
        Returns:
          - walls: list of pygame.Rect
          - dots: list of (x, y) tuples
          - width, height: overall pixel dimensions
        """
        walls, dots = [], []
        rows = maze_layout.splitlines()
        h, w = len(rows), len(rows[0])
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                x, y = i * grid, j * grid
                if ch == "#":
                    walls.append(pygame.Rect(x, y, grid, grid))
                elif ch == ".":
                    dots.append((x, y))
        return walls, dots, w * grid, h * grid

    def _get_valid_spawn(self):
        """
        Pick a random (x, y) not colliding with walls or Pac-Man’s start.
        """
        while True:
            gx = random.randint(1, (self.width // GRID_SIZE) - 2)
            gy = random.randint(1, (self.height // GRID_SIZE) - 2)
            x, y = gx * GRID_SIZE, gy * GRID_SIZE
            r = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            if not any(r.colliderect(w) for w in self.walls) and (x, y) != self.pacman_start:
                return x, y
    
    def _get_valid_spawn_player(self):
        """Find a random position for the player."""
        while True:
            x = random.randint(1, (self.width // GRID_SIZE) - 2) * GRID_SIZE
            y = random.randint(1, (self.height // GRID_SIZE) - 2) * GRID_SIZE
            new_rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            if not any(new_rect.colliderect(w) for w in self.walls):
                return x, y

    def reset(self):
        """
        Begin a new episode with fixed spawns:
          1) Restore dots from the initial copy.
          2) Place Pac-Man at fixed center spawn; zero velocity.
          3) Place enemies at fixed corner spawns; zero velocity (由 helper 返回).
        Returns:
          initial state tensor via get_state()
        """
        self.dots = deepcopy(self.init_dots)

        self.pacman_pos = list(self.pacman_start)
        self.pacman_dx = 0
        self.pacman_dy = 0
        self.prev_pacman_dx = 0
        self.prev_pacman_dy = 0
        self.last_event = {
            "wall_bump": False,
            "ate_dot": False,
            "terminal_reason": None,
        }

        self.enemies = [[ex, ey, dx, dy] for (ex, ey, dx, dy) in self.enemy_starts]

        return self.get_state()

    def _pixel_hits_wall(self, x, y):
        r = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        return any(r.colliderect(w) for w in self.walls)

    def valid_actions(self):
        """
        Return actions that keep Pac-Man inside walkable cells.
        """
        actions = []
        x, y = self.pacman_pos
        for action, (dx, dy) in ACTION_DELTAS.items():
            if not self._pixel_hits_wall(x + dx, y + dy):
                actions.append(action)
        return actions or list(ACTION_DELTAS)

    def _is_walkable(self, gx, gy):
        if gx < 0 or gy < 0 or gx >= self.grid_w or gy >= self.grid_h:
            return False
        return self.maze[gy][gx] != '#'

    def _to_pixel(self, gx, gy):
        return gx * GRID_SIZE, gy * GRID_SIZE

    def _nearest_walkable(self, sx, sy):
        if self._is_walkable(sx, sy):
            return sx, sy
        q = deque([(sx, sy)])
        seen = {(sx, sy)}
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, y = q.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (nx, ny) in seen:
                    continue
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    if self._is_walkable(nx, ny):
                        return nx, ny
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return sx, sy

    def _find_fixed_center_spawn(self):
        cx, cy = self.grid_w // 2, self.grid_h // 2
        gx, gy = self._nearest_walkable(cx, cy)
        return list(self._to_pixel(gx, gy))

    def _find_corner_spawns(self):
        candidates = [
            (1, 1),
            (1, self.grid_h - 2),
            (self.grid_w - 2, 1),
            (self.grid_w - 2, self.grid_h - 2),
        ]
        starts = []
        for gx, gy in candidates:
            ngx, ngy = self._nearest_walkable(gx, gy)
            px, py = self._to_pixel(ngx, ngy)
            starts.append([px, py, 0, 0])  # (ex, ey, dx, dy)
        return starts

    def step(self, action: int):
        """
        Apply one time-step:
          - Move Pac-Man based on action (0=←,1=→,2=↑,3=↓).
          - Check for wall collision (revert if needed).
          - Remove eaten dot if present.
          - Check terminal events and compute reward.
          - Move each enemy one random valid step.
        Returns:
          next_state (torch.Tensor), reward (float), done (bool)
        """
        old_x, old_y = self.pacman_pos.copy()
        prev_dx, prev_dy = self.pacman_dx, self.pacman_dy
        self.last_event = {
            "wall_bump": False,
            "ate_dot": False,
            "terminal_reason": None,
        }

        # 1. Move Pac-Man & record direction
        dx, dy = ACTION_DELTAS.get(action, (0, 0))
        self.pacman_pos[0] += dx
        self.pacman_pos[1] += dy
        self.pacman_dx, self.pacman_dy = np.sign(dx).item(), np.sign(dy).item()

        # 2. Wall collision?
        wall_bump = self._pixel_hits_wall(self.pacman_pos[0], self.pacman_pos[1])
        if wall_bump:
            self.pacman_pos = [old_x, old_y]
            self.last_event["wall_bump"] = True

        # 3. Dot eaten?
        pos = tuple(self.pacman_pos)
        ate_dot = pos in self.dots
        if pos in self.dots:
            self.dots.remove(pos)
        self.last_event["ate_dot"] = ate_dot

        # 4. Terminal event and reward
        death = any((self.pacman_pos[0] == e[0] and self.pacman_pos[1] == e[1])
                    for e in self.enemies)
        cleared = len(self.dots) == 0
        done = death or cleared
        terminal_reason = "death" if death else "clear" if cleared else None
        self.last_event["terminal_reason"] = terminal_reason
        reward = self.compute_reward(old_x, old_y,
                                     self.pacman_pos[0], self.pacman_pos[1],
                                     done,
                                     ate_dot=ate_dot,
                                     terminal_reason=terminal_reason,
                                     prev_dir=(prev_dx, prev_dy),
                                     wall_bump=wall_bump)

        self.prev_pacman_dx, self.prev_pacman_dy = self.pacman_dx, self.pacman_dy

        if done:
            return self.get_state(), reward, done

        # 6. Move enemies (one random valid shift each)
        for enemy in self.enemies:
            moved = False
            for _ in range(4):
                if random.random() < 0.5:
                    dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), 0
                else:
                    dx, dy = 0, random.choice([-GRID_SIZE, GRID_SIZE])
                nx, ny = enemy[0] + dx, enemy[1] + dy
                re = pygame.Rect(nx, ny, GRID_SIZE, GRID_SIZE)
                if not any(re.colliderect(w) for w in self.walls):
                    enemy[:] = [nx, ny, dx, dy]
                    moved = True
                    break
            if not moved:
                # reverse if stuck
                enemy[2], enemy[3] = -enemy[2], -enemy[3]

        # 7. New observation
        if self._collision_with_enemy(self.pacman_pos[0], self.pacman_pos[1]):
            self.last_event["terminal_reason"] = "death"
            return self.get_state(), float(R_DEATH), True

        next_state = self.get_state()
        return next_state, reward, done

    def get_state(self):
        """
        Encode the current maze + entities into a tensor:
          - A flattened grid of size (W×H) with values
            WALL_VAL, DOT_VAL, ENEMY_VAL, EMPTY_VAL, PACMAN_VAL
            :contentReference[oaicite:0]{index=0}
          - A 4-element one-hot of Pac-Man’s direction.
        Returns:
          torch.Tensor shape [1, W×H + 4]
        """
        gw, gh = self.width // GRID_SIZE, self.height // GRID_SIZE
        grid = np.zeros((gw, gh), dtype=float)

        # walls
        for w in self.walls:
            xs, ys = w.x//GRID_SIZE, w.y//GRID_SIZE
            xe, ye = (w.x+w.width)//GRID_SIZE, (w.y+w.height)//GRID_SIZE
            for x in range(xs, xe):
                for y in range(ys, ye):
                    grid[x, y] = WALL_VAL

        # dots
        for x, y in self.dots:
            gx, gy = x//GRID_SIZE, y//GRID_SIZE
            if grid[gx, gy] == EMPTY_VAL:
                grid[gx, gy] = DOT_VAL

        # enemies
        for ex, ey, _, _ in self.enemies:
            gx, gy = ex//GRID_SIZE, ey//GRID_SIZE
            if 0 <= gx < gw and 0 <= gy < gh and grid[gx, gy] in (EMPTY_VAL, DOT_VAL):
                grid[gx, gy] = ENEMY_VAL

        # Pac-Man
        px, py = self.pacman_pos[0]//GRID_SIZE, self.pacman_pos[1]//GRID_SIZE
        grid[px, py] = PACMAN_VAL

        # to tensor
        t_grid = torch.tensor(grid.flatten(), dtype=torch.float32).unsqueeze(0)

        # direction one-hot
        dir_oh = [0,0,0,0]
        if   self.pacman_dx == -1: dir_oh[0]=1
        elif self.pacman_dx ==  1: dir_oh[1]=1
        elif self.pacman_dy == -1: dir_oh[2]=1
        elif self.pacman_dy ==  1: dir_oh[3]=1
        t_dir = torch.tensor([dir_oh], dtype=torch.float32)

        return torch.cat((t_grid, t_dir), dim=1)


    # -----------------------------------------------
    # Helpers: grid coordinates and BFS shortest-path
    # -----------------------------------------------
    def _grid_coords(self, x, y):
        """Convert pixel coordinates to grid coordinates."""
        return x // GRID_SIZE, y // GRID_SIZE

    def _enemy_cells(self):
        """Return a set of enemy grid coordinates (gx, gy)."""
        return {(ex // GRID_SIZE, ey // GRID_SIZE) for (ex, ey, _, _) in self.enemies}

    def _dot_cells(self):
        """Return a set of dot grid coordinates (gx, gy)."""
        return {(dx // GRID_SIZE, dy // GRID_SIZE) for (dx, dy) in self.dots}

    def _collision_with_enemy(self, x, y):
        """Check if Pac-Man is on the same pixel cell as any enemy."""
        for ex, ey, _, _ in self.enemies:
            if (ex, ey) == (x, y):
                return True
        return False

    def _grid_distance_to_set(self, targets):
        """
        Multi-source BFS over the walkable grid.
        Returns a float32 array of shape [grid_h, grid_w] giving the shortest
        number of steps to the nearest target cell. Unreachable cells are inf.
        """
        dist = np.full((self.grid_h, self.grid_w), np.inf, dtype=np.float32)
        if not targets:
            return dist

        q = deque()
        for gx, gy in targets:
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h and self._is_walkable(gx, gy):
                dist[gy, gx] = 0.0
                q.append((gx, gy))

        moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while q:
            x, y = q.popleft()
            d0 = dist[y, x]
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and self._is_walkable(nx, ny):
                    if dist[ny, nx] > d0 + 1:
                        dist[ny, nx] = d0 + 1
                        q.append((nx, ny))
        return dist

    def compute_reward(
        self,
        old_x,
        old_y,
        new_x,
        new_y,
        done: bool,
        ate_dot: bool = False,
        terminal_reason: str | None = None,
        prev_dir: tuple[int, int] | None = None,
        wall_bump: bool = False,
    ):
        """
        Reward = terminal event + per-step events + potential-based shaping.
        Distances in shaping are shortest-path grid distances (via BFS), not Manhattan.
        Positions passed in are pixel coordinates; they are converted to grid cells.
        """
        # 1) Terminal events
        if done:
            if terminal_reason == "clear":
                return float(R_CLEAR)
            elif terminal_reason == "death" or self._collision_with_enemy(new_x, new_y):
                return float(R_DEATH)
            else:
                return float(R_TIMEOUT)

        r = 0.0

        # 2) Instantaneous events
        r += LIVING_COST

        # Dot eaten (note: at this point env may not have removed the dot yet)
        if ate_dot or (new_x, new_y) in self.dots:
            r += R_DOT

        # Wall bump / no movement
        if wall_bump or (new_x, new_y) == (old_x, old_y):
            r += WALL_BUMP

        # Reverse direction penalty (reduces oscillation)
        ndx, ndy = new_x - old_x, new_y - old_y
        if (ndx, ndy) != (0, 0):
            if prev_dir is None:
                prev_dir = (
                    getattr(self, "prev_pacman_dx", 0),
                    getattr(self, "prev_pacman_dy", 0),
                )
            prev_dx = np.sign(prev_dir[0])
            prev_dy = np.sign(prev_dir[1])
            cur_dx = np.sign(ndx)
            cur_dy = np.sign(ndy)
            if (prev_dx != 0 and cur_dx == -prev_dx) or (prev_dy != 0 and cur_dy == -prev_dy):
                r += BACKTRACK

        # 3) Potential-based shaping (does not change the optimal policy)
        gamma = GAMMA_SHAPING

        old_g = self._grid_coords(old_x, old_y)
        new_g = self._grid_coords(new_x, new_y)

        # Toward the nearest dot: Phi_dot(s) = - dist_to_nearest_dot_in_cells
        dot_cells = self._dot_cells()
        if dot_cells:
            dist_dot = self._grid_distance_to_set(dot_cells)
            old_dd = dist_dot[old_g[1], old_g[0]]
            new_dd = dist_dot[new_g[1], new_g[0]]
            if np.isfinite(old_dd) and np.isfinite(new_dd):
                r += LAMBDA_DOT * (gamma * (-new_dd) - (-old_dd))

        # Away from the nearest enemy: Phi_enemy(s) = + dist_to_nearest_enemy_in_cells
        enemy_cells = self._enemy_cells()
        if enemy_cells:
            dist_enemy = self._grid_distance_to_set(enemy_cells)
            old_de = dist_enemy[old_g[1], old_g[0]]
            new_de = dist_enemy[new_g[1], new_g[0]]
            if np.isfinite(old_de) and np.isfinite(new_de):
                r += LAMBDA_ENEMY * (gamma * (new_de) - (old_de))
                # Optional: small penalty when adjacent to an enemy (1 cell away)
                if new_de == 1:
                    r += ADJ_ENEMY_PEN

        return float(r)

