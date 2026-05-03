MAZE = """
############################
#............##............#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#..........................#
#.####.##.########.##.####.#
#......##....##....##......#
#.####.#####.##.#####.##.###
#.####.#          #......###
#.####.# ######## #.####.###
#.####.  ########  .####.###
#.####.# ######## #.####.###
#.####.#          #.####.###
#.####.# ######## #.####.###
#............##............#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#..........................#
############################
""".strip("\n")

GRID_SIZE = 20
_rows = MAZE.splitlines()
GRID_W = len(_rows[0])
GRID_H = len(_rows)
WIDTH  = len(_rows[0]) * GRID_SIZE
HEIGHT = len(_rows)    * GRID_SIZE
del _rows

# RL params
LR      = 0.0003
GAMMA   = 0.99
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 10
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4
MODEL_TYPE = "cnn"
MEMORY_CAPACITY = 50000
BATCH_SIZE = 50
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.1
VISUALIZE_TRAINING = False

REWARD_PROFILES = {
    "baseline": {
        "R_DEATH": -15.0,
        "R_CLEAR": 25.0,
        "R_TIMEOUT": -1.0,
        "R_DOT": 1.0,
        "LIVING_COST": -0.01,
        "WALL_BUMP": -1.0,
        "GAMMA_SHAPING": 0.99,
        "LAMBDA_DOT": 0.10,
        "LAMBDA_ENEMY": 0.06,
        "ADJ_ENEMY_PEN": -2.0,
    },
    "eat_focused": {
        "R_DEATH": -15.0,
        "R_CLEAR": 25.0,
        "R_TIMEOUT": -1.0,
        "R_DOT": 1.3,
        "LIVING_COST": -0.003,
        "WALL_BUMP": -1.0,
        "GAMMA_SHAPING": 0.99,
        "LAMBDA_DOT": 0.06,
        "LAMBDA_ENEMY": 0.06,
        "ADJ_ENEMY_PEN": -2.0,
    },
    "wall_focused": {
        "R_DEATH": -15.0,
        "R_CLEAR": 25.0,
        "R_TIMEOUT": -1.0,
        "R_DOT": 1.0,
        "LIVING_COST": -0.003,
        "WALL_BUMP": -1.5,
        "GAMMA_SHAPING": 0.99,
        "LAMBDA_DOT": 0.06,
        "LAMBDA_ENEMY": 0.06,
        "ADJ_ENEMY_PEN": -2.0,
    },
    "survival_focused": {
        "R_DEATH": -25.0,
        "R_CLEAR": 25.0,
        "R_TIMEOUT": -1.0,
        "R_DOT": 1.0,
        "LIVING_COST": -0.002,
        "WALL_BUMP": -1.0,
        "GAMMA_SHAPING": 0.99,
        "LAMBDA_DOT": 0.05,
        "LAMBDA_ENEMY": 0.10,
        "ADJ_ENEMY_PEN": -3.0,
    },
    "balanced": {
        "R_DEATH": -20.0,
        "R_CLEAR": 25.0,
        "R_TIMEOUT": -1.0,
        "R_DOT": 1.2,
        "LIVING_COST": -0.003,
        "WALL_BUMP": -1.2,
        "GAMMA_SHAPING": 0.99,
        "LAMBDA_DOT": 0.06,
        "LAMBDA_ENEMY": 0.08,
        "ADJ_ENEMY_PEN": -2.5,
    },
}

CHECKPOINT_DIR = "checkpoints/"
MEMORY_PATH = "checkpoints/memory.pkl"
MEMORY_SAVE_EVERY_EPISODES = 500
MODEL_PATH     = "./model/dqn.pth"

WALL_VAL   = -1.0
ENEMY_VAL  = -0.5  
EMPTY_VAL  =  0.0 
DOT_VAL    = +0.5 
PACMAN_VAL = +1.0

MAX_STEPS_PER_EPISODE = 1000
MAX_EPISODE_TIME = 30

K_FRAMES = 3
STATE_GRID_CHANNELS = 4
STATE_EXTRA_FEATURES = 10
CNN_CHANNELS = STATE_GRID_CHANNELS * K_FRAMES
EXTRA_FEATURES_TOTAL = (4 + STATE_EXTRA_FEATURES) * K_FRAMES
BASE_FEAT_DIM = (
    GRID_W * GRID_H * STATE_GRID_CHANNELS
    + 4
    + STATE_EXTRA_FEATURES
)
INPUT_SIZE = BASE_FEAT_DIM * K_FRAMES
