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
WIDTH  = len(_rows[0]) * GRID_SIZE
HEIGHT = len(_rows)    * GRID_SIZE
del _rows

# RL params
LR      = 0.001
GAMMA   = 0.99
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 10
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4
MEMORY_CAPACITY = 50000
BATCH_SIZE = 50
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.1

CHECKPOINT_DIR = "checkpoints/"
MEMORY_PATH = "checkpoints/memory.pkl"
MODEL_PATH     = "./model/dqn.pth"

WALL_VAL   = -1.0
ENEMY_VAL  = -0.5  
EMPTY_VAL  =  0.0 
DOT_VAL    = +0.5 
PACMAN_VAL = +1.0

MAX_STEPS_PER_EPISODE = 1000
MAX_EPISODE_TIME = 30

K_FRAMES = 3
BASE_FEAT_DIM = len(MAZE.splitlines()[0]) * len(MAZE.splitlines()) + 4
INPUT_SIZE = BASE_FEAT_DIM * K_FRAMES
