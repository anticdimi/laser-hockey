import itertools

BASIC_MOVEMENT_ACTIONS = [
    [-1, 0, 1],  # horizontal
    [-1, 0, 1],  # vertical
    [-1, 0, 1],  # rotation
]

CUSTOM_DISCRETE_ACTIONS = []

for element in itertools.product(*BASIC_MOVEMENT_ACTIONS):
    CUSTOM_DISCRETE_ACTIONS.append(element + (0,))  # movement + no-shooting
CUSTOM_DISCRETE_ACTIONS.append((0, 0, 0, 1))  # shooting

DEFAULT_DISCRETE_ACTIONS = [
    [0, 0, 0, 0],       # stand
    [-1, 0, 0, 0],      # left
    [1, 0, 0, 0],       # right
    [0, -1, 0, 0],      # down
    [0, 1, 0, 0],       # up
    [0, 0, -1, 0],      # clockwise
    [0, 0, 1, 0],       # counter-clockwise
    [0, 0, 0, 1]        # shoot
]

REDUCED_CUSTOM_DISCRETE_ACTIONS = [
    [0, 0, 0, 0],       # stand
    [-1, 0, 0, 0],      # left
    [1, 0, 0, 0],       # right
    [0, -1, 0, 0],      # down
    [0, 1, 0, 0],       # up
    [0, 0, -1, 0],      # clockwise
    [0, 0, 1, 0],       # counter-clockwise
    [-1, -1, 0, 0],     # left down
    [-1, 1, 0, 0],      # left up
    [1, -1, 0, 0],      # right down
    [1, 1, 0, 0],       # right up
    [-1, -1, -1, 0],    # left down clockwise
    [-1, -1, 1, 0],     # left down counter-clockwise
    [-1, 1, -1, 0],     # left up clockwise
    [-1, 1, 1, 0],      # left up counter-clockwise
    [1, -1, -1, 0],     # right down clockwise
    [1, -1, 1, 0],      # right down counter-clockwise
    [1, 1, -1, 0],      # right up clockwise
    [1, 1, 1, 0],       # right up counter-clockwise
    [0, 0, 0, 1],       # shoot
]

