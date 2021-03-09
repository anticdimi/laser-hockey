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
    [0, 0, 0, 0],       # 0 stand
    [-1, 0, 0, 0],      # 1 left
    [1, 0, 0, 0],       # 2 right
    [0, -1, 0, 0],      # 3 down
    [0, 1, 0, 0],       # 4 up
    [0, 0, -1, 0],      # 5 clockwise
    [0, 0, 1, 0],       # 6 counter-clockwise
    [0, 0, 0, 1]        # 7 shoot
]

REDUCED_CUSTOM_DISCRETE_ACTIONS = [
    [0, 0, 0, 0],       # 0 stand
    [-1, 0, 0, 0],      # 1 left
    [1, 0, 0, 0],       # 2 right
    [0, -1, 0, 0],      # 3 down
    [0, 1, 0, 0],       # 4 up
    [0, 0, -1, 0],      # 5 clockwise
    [0, 0, 1, 0],       # 6 counter-clockwise
    [-1, -1, 0, 0],     # 7 left down
    [-1, 1, 0, 0],      # 8 left up
    [1, -1, 0, 0],      # 9 right down
    [1, 1, 0, 0],       # 10 right up
    [-1, -1, -1, 0],    # 11 left down clockwise
    [-1, -1, 1, 0],     # 12 left down counter-clockwise
    [-1, 1, -1, 0],     # 13 left up clockwise
    [-1, 1, 1, 0],      # 14 left up counter-clockwise
    [1, -1, -1, 0],     # 15 right down clockwise
    [1, -1, 1, 0],      # 16 right down counter-clockwise
    [1, 1, -1, 0],      # 17 right up clockwise
    [1, 1, 1, 0],       # 18 right up counter-clockwise
    [0, 0, 0, 1],       # 19 shoot
]

