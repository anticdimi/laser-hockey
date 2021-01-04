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


def custom_discrete_to_continuous_action(action_id):
    return CUSTOM_DISCRETE_ACTIONS[action_id]
