def shooting_proxy(
    self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched=0
):
    constants = self._factors[self._config['mode']]

    reward_dict = {}
    reward_dict['closeness-reward'] = (1 - touched) * constants['factor_closeness'] * reward_closeness_to_puck
    reward_dict['existence-reward'] = (-1) * constants['factor_existence']
    reward_dict['outcome-reward'] = constants['factor_outcome'] * reward_game_outcome

    return reward_dict


def defense_proxy(self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction,
                  touched, step, max_allowed_steps):
    constants = self._factors[self._config['mode']]

    reward_dict = {}

    existence = 1

    if (1 <= env.player1.position[0] <= 2.5) and (2 <= env.player1.position[1] <= 6):
        reward_dict['existence-reward'] = existence
    else:
        reward_dict['existence-reward'] = -existence

    if reward_puck_direction < 0:
        reward_dict['closeness-reward'] = 10 * reward_closeness_to_puck
    elif reward_puck_direction == 0 and env.puck.position[0] < 5:
        reward_dict['closeness-reward'] = - 10 * existence

    if env.done:
        if env.winner == -1:
            reward_dict['outcome-reward'] = -50 - step * 0.5 * existence
        elif env.winner == 1:
            reward_dict['outcome-reward'] = (81 - step) * existence + 30    # Try + 60 factor

    # draw
    elif step == max_allowed_steps:
        reward_dict['outcome-reward'] = 0

    return reward_dict
