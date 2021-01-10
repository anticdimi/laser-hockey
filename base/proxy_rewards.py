def shooting_proxy(
    self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched=0
):
    constants = self._factors[self._config['mode']]

    reward_dict = {}
    reward_dict['closeness-reward'] = (1 - touched) * constants['factor_closeness'] * reward_closeness_to_puck
    reward_dict['existence-reward'] = (-1) * constants['factor_existence']
    reward_dict['outcome-reward'] = constants['factor_outcome'] * reward_game_outcome

    return reward_dict


def defense_proxy(
    self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched
):
    constants = self._factors[self._config['mode']]

    reward_dict = {}

    if (1 <= env.player1.position[0] <= 2.5) and (2 <= env.player1.position[1] <= 6):
        reward_dict['existence-reward'] = 1
    else:
        reward_dict['existence-reward'] = -1

    if reward_puck_direction < 0:
        reward_dict['closeness-reward'] = 130 * reward_closeness_to_puck

    if env.done:
        if env.winner == -1:
            reward_dict['outcome-reward'] = -25
        elif env.winner == 0:
            reward_dict['outcome-reward'] = 30

    return reward_dict
