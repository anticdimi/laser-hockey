import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

from base.experience_replay import *


class Agent(object):
    """
    The Agent class implements a base class for agents.

    Parameters
    ----------
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_dim: int
        The variable specifies the dimension of action space vector.
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    **userconfig:
        The variable specifies the config settings.
    """

    def __init__(self, logger, obs_dim, action_dim, userconfig):
        self.logger = logger

        if userconfig['mode'] == 'normal':
            self.reward_function = None
        elif userconfig['mode'] == 'shooting':
            self.reward_function = self._shooting_reward
        elif userconfig['mode'] == 'defense':
            self.reward_function = self._defense_reward
        else:
            raise ValueError('Unknown training mode. See --help')

        self._config = {
            'discount': 0.95,
            'buffer_size': int(1e5),
            'batch_size': 128,
            'hidden_sizes': [128, 128],
        }
        self._config.update(userconfig)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if self._config['per']:
            self.buffer = PrioritizedExperienceReplay(max_size=self._config['buffer_size'],
                                                      alpha=self._config['per_alpha'],
                                                      beta=1 - self._config['epsilon'])
        else:
            self.buffer = UniformExperienceReplay(max_size=self._config['buffer_size'])

    def act(self, observation, eps=None):
        raise NotImplementedError('Implement act method.')

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def _shooting_reward(
        self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched=0
    ):
        raise NotImplementedError('Implement proxy reward methods.')

    def _defense_reward(
        self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched
    ):
        raise NotImplementedError('Implement proxy reward methods.')

    def train(self):
        raise NotImplementedError('Implement train methods.')
