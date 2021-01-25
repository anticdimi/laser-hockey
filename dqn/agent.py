import sys
sys.path.insert(0, '..')

import copy
import numpy as np
from qfeedforward import QFunction
import time
from base.agent import Agent
from base import proxy_rewards


class DQNAgent(Agent):
    """
    The DQNAgent class implements a trainable DQN agent.

    Parameters
    ----------
    opponent: object
        The variable the agent that is used as an opponent during training/evaluation.
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_dim: int
        The variable specifies the dimension of action space vector.
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    CUSTOM_DISCRETE_ACTIONS: Iterable
        The variable specifies a custom action space
    **userconfig:
        The variable specifies the config settings.
    """

    def __init__(self, opponent, logger, obs_dim, action_dim, CUSTOM_DISCRETE_ACTIONS, userconfig):
        super().__init__(
            opponent=opponent,
            logger=logger,
            obs_dim=obs_dim,
            action_dim=action_dim,
            userconfig=userconfig
        )
        # Scaling factors for rewards
        self._factors = {
            'shooting': {
                'factor_closeness': 500,
                'factor_outcome': 10,
                'factor_existence': 1,
                'factor_neutral_result': -0.02,
            },
            'defense': {
                'closeness': 260,
                'outcome': 30,
                'existence': 1,
                'factor_touch': 200,
            },
            'normal': {},
        }

        self.CUSTOM_DISCRETE_ACTIONS = CUSTOM_DISCRETE_ACTIONS

        milestones = []
        if self._config['lr_milestones'] is not None:
            milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]
        else:
            milestones = np.arange(start=0,
                                   stop=self._config['max_episodes'] + 1,
                                   step=self._config['change_lr_every'])[1:]

        self.Q = QFunction(
            obs_dim,
            len(self.CUSTOM_DISCRETE_ACTIONS),
            hidden_sizes=self._config['hidden_sizes'],
            learning_rate=self._config['learning_rate'],
            lr_milestones=milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device'],
            dueling=self._config['dueling']
        )

        self.target_Q = copy.deepcopy(self.Q)

    def update_target_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._config['epsilon']
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = np.random.randint(0, len(self.CUSTOM_DISCRETE_ACTIONS))
        return action

    def _shooting_reward(
        self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched=0
    ):
        return proxy_rewards.shooting_proxy(self, env, reward_game_outcome, reward_closeness_to_puck,
                                            reward_touch_puck, reward_puck_direction, touched)

    def _defense_reward(
        self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched
    ):
        return proxy_rewards.defense_proxy(self, env, reward_game_outcome, reward_closeness_to_puck,
                                           reward_touch_puck, reward_puck_direction, touched)

    def train(self):
        losses = []
        for i in range(self._config['iter_fit']):
            data = self.buffer.sample(batch_size=self._config['batch_size'])
            s = np.stack(data[:, 0])  # s_t
            a = np.stack(data[:, 1])[:, None]  # a_t
            rew = np.stack(data[:, 2])[:, None]  # r
            s_next = np.stack(data[:, 3])  # s_t+1
            not_done = (~np.stack(data[:, 4])[:, None]).astype(np.int)  # not_done flag

            if self._config['double']:
                greedy_actions = self.Q.greedyAction(s_next)[:, None]
                value_s_next = self.target_Q.Q_value(s_next, greedy_actions).detach().numpy()
            else:
                value_s_next = self.target_Q.maxQ(s_next)[:, None]

            targets = rew + self._config['discount'] * np.multiply(not_done, value_s_next)

            if self._config['per']:
                weights = np.stack(data[:, 5])[:, None]
                indices = np.stack(data[:, 6])
            else:
                weights = np.ones(targets.shape)

            # optimize
            fit_loss, pred = self.Q.fit(s, a, targets, weights)

            if self._config['per']:
                # TODO parametrize per_epsilon
                priorities = np.abs(targets - pred) + 1e-6
                self.buffer.update_priorities(indices=indices, priorities=priorities.flatten())

            losses.append(fit_loss)

        self.Q.lr_scheduler.step()

        return losses

    def update_per_beta(self, beta):
        self.buffer.update_beta(beta=beta)
