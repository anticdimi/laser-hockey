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
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_dim: int
        The variable specifies the dimension of action space vector.
    action_mapping: Iterable
        The variable specifies a custom action space
    **userconfig:
        The variable specifies the config settings.
    """

    def __init__(self, logger, obs_dim, action_mapping, userconfig):
        super().__init__(
            logger=logger,
            obs_dim=obs_dim,
            action_dim=len(action_mapping),
            userconfig=userconfig
        )
        self.id = 1
        self.action_mapping = action_mapping

        lr_milestones = []
        if self._config['lr_milestones'] is not None:
            lr_milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]
        else:
            lr_milestones = np.arange(start=0,
                                      stop=self._config['max_episodes'] + 1,
                                      step=self._config['change_lr_every'])[1:]

        self.Q = QFunction(
            obs_dim=obs_dim,
            action_dim=len(action_mapping),
            hidden_sizes=self._config['hidden_sizes'],
            learning_rate=self._config['learning_rate'],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device'],
            dueling=self._config['dueling']
        )

        self.target_Q = copy.deepcopy(self.Q)

    def train(self):
        self.Q.train()
        self.target_Q.train()

    def eval(self):
        self.Q.eval()
        self.target_Q.eval()

    def update_target_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._config['epsilon']
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = np.random.randint(0, len(self.action_mapping))
        return action

    def train_model(self):
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
            priorities = np.abs(targets - pred) + 1e-6
            self.buffer.update_priorities(indices=indices, priorities=priorities.flatten())

        return fit_loss

    def update_per_beta(self, beta):
        self.buffer.update_beta(beta=beta)

    def step_lr_scheduler(self):
        self.Q.lr_scheduler.step()

    def __str__(self):
        return f"DQN {self.id}"
