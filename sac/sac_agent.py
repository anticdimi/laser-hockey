import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
import numpy as np

from base.agent import Agent
from models import *
from base import proxy_rewards
from utils.utils import soft_update
import time


class SACAgent(Agent):
    """
    The SACAgent class implements a trainable Soft Actor Critic agent, as described in: https://arxiv.org/pdf/1812.05905.pdf.

    Parameters
    ----------
    opponent: object
        The variable the agent that is used as an opponent during training/evaluation.
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    obs_dim: int
        The variable specifies the dimension of observation space vector.
    action_space: ndarray
        The variable specifies the action space of environment.
    userconfig:
        The variable specifies the config settings.
    """

    def __init__(self, opponent, logger, obs_dim, action_space, userconfig):
        super().__init__(
            opponent=opponent,
            logger=logger,
            obs_dim=obs_dim,
            action_dim=action_space.shape[0],
            userconfig=userconfig
        )
        self.device = userconfig['device']
        self.alpha = userconfig['alpha']
        self.automatic_entropy_tuning = self._config['automatic_entropy_tuning']
        self.eval_mode = False

        # Scaling factors for rewards
        self._factors = {
            'shooting': {
                'factor_closeness': 500,
                'factor_outcome': 10,
                'factor_existence': 1,
                'factor_neutral_result': -0.02,
            },
            'defense': {
                'closeness': 130,
                'outcome': 30,
                'existence': 1,
                'factor_touch': 200,
            },
            'normal': {},
        }

        if self._config['lr_milestones'] is None:
            raise ValueError('lr_milestones argument cannot be None!\nExample: --lr_milestones=100 200 300')

        lr_milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]

        # TODO: Should different lr's be passed to different nets?

        self.actor = ActorNetwork(
            input_dims=obs_dim,
            learning_rate=self._config['learning_rate'],
            action_space=action_space,
            hidden_sizes=[128],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device']
        )

        self.critic = CriticNetwork(
            input_dim=obs_dim,
            n_actions=action_space.shape[0],
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[128],
            lr_milestones=lr_milestones,
            lr_factor=self._config['lr_factor'],
            device=self._config['device']
        )

        self.critic_target = CriticNetwork(
            input_dim=obs_dim,
            n_actions=action_space.shape[0],
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[128],
            device=self._config['device']
        )

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.FloatTensor(action_space.shape[0]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self._config['learning_rate'])

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

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def act(self, obs):
        return self._act(obs, True) if self.eval_mode else self._act(obs)

    def _act(self, obs, evaluate=False):
        state = torch.FloatTensor(obs).to(self.actor.device)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()

    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.actor.lr_scheduler.step()

    def update_parameters(self, total_step):
        if self.buffer.size < self._config['batch_size']:
            return

        data = self.buffer.sample(self._config['batch_size'])

        state = torch.FloatTensor(
            np.stack(data[:, 0]),
            device=self.device
        )

        next_state = torch.FloatTensor(
            np.stack(data[:, 3]),
            device=self.device
        )

        action = torch.FloatTensor(
            np.stack(data[:, 1])[:, None],
            device=self.device
        ).squeeze(dim=1)

        reward = torch.FloatTensor(
            np.stack(data[:, 2])[:, None],
            device=self.device
        ).squeeze(dim=1)

        not_done = torch.FloatTensor(
            (~np.stack(data[:, 4])[:, None]).astype(np.int),
            device=self.device
        ).squeeze(dim=1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            q1_new, q2_new = self.critic(next_state, next_state_action)

            min_qf_next_target = torch.min(q1_new, q2_new) - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self._config['gamma'] * (min_qf_next_target).squeeze()

        qf1, qf2 = self.critic(state, action)

        qf1_loss = self.critic.loss(qf1.squeeze(), next_q_value)
        qf2_loss = self.critic.loss(qf2.squeeze(), next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        qf_loss.backward()
        self.critic.optimizer.step()

        pi, log_pi, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
        if total_step % self._config['update_target_every'] == 0:
            soft_update(self.critic_target, self.critic, self._config['soft_tau'])

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()
