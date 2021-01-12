import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
import numpy as np

from base.agent import Agent
from models import *
from base import proxy_rewards


class SACAgent(Agent):
    def __init__(self, opponent, logger, obs_dim, action_dim, max_action, userconfig):
        super().__init__(
            opponent=opponent,
            logger=logger,
            obs_dim=obs_dim,
            action_dim=action_dim,
            userconfig=userconfig
        )
        self.device = userconfig['device']

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
        self.actor = ActorNetwork(
            input_dims=obs_dim,
            max_action=max_action,
            n_actions=action_dim,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            device=self._config['device']
        )

        self.critic_1 = CriticNetwork(
            input_dims=obs_dim,
            n_actions=action_dim,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            device=self._config['device']
        )

        self.critic_2 = CriticNetwork(
            input_dims=obs_dim,
            n_actions=action_dim,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            device=self._config['device']
        )

        self.value = ValueNetwork(
            input_dims=obs_dim,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            device=self._config['device']
        )

        self.target_value = ValueNetwork(
            input_dims=obs_dim,
            learning_rate=self._config['learning_rate'],
            hidden_sizes=[256, 256],
            device=self._config['device']
        )

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

    def act(self, obs):
        state = torch.FloatTensor(obs).to(self.actor.device)
        acts, _ = self.actor.sample(state, reparam=False)

        return acts.cpu().detach().numpy()

    def train(self):
        if self.buffer.size < self._config['batch_size']:
            return

        data = self.buffer.sample(self._config['batch_size'])
        s = np.stack(data[:, 0])  # s_t
        a = np.stack(data[:, 1])[:, None]  # a_t
        rew = np.stack(data[:, 2])[:, None]  # r
        s_next = np.stack(data[:, 3])  # s_t+1
        done = (np.stack(data[:, 4])[:, None]).astype(np.int)

        state = torch.FloatTensor(s).to(self.device)
        next_state = torch.FloatTensor(s_next).to(self.device)
        action = torch.FloatTensor(a).squeeze(dim=1).to(self.device)
        reward = torch.FloatTensor(rew).squeeze(dim=1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).squeeze(dim=1).to(self.device)

        value = self.value(state).view(-1)
        target_value = self.target_value(next_state).view(-1)
        # target_value[done] = 0.0

        critic_val, log_probs = self._ask_critic(state, False)
        self.value.optimizer.zero_grad()
        value_target = critic_val - log_probs
        value_loss = 0.5 * self.value.loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        critic_val, log_probs = self._ask_critic(state, True)

        actor_loss = log_probs - critic_val
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = reward + self._config['gamma'] * target_value
        q1_old = self.critic_1.forward(state, action).view(-1)
        q2_old = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * self.critic_1.loss(q1_old, q_hat)
        critic_2_loss = 0.5 * self.critic_2.loss(q2_old, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self._update_target_net()

    def _update_target_net(self):
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._config['soft_tau']) + param.data * self._config['soft_tau']
            )

    def _ask_critic(self, state, reparam):
        actions, log_probs = self.actor.sample(state, reparam)
        log_probs = log_probs.view(-1)

        q1_new = self.critic_1.forward(state, actions)
        q2_new = self.critic_2.forward(state, actions)
        critic_val = torch.min(q1_new, q2_new).view(-1)

        return critic_val, log_probs
