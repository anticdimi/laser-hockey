import sys
import numpy as np
from base.agent import Agent
import torch
from ddpg.models import Actor, Critic
from base.experience_replay import UniformExperienceReplay
from utils.utils import soft_update

sys.path.insert(0, '..')


class DDPGAgent(Agent):
    """
        The DDPGAgent class implements a trainable DDPG agent.

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

    def __init__(self,  logger, obs_dim, action_space, userconfig):
        super().__init__(
            logger=logger,
            obs_dim=obs_dim,
            action_dim=action_space.shape[0],
            userconfig=userconfig
        )

        self._observation_dim = obs_dim
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.05,
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate_actor": 0.0002,
            "learning_rate_critic": 0.0002,
            "hidden_sizes": [256, 256,256],
            'tau': 0.0001
        }

        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._tau = self._config['tau']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = UniformExperienceReplay(max_size=self._config["buffer_size"])
        self.eval_mode = False

        if self._config['lr_milestones'] is None:
            raise ValueError('lr_milestones argument cannot be None!\nExample: --lr_milestones=100 200 300')

        # lr_milestones = [int(x) for x in (self._config['lr_milestones'][0]).split(' ')]
        lr_milestones = self._config['lr_milestones']
        # Critic
        self.critic = Critic(self._observation_dim, self._action_n,
                             hidden_sizes=self._config['hidden_sizes'],
                             learning_rate=self._config['learning_rate_critic'],
                             lr_milestones=lr_milestones,
                             lr_factor=self._config['lr_factor'],
                             device=self._config['device'])
        self.critic_target = Critic(self._observation_dim, self._action_n,
                                    hidden_sizes=self._config['hidden_sizes'],
                                    learning_rate=self._config['learning_rate_critic'],
                                    lr_milestones=lr_milestones,
                                    lr_factor=self._config['lr_factor'],
                                    device=self._config['device'])

        # Actor
        self.actor = Actor(self._observation_dim, self._action_n,
                           hidden_sizes=self._config['hidden_sizes'],
                           learning_rate=self._config['learning_rate_actor'],
                           lr_milestones=lr_milestones,
                           lr_factor=self._config['lr_factor'],
                           device=self._config['device'])
        self.actor_target = Actor(self._observation_dim, self._action_n,
                                  hidden_sizes=self._config['hidden_sizes'],
                                  learning_rate=self._config['learning_rate_actor'],
                                  lr_milestones=lr_milestones,
                                  lr_factor=self._config['lr_factor'],
                                  device=self._config['device'])

    def eval(self):
        self.eval_mode = True

    def train_mode(self):
        self.eval_mode = False

    def act(self, observation, eps=0, evaluation=False):
        state = torch.from_numpy(observation).float().to(self.device)
        if eps is None:
            eps = self._eps

        if np.random.random() > eps and not evaluation:

            action = self.actor.forward(state)
            action = action.detach().cpu().numpy()[0]
        else:
            action = self._action_space.sample()[:4]

        return action

    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.critic_target.lr_scheduler.step()
        self.actor.lr_scheduler.step()
        self.actor_target.lr_scheduler.step()

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self, total_step_counter, iter_fit=32):
        losses = []

        for i in range(iter_fit):
            data = self.buffer.sample(batch_size=self._config['batch_size'])
            s = torch.FloatTensor(
                np.stack(data[:, 0])
            ).to(self.device)

            s_next = torch.FloatTensor(
                np.stack(data[:, 3])
            ).to(self.device)
            a = torch.FloatTensor(
                np.stack(data[:, 1])[:, None]
            ).squeeze(dim=1).to(self.device)

            rew = torch.FloatTensor(
                np.stack(data[:, 2])[:, None]
            ).squeeze(dim=1).to(self.device)

            done = torch.FloatTensor(
                np.stack(data[:, 4])[:, None]
            ).squeeze(dim=1).to(self.device)  # done flag

            Q_target = self.critic(s, a).squeeze(dim=1).to(self.device)
            a_next = self.actor_target.forward(s_next)
            Q_next = self.critic_target.forward(s_next, a_next).squeeze(dim=1).to(self.device)
            # target
            targets = rew + self._config['discount'] * Q_next * (1.0 - done)

            # optimize critic
            targets = targets.to(self.device)

            critic_loss = self.critic.loss(Q_target.float(), targets.float())
            losses.append(critic_loss)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # optimize actor
            actions = self.actor.forward(s)
            actor_loss = - self.critic.forward(s, actions).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # update
            if (total_step_counter * iter_fit + i + 1) % self._config['update_target_every'] == 0:
                soft_update(self.critic_target, self.critic, self._tau)
                soft_update(self.actor_target, self.actor, self._tau)

        return losses
