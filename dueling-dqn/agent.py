import memory as mem
import copy
import numpy as np
from feedforward_duel import QFunction
import time
from collections import defaultdict


class DQNAgent(object):
    """
    The DQNAgent class implements a trainable agent.

    Parameters
    ----------
    opponent : object
        The variable the agent that is used as an opponent during training/evaluation.
    obs_space: object
        The variable specifies the observation space of the environment.
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    CUSTOM_DISCRETE_ACTIONS: list
        This variable specifies a custom action space
    **userconfig:
        This variable specifies the config settings.
    """

    def __init__(self, opponent, obs_space, CUSTOM_DISCRETE_ACTIONS, logger, **userconfig):
        self.opponent = opponent
        self.logger = logger

        if userconfig['mode'] == 'normal':
            self.reward_function = None
            raise NotImplementedError('Mode normal not implemented')
        elif userconfig['mode'] == 'shooting':
            self.reward_function = self._shooting_reward
        elif userconfig['mode'] == 'defense':
            self.reward_function = self._defense_reward
        else:
            raise ValueError('Unknown training mode. See --help')

        # Scaling factors for rewards
        self._factors = {
            'shooting': {
                'factor_closeness': 500,
                'factor_outcome': 10,
                'factor_existence': 1,
                'factor_neutral_result': -0.02,
            },
            'defense': {
                'factor_closeness': 130,
                'factor_outcome': 10,
                'factor_existence': 2,
                'factor_touch': 200,
            },
            'normal': {},
        }

        self.CUSTOM_DISCRETE_ACTIONS = CUSTOM_DISCRETE_ACTIONS
        self._observation_space = obs_space
        self._config = {
            'discount': 0.95,
            'buffer_size': int(1e5),
            'batch_size': 128,
            'hidden_sizes': [128],
            'update_target': True,
        }
        self._config.update(userconfig)

        self.buffer = mem.Memory(max_size=self._config['buffer_size'])

        self.Q = QFunction(
            self._observation_space.shape[0],
            len(self.CUSTOM_DISCRETE_ACTIONS),
            hidden_sizes=self._config['hidden_sizes'],
            learning_rate=self._config['learning_rate'],
            device=self._config['device'],
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

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def evaluate(self, env, eval_episodes, action_mapping):
        rew_stats = []
        touch_stats = {}
        won_stats = {}
        lost_stats = {}
        for episode_counter in range(eval_episodes):
            total_reward = 0
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and self._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and self._config['mode'] == 'shooting'
            ):
                continue

            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            for step in range(self._config['max_steps']):
                a1 = self.act(ob, eps=0)
                # a1_discrete = env.discrete_to_continous_action(a1)
                a1_discrete = action_mapping(a1)

                if self._config['mode'] == 'defense':
                    a2 = self.opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {self._config["mode"]} not implemented.')

                (ob_new, reward, done, _info) = env.step(np.hstack([a1_discrete, a2]))

                if _info['reward_touch_puck'] > 0:
                    touch_stats[episode_counter] = 1

                total_reward += reward
                ob = ob_new
                obs_agent2 = env.obs_agent_two()
                if self._config['show']:
                    time.sleep(0.01)
                    env.render()
                if done:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break

            rew_stats.append(total_reward)

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon=0)

        # Print evaluation stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

    def _shooting_reward(
        self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched=0
    ):
        factors = self._factors[self._config['mode']]

        reward_dict = {}
        reward_dict['closeness-reward'] = (1 - touched) * factors['factor_closeness'] * reward_closeness_to_puck
        reward_dict['existence-reward'] = (-1) * factors['factor_existence']
        reward_dict['outcome-reward'] = factors['factor_outcome'] * reward_game_outcome

        return reward_dict

    def _defense_reward(
        self, env, reward_game_outcome, reward_closeness_to_puck, reward_touch_puck, reward_puck_direction, touched
    ):
        factors = self._factors[self._config['mode']]

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

    def train_in_env(self, env, evaluate, action_mapping):
        epsilon = self._config['epsilon']
        epsilon_decay = self._config['epsilon_decay']
        min_epsilon = self._config['min_epsilon']
        episode_counter = 0

        rew_stats = []
        loss_stats = []
        lost_stats = {}
        touch_stats = {}
        won_stats = {}
        rewards = defaultdict(lambda: [])

        while episode_counter < self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and self._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and self._config['mode'] == 'shooting'
            ):
                continue

            epsilon = max(epsilon_decay * epsilon, min_epsilon)

            total_reward = 0
            touched = 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            for step in range(self._config['max_steps']):
                a1 = self.act(ob, eps=epsilon)
                a1_discrete = action_mapping(a1)

                if self._config['mode'] == 'defense':
                    a2 = self.opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {self._config["mode"]} not implemented.')

                (ob_new, reward, done, _info) = env.step(np.hstack([a1_discrete, a2]))
                touched = max(touched, _info['reward_touch_puck'])

                reward_dict = self.reward_function(
                    env,
                    reward_game_outcome=reward,
                    reward_closeness_to_puck=_info['reward_closeness_to_puck'],
                    reward_touch_puck=_info['reward_touch_puck'],
                    reward_puck_direction=_info['reward_puck_direction'],
                    touched=touched,
                )

                for reward_type, reward_value in reward_dict.items():
                    rewards[reward_type].append(reward_value)

                summed_reward = sum(list(reward_dict.values()))
                total_reward += summed_reward
                self.store_transition((ob, a1, summed_reward, ob_new, done))

                if self._config['show']:
                    time.sleep(0.01)
                    env.render()

                if touched > 0:
                    touch_stats[episode_counter] = 1

                if done:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break

                ob = ob_new
                obs_agent2 = env.obs_agent_two()

            loss_stats.extend(self._train(episode_number=episode_counter, iter_fit=self._config['iter_fit']))
            rew_stats.append(total_reward)

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon)

            episode_counter += 1

        if self._config['show']:
            env.close()

        # Print train stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        # Plot reward
        self.logger.plot_running_mean(rew_stats, 'Total reward', 'total-reward.pdf', show=False)

        # Plot loss
        self.logger.plot_running_mean(loss_stats, 'Loss', 'loss.pdf', show=False)

        # Save model
        self.logger.save_model(self, 'agent.pkl')

        # Log rew histograms
        self.logger.clean_rew_dir()
        for reward_type, reward_values in rewards.items():
            self.logger.hist(reward_values, reward_type, f'{reward_type}.pdf', False)

        if evaluate:
            self._config['show'] = True
            self.evaluate(env, self._config['eval_episodes'], action_mapping)

    def _train(self, episode_number, iter_fit=20):
        if self._config['update_target']:
            self.update_target_net()

        if self._config['halve_lr'] and (episode_number % self._config['halve_lr_every'] == 0):
            self.Q.halve_learning_rate()

        losses = []
        for i in range(iter_fit):
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:, 0])  # s_t
            a = np.stack(data[:, 1])[:, None]  # a_t
            rew = np.stack(data[:, 2])[:, None]  # r
            s_next = np.stack(data[:, 3])  # s_t+1
            not_done = (~np.stack(data[:, 4])[:, None]).astype(np.int)  # not_done flag

            value_s_next = self.target_Q.maxQ(s_next)[:, None]
            targets = rew + self._config['discount'] * np.multiply(not_done, value_s_next)

            # optimize
            fit_loss = self.Q.fit(s, a, targets)
            losses.append(fit_loss)
        return losses
