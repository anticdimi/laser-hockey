import memory as mem
import copy
import numpy as np
from feedforward import QFunction
import time


class DQNAgent(object):
    """
    The DQNAgent class implements a trainable agent.

    Parameters
    ----------
    opponent: object
        The variable the agent that is used as an opponent during training/evaluation.
    obs_space: object
        The variable specifies the observation space of the environment.
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    CUSTOM_DISCRETE_ACTIONS: list
        The variable specifies a custom action space
    **userconfig:
        The variable specifies the config settings.
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
                'closeness': 130,
                'outcome': 30,
                'existence': 1,
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
        }
        self._config.update(userconfig)

        self.buffer = mem.Memory(max_size=self._config['buffer_size'])

        milestones = np.arange(start=0,
                               stop=self._config['max_episodes'] + 1,
                               step=self._config['change_lr_every'])[1:]
        self.Q = QFunction(
            self._observation_space.shape[0],
            len(self.CUSTOM_DISCRETE_ACTIONS),
            hidden_sizes=self._config['hidden_sizes'],
            learning_rate=self._config['learning_rate'],
            lr_milestones=milestones,
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
        constants = self._factors[self._config['mode']]

        reward_dict = {}

        if (1 <= env.player1.position[0] <= 2.5) and (2 <= env.player1.position[1] <= 6):
            reward_dict['existence-reward'] = constants['existence']
        else:
            reward_dict['existence-reward'] = (-1) * constants['existence']

        if reward_puck_direction < 0:
            reward_dict['closeness-reward'] = constants['closeness'] * reward_closeness_to_puck

        if env.done:
            if env.winner == -1:
                reward_dict['outcome-reward'] = (-1) * constants['outcome'] + 5
            elif env.winner == 0:
                reward_dict['outcome-reward'] = constants['outcome']

        return reward_dict

    def train(self):
        losses = []
        for i in range(self._config['iter_fit']):
            data = self.buffer.sample(batch=self._config['batch_size'])
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

            # optimize
            fit_loss = self.Q.fit(s, a, targets)
            losses.append(fit_loss)

        self.Q.lr_scheduler.step()

        return losses
