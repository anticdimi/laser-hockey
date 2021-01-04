from collections import defaultdict
import numpy as np
import time


class DQNTrainer:
    """
    The DQNTrainer class implements a trainer for the DQNAgent.

    Parameters
    ----------
    logger: Logger
        The variable specifies a logger for model management, plotting and printing.
    config: dict
        The variable specifies config variables.
    """

    def __init__(self, logger, config) -> None:
        self.logger = logger
        self._config = config

    def train(self, agent, env, evaluate, action_mapping):
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
                a1 = agent.act(ob, eps=epsilon)
                a1_discrete = action_mapping(a1)

                if self._config['mode'] == 'defense':
                    a2 = agent.opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {self._config["mode"]} not implemented.')

                (ob_new, reward, done, _info) = env.step(np.hstack([a1_discrete, a2]))
                touched = max(touched, _info['reward_touch_puck'])

                reward_dict = agent.reward_function(
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
                agent.store_transition((ob, a1, summed_reward, ob_new, done))

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

            loss_stats.extend(agent._train(episode_number=episode_counter))
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
        self.logger.save_model(agent, 'agent.pkl')

        # Log rew histograms
        self.logger.clean_rew_dir()
        for reward_type, reward_values in rewards.items():
            self.logger.hist(reward_values, reward_type, f'{reward_type}.pdf', False)

        if evaluate:
            agent._config['show'] = True
            agent.evaluate(env, self._config['eval_episodes'], action_mapping)
