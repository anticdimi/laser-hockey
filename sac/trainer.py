from collections import defaultdict
import numpy as np
import time


class SACTrainer:
    def __init__(self, logger, config) -> None:
        self.logger = logger
        self._config = config

    def train(self, agent, env, evaluate):
        rew_stats = []
        loss_stats = []
        lost_stats = {}
        touch_stats = {}
        won_stats = {}
        rewards = defaultdict(lambda: [])

        episode_counter = 0
        while episode_counter < self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and self._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and self._config['mode'] == 'shooting'
            ):
                continue

            total_reward = 0
            touched = 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            for step in range(self._config['max_steps']):
                a1 = agent.act(ob).squeeze()

                if self._config['mode'] == 'defense':
                    a2 = agent.opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = np.zeros_like(a1)
                else:
                    raise NotImplementedError(f'Training for {self._config["mode"]} not implemented.')
                actions = np.hstack([a1, a2])
                ob_new, reward, done, _info = env.step(actions)
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

                agent.train()

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

            # loss_stats.extend(agent.train())
            rew_stats.append(total_reward)

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward)

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

        # if evaluate:
        #     agent._config['show'] = True
        #     agent.evaluate(env, self._config['eval_episodes'])
