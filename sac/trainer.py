from collections import defaultdict
import numpy as np
import time
from base.evaluator import evaluate


class SACTrainer:
    """
    The SACTrainer class implements a trainer for the SACAgent.

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

    def train(self, agent, env, run_evaluation):
        rew_stats, q1_losses, q2_losses, actor_losses, alpha_losses = [], [], [], [], []

        lost_stats, touch_stats, won_stats = {}, {}, {}
        rewards = defaultdict(lambda: [])
        eval_stats = {
            'reward': [],
            'touch': [],
            'won': [],
            'lost': []
        }

        episode_counter = 0
        total_step_counter = 0
        while episode_counter < self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and self._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and self._config['mode'] == 'shooting'
            ):
                continue

            total_reward, touched = 0, 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            for step in range(self._config['max_steps']):
                a1 = agent.act(ob)

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

                losses = agent.update_parameters(total_step_counter)
                if losses is not None:
                    q1_losses.append(losses[0])
                    q2_losses.append(losses[1])
                    actor_losses.append(losses[2])
                    alpha_losses.append(losses[3])

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
                total_step_counter += 1

            if episode_counter % self._config['evaluate_every'] == 0:
                rew, touch, won, lost = evaluate(agent, env, 50, quiet=True)

                eval_stats['reward'].append(rew)
                eval_stats['touch'].append(touch)
                eval_stats['won'].append(won)
                eval_stats['lost'].append(lost)
                self.logger.save_model(agent, f'a-{episode_counter}.pkl')

            rew_stats.append(total_reward)

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward)

            episode_counter += 1

        if self._config['show']:
            env.close()

        # Print train stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        self.logger.info('Saving training statistics...')

        # Plot reward
        self.logger.plot_running_mean(rew_stats, 'Total reward', 'total-reward.pdf', show=False)

        # Plot evaluation stats
        self.logger.plot_intermediate_stats(eval_stats, show=False)

        # Plot losses
        for loss, title in zip([q1_losses, q2_losses, actor_losses, alpha_losses],
                               ['Q1 loss', 'Q2 loss', 'Policy loss', 'Alpha loss']):
            self.logger.plot_running_mean(loss, title, f'{title.replace(" ", "-")}.pdf', show=False)

        # Save agent
        self.logger.save_model(agent, 'agent.pkl')
        # Log rew histograms
        self.logger.clean_rew_dir()
        for reward_type, reward_values in rewards.items():
            self.logger.hist(reward_values, reward_type, f'{reward_type}.pdf', False)

        if run_evaluation:
            agent._config['show'] = True
            evaluate(agent, env, self._config['eval_episodes'])
