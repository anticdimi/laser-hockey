from collections import defaultdict
import numpy as np
import time
from base.evaluator import evaluate
from laserhockey import hockey_env as h_env
from utils.utils import poll_opponent
from copy import deepcopy
from utils.utils import mu_norm, std_norm


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

    def train(self, agent, env):
        epsilon = self._config['epsilon']
        epsilon_decay = self._config['epsilon_decay']
        min_epsilon = self._config['min_epsilon']
        episode_counter = 1
        total_step_counter = 0
        total_grad_updates = 0

        beta = self._config['per_beta']
        beta_inc = self._config['per_beta_inc']
        beta_max = self._config['per_beta_max']

        rew_stats = []
        loss_stats = []
        lost_stats = {}
        touch_stats = {}
        won_stats = {}

        eval_stats = {
            'reward': [],
            'touch': [],
            'won': [],
            'lost': []
        }

        opponents = [h_env.BasicOpponent(weak=True)]

        opponent = poll_opponent(opponents=opponents)

        while episode_counter <= self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and self._config['mode'] == 'defense') or (
                    env.puck.position[0] > 5 and self._config['mode'] == 'shooting'):
                # TODO: Remove upper line when teaching to reach towards the ball
                continue

            epsilon = max(epsilon - epsilon_decay, min_epsilon)
            if self._config['per']:
                beta = min(beta_max, beta + beta_inc)
                agent.update_per_beta(beta=beta)

            total_reward = 0
            touched = 0
            first_time_touch = 1
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            for step in range(1, self._config['max_steps'] + 1):
                a1 = agent.act(ob, eps=epsilon)
                a1_list = agent.action_mapping[a1]

                if self._config['mode'] in ['defense', 'normal']:
                    a2 = opponent.act(obs_agent2)
                    # a copy of our agent has been chosen, transform the action id to a list
                    if not isinstance(a2, np.ndarray):
                        a2 = agent.action_mapping[a2]
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {self._config["mode"]} not implemented.')

                (ob_new, reward, done, _info) = env.step(np.hstack([a1_list, a2]))

                touched = max(touched, _info['reward_touch_puck'])

                step_reward = reward + 5 * _info['reward_closeness_to_puck'] - (1 - touched) * 0.1 + \
                              touched * first_time_touch * 0.1 * step

                first_time_touch = 1 - touched

                # step_reward = reward

                total_reward += step_reward

                agent.store_transition((ob, a1, step_reward, ob_new, done))

                if self._config['show']:
                    time.sleep(0.01)
                    env.render()

                if touched > 0:
                    touch_stats[episode_counter] = 1

                if done:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break

                if total_step_counter % self._config['train_every'] == 0 and \
                        total_step_counter > self._config['start_learning_from']:

                    loss_stats.append(agent.train_model())
                    rew_stats.append(total_reward)
                    total_grad_updates += 1

                    if total_grad_updates % self._config['update_target_every'] == 0:
                        agent.update_target_net()

                ob = ob_new
                obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon, touched)

            if episode_counter % self._config['evaluate_every'] == 0:
                self.logger.info("Evaluating agent")
                agent.eval()
                old_show = agent._config['show']
                agent._config['show'] = False
                rew, touch, won, lost = evaluate(agent=agent, env=env, opponent=h_env.BasicOpponent(weak=True),
                                                 eval_episodes=self._config['eval_episodes'], quiet=True,
                                                 action_mapping=agent.action_mapping)
                agent.train()
                agent._config['show'] = old_show

                eval_stats['reward'].append(rew)
                eval_stats['touch'].append(touch)
                eval_stats['won'].append(won)
                eval_stats['lost'].append(lost)
                self.logger.save_model(agent, f'a-{episode_counter}.pkl')

            # TODO: COME BACK TO THIS
            if total_step_counter > self._config['start_learning_from']:
                agent.step_lr_scheduler()

            # if self._config['self_play'] and episode_counter >= self._config['max_episodes'] // 2 and \
            #         episode_counter % self._config['poll_opponent_every'] == 0:
            if self._config['self_play'] and episode_counter >= self._config['start_polling_from'] and \
                    episode_counter % self._config['poll_opponent_every'] == 0:
                opponents.append(deepcopy(agent))
                opponent = poll_opponent(opponents=opponents)

            episode_counter += 1

        if self._config['show']:
            env.close()

        # Print train stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        self.logger.info('Saving statistics...')

        # Plot reward
        self.logger.plot_running_mean(rew_stats, 'Total reward', 'total-reward.pdf', show=False)

        # Plot loss
        self.logger.plot_running_mean(loss_stats, 'Loss', 'loss.pdf', show=False)

        # Plot evaluation stats
        self.logger.plot_intermediate_stats(eval_stats, show=False)

        # Save model
        self.logger.save_model(agent, 'agent.pkl')

        # Save arrays of won-lost stats
        self.logger.save_array(data=eval_stats["won"], filename="eval-won-stats")
        self.logger.save_array(data=eval_stats["lost"], filename="eval-lost-stats")
