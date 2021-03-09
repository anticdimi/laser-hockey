from collections import defaultdict
import numpy as np
import time
from base.evaluator import evaluate
from utils import utils
from laserhockey import hockey_env as h_env


class DDPGTrainer:
    """
    The DQNTrainer class implements a trainer for the DDPGAgent.

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

    def train(self, agent, opponents,  env, eval):
        epsilon = self._config['eps']
        epsilon_decay = self._config['epsilon_decay']
        min_epsilon = self._config['min_epsilon']
        iter_fit = self._config['iter_fit']
        episode_counter = 1
        total_step_counter = 0

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
        while episode_counter <= self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)
            total_reward = 0
            touched = 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0
            opponent = utils.poll_opponent(opponents)

            first_time_touch = 1
            for step in range(self._config['max_steps']):

                a1 = agent.act(ob, eps=epsilon)
                if self._config['mode'] == 'defense':
                    a2 = opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    a2 = opponent.act(obs_agent2)
                (ob_new, reward, done, _info) = env.step(np.hstack([a1, a2]))
                touched = max(touched, _info['reward_touch_puck'])

                total_reward += reward + 5 * _info['reward_closeness_to_puck'] - (
                        1 - touched) * 0.1 + touched * first_time_touch * 0.1 * step
                first_time_touch = 1 - touched
                agent.store_transition((ob, a1, reward, ob_new, done))

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

            loss_stats.extend(agent.train(iter_fit=iter_fit, total_step_counter=total_step_counter))
            rew_stats.append(total_reward)

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon)

            if episode_counter % self._config['evaluate_every'] == 0:
                agent.eval()
                rew, touch, won, lost = evaluate(agent, env, h_env.BasicOpponent(weak=False),
                                                 self._config['eval_episodes'], quiet=True)
                agent.train_mode()

                eval_stats['reward'].append(rew)
                eval_stats['touch'].append(touch)
                eval_stats['won'].append(won)
                eval_stats['lost'].append(lost)
                self.logger.save_model(agent, f'a-{episode_counter}.pk l')

            agent.schedulers_step()
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

        # Plot loss
        self.logger.plot_running_mean(loss_stats, 'Loss', 'loss.pdf', show=False)

        # Save model
        self.logger.save_model(agent, 'agent.pkl')

        # Log rew histograms

        if eval:
            agent.eval()
            agent._config['show'] = True
            evaluate(agent, env, h_env.BasicOpponent(weak=False), self._config['eval_episodes'])
            agent.train_mode()
