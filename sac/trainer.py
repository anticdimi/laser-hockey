import numpy as np
import time

from base.evaluator import evaluate
from sac_agent import SACAgent
from utils import utils
from laserhockey import hockey_env as h_env


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

    def train(self, agent, opponents, env, run_evaluation):
        rew_stats, q1_losses, q2_losses, actor_losses, alpha_losses = [], [], [], [], []

        lost_stats, touch_stats, won_stats = {}, {}, {}
        eval_stats = {
            'weak': {
                'reward': [],
                'touch': [],
                'won': [],
                'lost': []
            },
            'strong': {
                'reward': [],
                'touch': [],
                'won': [],
                'lost': []
            }
        }

        episode_counter = 1
        total_step_counter = 0
        grad_updates = 0
        new_op_grad = []
        while episode_counter <= self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            total_reward, touched = 0, 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            opponent = utils.poll_opponent(opponents)

            first_time_touch = 1
            for step in range(self._config['max_steps']):
                a1 = agent.act(ob)

                if self._config['mode'] == 'defense':
                    a2 = opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = np.zeros_like(a1)
                else:
                    a2 = opponent.act(obs_agent2)

                actions = np.hstack([a1, a2])
                next_state, reward, done, _info = env.step(actions)

                touched = max(touched, _info['reward_touch_puck'])

                step_reward = (
                    reward
                    + 5 * _info['reward_closeness_to_puck']
                    - (1 - touched) * 0.1
                    + touched * first_time_touch * 0.1 * step
                )
                first_time_touch = 1 - touched

                total_reward += step_reward

                agent.store_transition((ob, a1, step_reward, next_state, done))

                if self._config['show']:
                    time.sleep(0.01)
                    env.render()

                if touched > 0:
                    touch_stats[episode_counter] = 1

                if done:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                    break

                ob = next_state
                obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

            if agent.buffer.size < self._config['batch_size']:
                continue

            for _ in range(self._config['grad_steps']):
                losses = agent.update_parameters(total_step_counter)
                grad_updates += 1

                q1_losses.append(losses[0])
                q2_losses.append(losses[1])
                actor_losses.append(losses[2])
                alpha_losses.append(losses[3])

                # Add trained agent to opponents queue
                if self._config['selfplay']:
                    if (
                        grad_updates % self._config['add_self_every'] == 0
                    ):
                        new_opponent = SACAgent.clone_from(agent)
                        new_opponent.eval()
                        opponents.append(new_opponent)
                        new_op_grad.append(grad_updates)

            agent.schedulers_step()
            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward)

            if episode_counter % self._config['evaluate_every'] == 0:
                agent.eval()
                for eval_op in ['strong', 'weak']:
                    ev_opponent = opponents[0] if eval_op == 'strong' else h_env.BasicOpponent(False)
                    rew, touch, won, lost = evaluate(
                        agent,
                        env,
                        ev_opponent,
                        100,
                        quiet=True
                    )
                    eval_stats[eval_op]['reward'].append(rew)
                    eval_stats[eval_op]['touch'].append(touch)
                    eval_stats[eval_op]['won'].append(won)
                    eval_stats[eval_op]['lost'].append(lost)
                agent.train()

                self.logger.save_model(agent, f'a-{episode_counter}.pkl')

            rew_stats.append(total_reward)

            episode_counter += 1

        if self._config['show']:
            env.close()

        # Print train stats
        self.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)

        self.logger.info('Saving training statistics...')

        # Plot reward
        self.logger.plot_running_mean(data=rew_stats, title='Total reward', filename='total-reward.pdf', show=False)

        # Plot evaluation stats
        self.logger.plot_evaluation_stats(eval_stats, self._config['evaluate_every'], 'evaluation-won-lost.pdf')

        # Plot losses
        for loss, title in zip([q1_losses, q2_losses, actor_losses, alpha_losses],
                               ['Q1 loss', 'Q2 loss', 'Policy loss', 'Alpha loss']):
            self.logger.plot_running_mean(
                data=loss,
                title=title,
                filename=f'{title.replace(" ", "-")}.pdf',
                show=False,
                v_milestones=new_op_grad,
            )

        # Save agent
        self.logger.save_model(agent, 'agent.pkl')

        if run_evaluation:
            agent.eval()
            agent._config['show'] = True
            evaluate(agent, env, h_env.BasicOpponent(weak=False), self._config['eval_episodes'])
