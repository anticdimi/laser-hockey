from collections import defaultdict
import numpy as np
import time
from base.evaluator import evaluate


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

    def train(self, agent, env, run_evaluation, action_mapping):
        epsilon = self._config['epsilon']
        epsilon_decay = self._config['epsilon_decay']
        min_epsilon = self._config['min_epsilon']
        episode_counter = 1
        total_step_counter = 0

        rew_stats = []
        loss_stats = []
        lost_stats = {}
        touch_stats = {}
        won_stats = {}
        # rewards = defaultdict(lambda: [])

        eval_stats = {
            'reward': [],
            'touch': [],
            'won': [],
            'lost': []
        }

        while episode_counter <= self._config['max_episodes']:
            ob = env.reset()
            obs_agent2 = env.obs_agent_two()

            if (env.puck.position[0] < 5 and self._config['mode'] == 'defense') or (
                    env.puck.position[0] > 5 and self._config['mode'] == 'shooting'):
                # TODO: Remove upper line when teaching to reach towards the ball
                continue

            epsilon = max(epsilon_decay * epsilon, min_epsilon)
            if self._config['per']:
                agent.update_per_beta(beta=1 - epsilon)

            total_reward = 0
            touched = 0
            touch_stats[episode_counter] = 0
            won_stats[episode_counter] = 0
            lost_stats[episode_counter] = 0

            for step in range(1, self._config['max_steps'] + 1):

                if total_step_counter % self._config['update_target_every'] == 0:
                    agent.update_target_net()

                a1 = agent.act(ob, eps=epsilon)
                a1_discrete = action_mapping(a1)

                if self._config['mode'] in ['defense', 'normal']:
                    a2 = agent.opponent.act(obs_agent2)
                elif self._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {self._config["mode"]} not implemented.')

                (ob_new, reward, done, _info) = env.step(np.hstack([a1_discrete, a2]))
                touched = max(touched, _info['reward_touch_puck'])

                # reward_dict = agent.reward_function(
                #     env,
                #     reward_game_outcome=reward,
                #     reward_closeness_to_puck=_info['reward_closeness_to_puck'],
                #     reward_touch_puck=_info['reward_touch_puck'],
                #     reward_puck_direction=_info['reward_puck_direction'],
                #     touched=touched,
                #     step=step,
                #     max_allowed_steps=self._config['max_steps']
                # )

                # for reward_type, reward_value in reward_dict.items():
                #     rewards[reward_type].append(reward_value)

                total_reward += reward
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

            self.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon)

            if episode_counter % self._config['evaluate_every'] == 0:
                self.logger.info("Evaluating agent")
                agent.eval()
                old_show = agent._config['show']
                agent._config['show'] = False
                rew, touch, won, lost = evaluate(agent=agent, env=env, eval_episodes=self._config['eval_episodes'],
                                                 quiet=True, action_mapping=action_mapping)
                agent.train()
                agent._config['show'] = old_show

                eval_stats['reward'].append(rew)
                eval_stats['touch'].append(touch)
                eval_stats['won'].append(won)
                eval_stats['lost'].append(lost)
                self.logger.save_model(agent, f'a-{episode_counter}.pkl')

            loss_stats.extend(agent.train())
            rew_stats.append(total_reward)

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

        # # Log rew histograms
        # for reward_type, reward_values in rewards.items():
        #     self.logger.hist(reward_values, reward_type, f'{reward_type}.pdf', False)

        if run_evaluation:
            agent._config['show'] = False
            agent.eval()
            evaluate(agent=agent, env=env, eval_episodes=self._config['eval_episodes'], quiet=False,
                     action_mapping=action_mapping, evaluate_on_opposite_side=False)
