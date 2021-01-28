import time
import numpy as np


def evaluate(agent, env, eval_episodes, action_mapping=None, evaluate_on_opposite_side=False):
    rew_stats = []
    touch_stats = {}
    won_stats = {}
    lost_stats = {}
    for episode_counter in range(eval_episodes):
        total_reward = 0
        ob = env.reset()
        obs_agent2 = env.obs_agent_two()

        if (env.puck.position[0] < 5 and agent._config['mode'] == 'defense') or (
                env.puck.position[0] > 5 and agent._config['mode'] == 'shooting'
        ):
            continue

        touch_stats[episode_counter] = 0
        won_stats[episode_counter] = 0
        for step in range(agent._config['max_steps']):

            if evaluate_on_opposite_side:
                if action_mapping is not None:
                    # DQN act
                    a2 = agent.act(obs_agent2, eps=0)
                    a2 = action_mapping(a2)
                else:
                    a2 = agent.act(obs_agent2, True)

                if agent._config['mode'] == 'defense':
                    a1 = agent.opponent.act(ob)
                elif agent._config['mode'] == 'shooting':
                    a1 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {agent._config["mode"]} not implemented.')

            else:
                if action_mapping is not None:
                    # DQN act
                    a1 = agent.act(ob, eps=0)
                    a1 = action_mapping(a1)
                else:
                    # SAC act
                    a1 = agent.act(ob, True)

                if agent._config['mode'] == 'defense':
                    a2 = agent.opponent.act(obs_agent2)
                elif agent._config['mode'] == 'shooting':
                    a2 = [0, 0, 0, 0]
                else:
                    raise NotImplementedError(f'Training for {agent._config["mode"]} not implemented.')

            (ob_new, reward, done, _info) = env.step(np.hstack([a1, a2]))
            ob = ob_new
            obs_agent2 = env.obs_agent_two()

            if evaluate_on_opposite_side:
                # Not really a way to implement this, given the structure of the env...
                touch_stats[episode_counter] = 0
                total_reward -= reward

            else:
                if _info['reward_touch_puck'] > 0:
                    touch_stats[episode_counter] = 1

                total_reward += reward

            if agent._config['show']:
                time.sleep(0.01)
                env.render()
            if done:
                if evaluate_on_opposite_side:
                    won_stats[episode_counter] = 1 if env.winner == -1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == 1 else 0
                else:
                    won_stats[episode_counter] = 1 if env.winner == 1 else 0
                    lost_stats[episode_counter] = 1 if env.winner == -1 else 0
                break

        rew_stats.append(total_reward)

        agent.logger.print_episode_info(env.winner, episode_counter, step, total_reward, epsilon=0)

    # Print evaluation stats
    agent.logger.print_stats(rew_stats, touch_stats, won_stats, lost_stats)
