import os
import torch
from laserhockey import hockey_env as h_env
from agent import DQNAgent
from argparse import ArgumentParser
import sys
from custom_action_space import DEFAULT_DISCRETE_ACTIONS, REDUCED_CUSTOM_DISCRETE_ACTIONS
from trainer import DQNTrainer

from copy import deepcopy
from laserhockey.hockey_env import CENTER_X, CENTER_Y, SCALE

import time
import random

# TODO: fix if possible, not the best way of importing
sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils.utils import *


def dist_positions(p1, p2):
    return np.sqrt(np.sum(np.asarray(p1 - p2) ** 2, axis=-1))

def compute_reward_closeness_to_puck(transition):
    observation = np.asarray(transition[2])
    reward_closeness_to_puck = 0
    if (observation[-6] + CENTER_X) < CENTER_X and observation[-4] <= 0:
        dist_to_puck = dist_positions(observation[:2], observation[-6:-4])
        max_dist = 250. / SCALE
        max_reward = -30.  # max (negative) reward through this proxy
        factor = max_reward / (max_dist * 250 / 2)
        reward_closeness_to_puck += dist_to_puck * factor  # Proxy reward for being close to puck in the own half

    return reward_closeness_to_puck

def compute_winning_reward(transition, is_player_one):
    r = 0

    if transition[4]:
        if transition[5]['winner'] == 0:  # tie
            r += 0
        elif transition[5]['winner'] == 1 and is_player_one:  # you won
            r += 10
        elif transition[5]['winner'] == -1 and not is_player_one:
            r += 10
        else:  # opponent won
            r -= 10
    return r

def recompute_rewards(match, username):
    transitions = match['transitions']
    is_player_one = match['player_one'] == username
    new_transitions = []
    for transition in transitions:
        new_transition = list(deepcopy(transition))
        new_transition[3] = compute_winning_reward(transition, is_player_one) + compute_reward_closeness_to_puck(transition)
        new_transition[5]['reward_closeness_to_puck']
        new_transitions.append(tuple(new_transition))

    return new_transitions

parser = ArgumentParser()
parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true')
parser.add_argument('--cuda', help='Set if want to train on graphic card', action='store_true')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='normal')

# Training params
# TODO: ------------------------------------- ARGUMENTS TO LOOK OUT FOR -------------------------------------
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=10_000)
parser.add_argument('--per_beta_inc', help='Beta increment for PER', type=float, default=0.00008)
parser.add_argument('--self_play', help='Utilize self play', action='store_true')
parser.add_argument('--start_self_play_from', help='# of episode to start self play from', type=int, default=1)
parser.add_argument('--epsilon', help='Epsilon', type=float, default=0.1)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.00005) 
parser.add_argument('--add_opponent_every', help='# of grad updates until copying ourself', type=int, default=50_000)
# TODO: -----------------------------------------------------------------------------------------------------

parser.add_argument('--max_steps', help='Max steps for training', type=int, default=250)
parser.add_argument('--start_learning_from', help='# of steps from which on learning happens', type=int, default=1)
parser.add_argument('--train_every', help='Train every # of steps', type=int, default=4)
parser.add_argument('--update_target_every', help='# of gradient updates to updating target', type=int, default=1_000)
parser.add_argument('--change_lr_every', help='Change learning rate every # of episodes', type=int, default=1_000)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true')
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=1_000)
parser.add_argument('--evaluate_every', help='Evaluate every # of episodes', type=int, default=2_000)
parser.add_argument('--discount', help='Discount', type=float, default=0.99)
parser.add_argument('--epsilon_decay', help='Epsilon decay', type=float, default=0.0005)
parser.add_argument('--min_epsilon', help='min_epsilon', type=float, default=0.1)
parser.add_argument('--dueling', help='Specifies whether the architecture should be dueling', action='store_true')
parser.add_argument('--double', help='Calculate target with Double DQN', action='store_true')
parser.add_argument('--per', help='Utilize Prioritized Experience Replay (PER)', action='store_true')
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)
parser.add_argument('--per_beta', help='Beta for PER', type=float, default=0.4)
parser.add_argument('--per_beta_max', help='Max beta for PER', type=float, default=1)
parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
parser.add_argument('--buffer_size', help='Buffer capacity for the experience replay', type=int, default=int(1e6))

opts = parser.parse_args()

if __name__ == '__main__':
    if opts.dry_run:
        opts.max_episodes = 10

    if opts.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif opts.mode == 'shooting':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    opts.device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')

    dirname = time.strftime(f'%y%m%d_%H%M%S_{random.randint(0, 1e6):06}', time.gmtime(time.time()))
    abs_path = os.path.dirname(os.path.realpath(__file__))
    logger = Logger(prefix_path=os.path.join(abs_path, dirname),
                    mode=opts.mode,
                    cleanup=True,
                    quiet=opts.q)

    env = h_env.HockeyEnv(mode=mode, verbose=(not opts.q))

    action_mapping = REDUCED_CUSTOM_DISCRETE_ACTIONS

    q_agent = DQNAgent(
        logger=logger,
        obs_dim=env.observation_space.shape[0],
        action_mapping=action_mapping,
        userconfig=vars(opts)
    )

    with open(f"210316_051533_670487/agents/agent.pkl", 'rb') as inp:
        q_agent = pickle.load(inp)

    q_agent.train()
    q_agent._config.update(vars(opts))
    
    cnt = 0

    for root in [
        '/tmp/ALRL2020/client/user0/games/2021/3/15',
        '/tmp/ALRL2020/client/user0/games/2021/3/16'
    ]:
        for file in os.listdir(root):
            if file.endswith(".npz"):
                fpath = os.path.join(root, file)

                with np.load(fpath, allow_pickle=True) as d:
                    np_data = d['arr_0'].item()

                    if (
                            np_data['player_one'] == 'Zafir_Stojanovski_-_Dueling_DQN_ЈУГО'
                            and np_data['player_two'] != 'StrongBasicOpponent'
                            and np_data['player_two'] != 'WeakBasicOpponent'
                            # and np_data['player_two'] != 'Dimitrije_Antic_-_SAC_ЈУГО'
                    ):
                        transitions = recompute_rewards(np_data, 'Zafir_Stojanovski_-_Dueling_DQN_ЈУГО')
                        for t in transitions:
                            try:
                                tr = (
                                    t[0],
                                    REDUCED_CUSTOM_DISCRETE_ACTIONS.index(t[1]),
                                    float(t[3]),
                                    t[2],
                                    bool(t[4]),
                                )
                                q_agent.store_transition(tr)
                                cnt += 1
                            except Exception as e:
                                pass
    print(f'{cnt} historical transitions added')

    trainer = DQNTrainer(logger=logger, config=vars(opts))
    print(q_agent.Q)
    trainer.train(agent=q_agent, env=env)
