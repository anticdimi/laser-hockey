import os
import torch
import numpy as np
from laserhockey import hockey_env as h_env
from agent import DQNAgent
from importlib import reload
from argparse import ArgumentParser
import sys
from custom_action_space import DEFAULT_DISCRETE_ACTIONS, REDUCED_CUSTOM_DISCRETE_ACTIONS
from trainer import DQNTrainer

# TODO: fix if possible, not the best way of importing
sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils.utils import *

parser = ArgumentParser()
parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true')
parser.add_argument('--cuda', help='Set if want to train on graphic card', action='store_true')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='normal')

# Training params
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=30_000)
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=250)
parser.add_argument('--start_learning_from', help='# of steps from which on learning happens', type=int, default=50_000)
parser.add_argument('--train_every', help='Train every # of steps', type=int, default=4)
parser.add_argument('--update_target_every', help='# of gradient updates to updating target', type=int, default=1_000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.0001)
parser.add_argument('--change_lr_every', help='Change learning rate every # of episodes', type=int, default=1_000)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true')
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=1_000)
parser.add_argument('--evaluate_every', help='Evaluate every # of episodes', type=int, default=2_000)
parser.add_argument('--discount', help='Discount', type=float, default=0.99)
parser.add_argument('--epsilon', help='Epsilon', type=float, default=1)
parser.add_argument('--epsilon_decay', help='Epsilon decay', type=float, default=0.0005)
parser.add_argument('--min_epsilon', help='min_epsilon', type=float, default=0.1)
parser.add_argument('--dueling', help='Specifies whether the architecture should be dueling', action='store_true')
parser.add_argument('--double', help='Calculate target with Double DQN', action='store_true')
parser.add_argument('--per', help='Utilize Prioritized Experience Replay (PER)', action='store_true')
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)
parser.add_argument('--per_beta', help='Beta for PER', type=float, default=0.4)
parser.add_argument('--per_beta_inc', help='Beta increment for PER', type=float, default=0.000035)
parser.add_argument('--per_beta_max', help='Max beta for PER', type=float, default=1)
parser.add_argument('--self_play', help='Utilize self play', action='store_true')
parser.add_argument('--poll_opponent_every', help='Number of episodes betwe1en opponent polls', type=int, default=3_000)
parser.add_argument('--start_polling_from', help='Episode from which on we start polling', type=int, default=3_000)

opts = parser.parse_args()

# TODO: ************ Change per_beta_increment when changing number of episodes ************

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

    opts.device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')
    logger = Logger(prefix_path=os.path.dirname(os.path.realpath(__file__)) + '/logs',
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
    trainer = DQNTrainer(logger=logger, config=vars(opts))
    print(q_agent.Q)
    trainer.train(agent=q_agent, env=env)
