import os
import torch
import numpy as np
from laserhockey import hockey_env as h_env
from agent import DQNAgent
from importlib import reload
from argparse import ArgumentParser
import sys
from custom_action_space import custom_discrete_to_continuous_action, CUSTOM_DISCRETE_ACTIONS
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
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='defense')

# Training params
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=12000)
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=160)
parser.add_argument('--iter_fit', help='Iter fit', type=int, default=32)
parser.add_argument('--update_target_every', help='# of steps between updating target net', type=int, default=1000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.0001)
parser.add_argument('--change_lr_every', help='Change learning rate every # of episodes', type=int, default=1000)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=30)
parser.add_argument('--discount', help='Discount', type=float, default=0.95)
parser.add_argument('--epsilon', help='Epsilon', type=float, default=0.95)
parser.add_argument('--epsilon_decay', help='Epsilon decay', type=float, default=0.9987)
parser.add_argument('--min_epsilon', help='min_epsilon', type=float, default=0.05)
parser.add_argument('--dueling', help='Specifies whether the architecture should be dueling', action='store_true')
parser.add_argument('--double', help='Calculate target with Double DQN', action='store_true')
parser.add_argument('--per', help='Utilize Prioritized Experience Replay', action='store_true')
parser.add_argument('--per_alpha', help='Alpha for PER', type=float, default=0.6)

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

    opts.device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')
    logger = Logger(prefix_path=os.path.dirname(os.path.realpath(__file__)) + '/logs', mode=opts.mode, quiet=opts.q)
    opponent = h_env.BasicOpponent(weak=False)
    env = h_env.HockeyEnv(mode=mode, verbose=(not opts.q))

    q_agent = DQNAgent(
        opponent=opponent,
        logger=logger,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        CUSTOM_DISCRETE_ACTIONS=CUSTOM_DISCRETE_ACTIONS,
        userconfig=vars(opts)
    )
    trainer = DQNTrainer(logger, vars(opts))
    trainer.train(q_agent, env, opts.evaluate, custom_discrete_to_continuous_action)
