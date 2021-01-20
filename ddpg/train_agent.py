import os
import torch
from laserhockey import hockey_env as h_env
from ddpg.agent import DDPGAgent
from argparse import ArgumentParser
import sys
from ddpg.trainer import DDPGTrainer
from utils.utils import *


sys.path.insert(0, '.')
sys.path.insert(1, '..')


parser = ArgumentParser()
parser.add_argument('--dry-run', help='Set if running only for sanity check', action='store_true')
parser.add_argument('--cuda', help='Set if want to train on graphic card', action='store_true')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--evaluate', help='Set if want to evaluate agent after the training', action='store_true')
parser.add_argument('--mode', help='Mode for training currently: (shooting | defense | normal)', default='defense')

# Training params
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=5000)
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=160)
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=30)

parser.add_argument('--learning_rate_actor', help='Learning rate', type=float, default=3e-4)
parser.add_argument('--learning_rate_critic', help='Learning rate', type=float, default=3e-4)
parser.add_argument('--discount', help='Discount', type=float, default=0.95)
parser.add_argument('--eps', help='Epsilon', type=float, default=0.95)
parser.add_argument('--epsilon_decay', help='Epsilon decay', type=float, default=0.9987)
parser.add_argument('--min_epsilon', help='min_epsilon', type=float, default=0.07)
parser.add_argument('--iter_fit', help='fit every n iterations', type=float, default=30)
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--tau', help='tau', type=float, default=0.005)


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
    env = h_env.HockeyEnv(mode=mode, quiet=opts.q)

    agent = DDPGAgent(
        opponent=opponent,
        logger=logger,
        obs_dim=env.observation_space.shape,
        action_space=env.action_space,
        userconfig=vars(opts)
    )
    trainer = DDPGTrainer(logger, vars(opts))
    trainer.train(agent, env, opts.evaluate)
