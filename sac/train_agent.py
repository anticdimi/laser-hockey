import os
import torch
from laserhockey import hockey_env as h_env
from sac_agent import SACAgent
from importlib import reload
from argparse import ArgumentParser
import sys
from trainer import SACTrainer
import time
import random

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
parser.add_argument('--max_episodes', help='Max episodes for training', type=int, default=5000)
parser.add_argument('--max_steps', help='Max steps for training', type=int, default=250)
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=30)
parser.add_argument('--evaluate_every',
                    help='# of episodes between evaluating agent during the training', type=int, default=1000)
parser.add_argument('--add_self_every',
                    help='# of episodes between adding agent (self) to opponent list', type=int, default=1e5)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--alpha_lr', help='Learning rate', type=float, default=1e-4)
parser.add_argument('--lr_factor', help='Scale learning rate by', type=float, default=0.5)
parser.add_argument('--lr_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--alpha_milestones', help='Learning rate milestones', nargs='+')
parser.add_argument('--update_target_every', help='# of steps between updating target net', type=int, default=1)
parser.add_argument('--gamma', help='Discount', type=float, default=0.95)
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--grad_steps', help='grad_steps', type=int, default=32)
parser.add_argument(
    '--alpha',
    type=float,
    default=0.2,
    help='Temperature parameter alpha determines the relative importance of the entropy term against the reward')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False,
                    help='Automatically adjust alpha')
parser.add_argument('--selfplay', type=bool, default=False, help='Should agent train selfplaf')
parser.add_argument('--soft_tau', help='tau', type=float, default=0.005)
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

    dirname = time.strftime(f'%y%m%d_%H%M%S_{random.randint(0, 1e6):06}', time.gmtime(time.time()))
    abs_path = os.path.dirname(os.path.realpath(__file__))
    logger = Logger(prefix_path=os.path.join(abs_path, dirname),
                    mode=opts.mode,
                    cleanup=True,
                    quiet=opts.q)

    env = h_env.HockeyEnv(mode=mode, verbose=(not opts.q))
    opponents = [
        h_env.BasicOpponent(weak=False),
        h_env.BasicOpponent(weak=False),
    ]

    pretrained_agents = [
        # '/Users/dimi/Coding/laser-hockey/sac/210207_151028_954798/agents/a-13000.pkl',
        # '/Users/dimi/Coding/laser-hockey/sac/210208_094326_681163/agents/agent.pkl'
    ]

    if opts.selfplay:
        for p in pretrained_agents:
            a = SACAgent.load_model(p)
            a.eval()
            opponents.append(a)

    agent = SACAgent(
        logger=logger,
        obs_dim=env.observation_space.shape,
        action_space=env.action_space,
        userconfig=vars(opts)
    )
    trainer = SACTrainer(logger, vars(opts))
    trainer.train(agent, opponents, env, opts.evaluate)
