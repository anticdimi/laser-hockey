from argparse import ArgumentParser
from laserhockey import hockey_env as h_env
import os

import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils.utils import *
from base.evaluator import evaluate

parser = ArgumentParser()

# Evaluation params
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=30)
parser.add_argument('--filename', help='Path to the pretrained model', default=None)
parser.add_argument('--mode', help='Mode for evaluating currently: (shooting | defense)', default='normal')
parser.add_argument('--show', help='Set if want to render training process', action='store_true')
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--opposite', help='Evaluate agent on opposite side', action='store_true')

opts = parser.parse_args()

if __name__ == '__main__':
    if opts.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif opts.mode == 'shooting':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')

    logger = Logger(prefix_path=os.path.dirname(os.path.realpath(__file__)) + '/logs',
                    mode=opts.mode,
                    quiet=opts.q)
    q_agent = logger.load_model(filename=opts.filename)
    q_agent._config['show'] = opts.show
    q_agent._config['max_steps'] = 250
    q_agent.eval()
    env = h_env.HockeyEnv(mode=mode)
    opponent = h_env.BasicOpponent(weak=False)
    evaluate(agent=q_agent, env=env, opponent=opponent, eval_episodes=opts.eval_episodes,
             action_mapping=q_agent.action_mapping, evaluate_on_opposite_side=opts.opposite)
