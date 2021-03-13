from argparse import ArgumentParser
from laserhockey import hockey_env as h_env
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
from utils.utils import *
from base.evaluator import evaluate

parser = ArgumentParser()

# Training params
parser.add_argument('--eval_episodes', help='Set number of evaluation episodes', type=int, default=1000)
parser.add_argument('--filename', help='Path to the pretrained model', default='td3/a-')
parser.add_argument('--mode', help='Mode for evaluating currently: (shooting | defense)', default='normal')
parser.add_argument('--show', help='Set if want to render training process', action='store_true', default=False)
parser.add_argument('--q', help='Quiet mode (no prints)', action='store_true')
parser.add_argument('--opposite', help='Evaluate agent on opposite side', default=False, action='store_true')

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

    logger = Logger(os.path.dirname(os.path.realpath(__file__)) + '/logs', mode=opts.mode, quiet=opts.q)
    stats = []
    for i in range(32000, 35001, 1000):
        q_agent = logger.load_model(opts.filename + str(i) + '.pkl')
        q_agent._config['show'] = opts.show
        q_agent._config['max_steps'] = 250
        env = h_env.HockeyEnv(mode=mode)
        q_agent.eval()
        opponent = h_env.BasicOpponent(weak=False)
        stats.append(evaluate(q_agent, env, opponent, opts.eval_episodes, evaluate_on_opposite_side=opts.opposite)[2])
    print(stats)
