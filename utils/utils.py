import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import shutil
from tabulate import tabulate
import random
from copy import deepcopy
from laserhockey.hockey_env import CENTER_X, CENTER_Y, SCALE


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def poll_opponent(opponents):
    # TODO: Implement smarter polling
    return random.choice(opponents)


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
        new_transition[3] = compute_winning_reward(transition, is_player_one) + \
            compute_reward_closeness_to_puck(transition)
        new_transition[5]['reward_closeness_to_puck']
        new_transitions.append(tuple(new_transition))

    return new_transitions


class Logger:
    """
    The Logger class is used printing statistics, saving/loading models and plotting.

    Parameters
    ----------
    prefix_path : Path
        The variable is used for specifying the root of the path where the plots and models are saved.
    mode: str
        The variable specifies in which mode we are currently running. (shooting | defense | normal)
    cleanup: bool
        The variable specifies whether the logging folder should be cleaned up.
    quiet: boolean
        This variable is used to specify whether the prints are hidden or not.
    """

    def __init__(self, prefix_path, mode, cleanup=False, quiet=False) -> None:
        self.prefix_path = Path(prefix_path)

        self.agents_prefix_path = self.prefix_path.joinpath('agents')
        self.plots_prefix_path = self.prefix_path.joinpath('plots')
        self.arrays_prefix_path = self.prefix_path.joinpath('arrays')

        self.prefix_path.mkdir(exist_ok=True)

        if cleanup:
            self._cleanup()

        self.quiet = quiet

        if not self.quiet:
            print(f"Running in mode: {mode}")

    # TODO: add logging to file if needed
    def info(self, message):
        print(message)

    def save_model(self, model, filename):
        savepath = self.agents_prefix_path.joinpath(filename).with_suffix('.pkl')
        with open(savepath, 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    def print_episode_info(self, game_outcome, episode_counter, step, total_reward, epsilon=None, touched=None):
        if not self.quiet:
            padding = 8 if game_outcome == 0 else 0
            msg_string = '{} {:>4}: Done after {:>3} steps. \tReward: {:<15}'.format(
                " " * padding, episode_counter, step + 1, round(total_reward, 4))

            if touched is not None:
                msg_string = '{}Touched: {:<15}'.format(msg_string, int(touched))

            if epsilon is not None:
                msg_string = '{}Epsilon: {:<5}'.format(msg_string, round(epsilon, 2))

            print(msg_string)

    def print_stats(self, rew_stats, touch_stats, won_stats, lost_stats):
        if not self.quiet:
            print(tabulate([['Mean reward', np.around(np.mean(rew_stats), 3)],
                            ['Mean touch', np.around(np.mean(list(touch_stats.values())), 3)],
                            ['Mean won', np.around(np.mean(list(won_stats.values())), 3)],
                            ['Mean lost', np.around(np.mean(list(lost_stats.values())), 3)]], tablefmt='grid'))

    def load_model(self, filename):
        if filename is None:
            load_path = self.agents_prefix_path.joinpath('agent.pkl')
        else:
            load_path = Path(filename)
        with open(load_path, 'rb') as inp:
            return pickle.load(inp)

    def hist(self, data, title, filename=None, show=True):
        plt.figure()
        plt.hist(data, density=True)
        plt.title(title)

        plt.savefig(self.reward_prefix_path.joinpath(filename).with_suffix('.pdf'))
        if show:
            plt.show()
        plt.close()

    def plot_running_mean(self, data, title, filename=None, show=True, v_milestones=None):
        data_np = np.asarray(data)
        mean = running_mean(data_np, 1000)
        self._plot(mean, title, filename, show)

    def plot_evaluation_stats(self, data, eval_freq, filename):
        style = {
            'weak': 'dotted',
            'strong': 'solid'
        }

        xlen = 0
        for opponent in data.keys():
            stats = data[opponent]
            xlen = len(stats['won'])
            x = np.arange(eval_freq, eval_freq * xlen + 1, eval_freq)
            plt.plot(
                x,
                stats['won'],
                label=f'Won vs {opponent} opponent',
                color='blue',
                linestyle=style[opponent]
            )
            plt.plot(
                x,
                stats['lost'],
                label=f'Lost vs {opponent} opponent',
                color='red',
                linestyle=style[opponent]
            )

            self.to_csv(stats['won'], f'{opponent}_won')

        ticks = labels = np.arange(eval_freq, eval_freq * xlen + 1, eval_freq)
        plt.xticks(ticks, labels, rotation=45)
        plt.ylim((0, 1))
        plt.xlim((eval_freq, xlen * eval_freq))
        plt.title('Evaluation statistics')
        plt.xlabel('Number of training episodes')
        plt.ylabel('Percentage of lost/won games in evaluation')

        lgd = plt.legend(bbox_to_anchor=(1.5, 1))
        plt.savefig(
            self.plots_prefix_path.joinpath(filename).with_suffix('.pdf'),
            bbox_extra_artists=(lgd,),
            bbox_inches='tight'
        )
        plt.close()

    def plot(self, data, title, filename=None, show=True):
        self._plot(data, title, filename, show)

    def plot_intermediate_stats(self, data, show=True):
        self._plot((data["won"], data["lost"]), "Evaluation won vs loss", "evaluation-won-loss", show, ylim=(0, 1))

        for key in data.keys() - ["won", "lost"]:
            title = f'Evaluation {key} mean'
            filename = f'evaluation-{key}.pdf'

            self._plot(data[key], title, filename, show)

    def _plot(self, data, title, filename=None, show=True, ylim=None, v_milestones=None):
        plt.figure()
        # Plotting Won vs lost
        if isinstance(data, tuple):
            plt.plot(data[0], label="Won", color="blue")
            plt.plot(data[1], label="Lost", color='red')
            plt.ylim(*ylim)
            plt.legend()
        else:
            plt.plot(data)
        plt.title(title)

        if v_milestones is not None:
            plt.vlines(
                v_milestones,
                linestyles='dashed',
                colors='orange',
                label='Added self as opponent',
                linewidths=0.5,
                ymin=np.min(data),
                ymax=np.max(data)
            )

        plt.savefig(self.plots_prefix_path.joinpath(filename).with_suffix('.pdf'))
        if show:
            plt.show()

        plt.close()

    def to_csv(self, data, filename):
        savepath = self.arrays_prefix_path.joinpath(filename).with_suffix('.csv')
        np.savetxt(savepath, data, delimiter=',')

    def save_array(self, data, filename):
        savepath = self.arrays_prefix_path.joinpath(filename).with_suffix('.pkl')
        with open(savepath, 'wb') as outp:
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    def load_array(self, filename):
        loadpath = self.arrays_prefix_path.joinpath(filename).with_suffix('.pkl')
        with open(loadpath, 'rb') as inp:
            return pickle.load(inp)

    # TODO: REMOVE
    def clean_rew_dir(self):
        print("This function will soon be deprecated")
        shutil.rmtree(self.reward_prefix_path, ignore_errors=True)
        self.reward_prefix_path.mkdir(exist_ok=True)

    def _cleanup(self):
        shutil.rmtree(self.agents_prefix_path, ignore_errors=True)
        shutil.rmtree(self.plots_prefix_path, ignore_errors=True)
        shutil.rmtree(self.arrays_prefix_path, ignore_errors=True)
        self.agents_prefix_path.mkdir(exist_ok=True)
        self.plots_prefix_path.mkdir(exist_ok=True)
        self.arrays_prefix_path.mkdir(exist_ok=True)
