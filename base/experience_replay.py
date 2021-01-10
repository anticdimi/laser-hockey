import numpy as np


class ExperienceReplay:
    """
    The ExperienceReplay class implements a base class for an experience replay buffer.

    Parameters
    ----------
    max_size : int
        The variable specifies maximum number of (s, a, r, new_state, done) tuples in the buffer.
    """

    def __init__(self, max_size=100000):
        self._transitions = np.asarray([])
        self._current_idx = 0
        self.size = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self._transitions = np.asarray(blank_buffer)

        self._transitions[self._current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self._current_idx = (self._current_idx + 1) % self.max_size

    def sample(self, batch_size):
        raise NotImplementedError("Implement the sample method")
