import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree


class ExperienceReplay:
    """
    The Memory class implements an abstract class for an experience replay buffer.

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


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, max_size=100000):
        super(UniformExperienceReplay, self).__init__(max_size)

    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size
        indices = np.random.choice(range(self.size), size=batch_size, replace=False)
        return self._transitions[indices, :]


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, max_size, alpha, beta):
        super(PrioritizedExperienceReplay, self).__init__(max_size)
        self._alpha = alpha
        self._beta = beta
        self._max_priority = 1.0

        st_capacity = 1
        while st_capacity < max_size:
            st_capacity *= 2
        self._st_sum = SumSegmentTree(st_capacity)
        self._st_min = MinSegmentTree(st_capacity)

    def add_transition(self, transitions_new):
        idx = self._current_idx
        super(PrioritizedExperienceReplay, self).add_transition(transitions_new)
        self._st_min[idx] = self._max_priority ** self._alpha
        self._st_sum[idx] = self._max_priority ** self._alpha

    def _sample_proportionally(self, batch_size):
        indices = []
        p_total = self._st_sum.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.uniform(0, 1) * every_range_len + i * every_range_len
            idx = self._st_sum.find_prefixsum_idx(mass)
            indices.append(idx)
        return np.array(indices)

    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size
        indices = self._sample_proportionally(batch_size)
        weights = []

        # obtain the min probability (max weight accordingly) to scale the other weights (for stability)
        p_min = self._st_min.min() / self._st_sum.sum()
        max_weight = (p_min * self.size) ** (-self._beta)

        for idx in indices:
            # compute probability P(i)
            p_sample = self._st_sum[idx] / self._st_sum.sum()
            weight = (p_sample * self.size) ** (-self._beta)
            weights.append(weight / max_weight)

        weights = np.expand_dims(np.array(weights), axis=0)
        indices = np.expand_dims(indices, axis=0)

        return np.concatenate([self._transitions[indices, :], weights, indices], axis=-1)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            assert priority > 0 # send abs of error
            assert 0 <= idx < self.size
            self._st_sum[idx] = priority ** self._alpha
            self._st_sum[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def update_beta(self, beta):
        # (read up more on how to change it, but I would suggest as 1-eps and update it every episode;
        # think about setting eps as 0.8 or something for it to be closer to the value in the paper (around 0.4)
        self._beta = beta
