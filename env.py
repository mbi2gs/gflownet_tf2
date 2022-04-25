import numpy as np
from matplotlib import pyplot as plt


class RewardEnvironment():
    """Reproduce environment used in
    https://arxiv.org/abs/2106.04399 and
    https://arxiv.org/abs/2201.13259
    """
    def __init__(self, dim=2, H=8, r0=0.01, r1=0.5, r2=1):
        """Initialize environment object.
        :param dim: (int) Number of dimensions
        :param H: (int) "Height" (or length, if you prefer) of each dimension.
        :param r0: (float) Lowest reward value (see reward_fn)
        :param r1: (float) Medium reward value (see reward_fn)
        :param r2: (float) Highest reward value (see reward_fn)
        """
        assert H > 3
        assert r0 > 0
        assert r1 > 0
        assert r2 > 0
        self.dim = dim
        self.length = H
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.populate_reward_space()


    def reward_fn(self, x_coord):
        """Calculate reward function according to section 4.1 in https://arxiv.org/abs/2106.04399.
        :param x_coord: (array-like) Coordinates
        :return: (float) Reward value at coordinates
        """
        assert len(x_coord) >= 1
        assert len(x_coord) == self.dim
        r1_term = 1
        r2_term = 1
        reward = self.r0
        for i in range(len(x_coord)):
            r1_term *= int(0.25 < np.abs(x_coord[i]/(self.length-1) - 0.5))
            r2_term *= int(0.3 < np.abs(x_coord[i]/(self.length-1) - 0.5) <= 0.4)
        reward += self.r1*r1_term + self.r2*r2_term
        return reward


    def populate_reward_space(self):
        """Calculate reward for every position in grid with
        `self.dim` dimensions each of length `self.length`.
        :return: (None) Saves values to self.reward and self.env_prob
        """
        # Calculate reward for every position in grid with
        # "self.dim" dimensions each of length "self.length"
        shape = [self.length]*self.dim
        reward_space = np.zeros(shape)
        for coord, i in np.ndenumerate(reward_space):
            reward_space[coord] = self.reward_fn(coord)
        self.reward = reward_space
        self.env_prob = self.reward / np.sum(self.reward)


    def get_reward(self, x_coord):
        """Return reward value at the query coordinates.
        :param x_coord: (array-like) Coordinates
        :return: (float) Reward value at coordinates
        """
        assert len(x_coord) >= 1
        assert len(x_coord) == self.dim
        assert np.all(x_coord >= 0)
        return self.reward[tuple(x_coord)]


    def plot_reward_2d(self):
        """Matplotlib output of first two dimensions of reward environment.
        :return: (None) Matplotlib figure object
        """
        top_slice = tuple([slice(0,self.length),slice(0,self.length)] + [0]*(self.dim - 2))
        plt.imshow(self.reward[top_slice], origin='lower');