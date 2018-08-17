import numpy as np


class GenerateAffineFromOdom:
    def __init__(self, grid_size: int, resolution: float, batch_size: int = None):
        """
        :param grid_size: The grid is assumed to be a square, grid_size is size of either dim
        :param resolution: Calculated as real_world_distance (in m) / grid_size
        :param batch_size: Minibatch size, set as None for runtime inference
        """

        self.grid_size = grid_size
        self.resolution = resolution
        self.batch_size = batch_size

    def __call__(self, odom: np.ndarray):
        """
        :param odom: shape is [batch_size, 3], where each element is [x, y, theta]
                     x and y are in metres
                     theta is in radians (clockwise positive)
        :return: an affine transformation matrix of shape [batch_size, 2, 3]
                [cos(theta) sin(theta) x
                -sin(theta) cos(theta) y]
        """

        if self.batch_size is not None:
            assert odom.shape[0] == self.batch_size

        assert odom.shape[1] == 3

        cos_vals = np.cos(odom[:, 2])
        sin_vals = np.cos(odom[:, 2])
        x = odom[:, 0] / self.resolution
        y = odom[:, 1] / self.resolution

        aff_t = np.array([[cos_vals, sin_vals, x], [-sin_vals, cos_vals, y]])
        return np.transpose(aff_t, [2, 0, 1])