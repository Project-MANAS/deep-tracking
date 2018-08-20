import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_convolution_filters(num_layers, num_channels, num_gates):
    """
    :param num_layers: number of layers in the RNN cell
    :param num_channels: number of channels in the hidden state
    :param num_gates: number of gates required to compute new states (4 for LSTM, 3 for GRU)
    :return: a list containing the convolutions modules as per specification
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filters = [nn.Conv2d(2 * num_channels, num_gates * num_channels, 3, 1, 2 ** i, 2 ** i).to(device)
               for i in range(1, num_layers)]
    filters.append(
        nn.Conv2d(num_layers * num_channels, 1, 3, 1, 2 ** (num_layers - 1), 2 ** (num_layers - 1)).to(device))
    filters.insert(0, nn.Conv2d(2 + num_channels, num_gates * num_channels, 3, 1, 1, 1).to(device))

    return filters


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
        return torch.from_numpy(np.transpose(aff_t, [2, 0, 1])).type(torch.FloatTensor)


class SpatialTransformerModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, affine_matrix):
        affine_grid = F.affine_grid(affine_matrix, img.size())
        return F.grid_sample(img, affine_grid)
