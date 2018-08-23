import numpy as np
import torch
from torch.utils.serialization import load_lua


class DataLoader(object):

    def __init__(self, x_bounds=(-25, 25), y_bounds=(-45, 5), grid_step=1, sensor_start=-180, sensor_step=0.5):
        self.min_x, self.max_x = x_bounds
        self.min_y, self.max_y = y_bounds
        self.grid_step = grid_step
        self.sensor_start = sensor_start
        self.sensor_step = sensor_step
        self.data = None
        self.height = None
        self.width = None
        self.dist = None
        self.index = None

        assert self.min_x < self.max_x
        assert self.min_y < self.max_y

    def __getitem__(self, i):
        dist = self.data[i].__getitem__(self.index).reshape(self.height, self.width)
        input_data = torch.empty((2, self.height, self.width), dtype=torch.float)
        input_data[0] = torch.lt(torch.abs(dist - self.dist), self.grid_step * 0.7071)
        input_data[1] = torch.gt(dist + self.grid_step * 0.7071, self.dist)
        return input_data.cuda()

    def get_training_size(self):
        return self.data.shape[0]

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def load_sensor_data(self, file):
        self.data = load_lua(file)
        self.width = int((self.max_x - self.min_x) / self.grid_step) + 1
        self.height = int((self.max_y - self.min_y) / self.grid_step) + 1
        self.dist = torch.empty((self.height, self.width), dtype=torch.float)
        self.index = torch.empty((self.height, self.width), dtype=torch.long)
        for y in range(self.height):
            for x in range(self.width):
                px = x * self.grid_step + self.min_x
                py = y * self.grid_step + self.min_y
                angle = np.degrees(np.arctan2(px, py))
                self.dist[y][x] = np.sqrt(px * px + py * py)
                self.index[y][x] = np.floor((angle - self.sensor_start) / self.sensor_step + 1.5) - 1
        self.index = self.index.reshape(self.width * self.height)
