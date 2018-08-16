from torch.utils.serialization import load_lua
import torch
import numpy as np


class DataLoader(object):

    def __init__(self, grid_minX, grid_maxX, grid_minY, grid_maxY, grid_step, sensor_start, sensor_step):
        self.grid_minX = grid_mixX
        self.grid_maxX = grid_maxX
        self.grid_minY = grid_minY
        self.grid_maxY = grid_maxY
        self.grid_step = grid_step
        self.sensor_start = sensor_step

    def __init__(self):
        self.grid_minX = -25
        self.grid_maxX = 25
        self.grid_minY = -45
        self.grid_maxY = 5
        self.grid_step = 1
        self.sensor_start = -180
        self.sensor_step = 0.5
        
    def __getitem__(self, i):
        dist = self.data[i].__getitem__(self.index).reshape(self.height,self.width)
        input = torch.empty((2, self.height, self.width),dtype=torch.float)
        input[0] = torch.lt(torch.abs(dist - self.dist), self.grid_step * 0.7071)
        input[1] = torch.gt(dist + self.grid_step * 0.7071, self.dist)
        return input
    
    def getTrainingSize(self):
        return self.data.shape[0]
    
    def getHeight(self):
        return self.height
    
    def getWidth(self):
        return self.width

    def LoadSensorData(self, file):
        self.data = load_lua(file)
        self.width = int((self.grid_maxX - self.grid_minX) / self.grid_step) + 1
        self.height = int((self.grid_maxY - self.grid_minY) / self.grid_step) + 1
        self.dist = torch.empty((self.height, self.width), dtype=torch.float)
        self.index = torch.empty((self.height, self.width), dtype=torch.long)
        for y in range(self.height):
            for x in range(self.width):
                px = x * self.grid_step + self.grid_minX
                py = y * self.grid_step + self.grid_minY
                angle = np.degrees(np.arctan2(px, py))
                self.dist[y][x] = np.sqrt(px * px + py * py)
                self.index[y][x] = np.floor((angle - self.sensor_start)/ self.sensor_step + 1.5) - 1
        self.index = self.index.reshape(self.width * self.height)


    def test(self):
        print(self.sensor_start)

obj = DataLoader()
obj.LoadSensorData('/home/vdasu/Downloads/data.t7')