from model import DeepTracker
from utils import GenerateAffineFromOdom
import torch
import numpy as np

batch_size = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dt = DeepTracker((3, batch_size, 16, 256, 256), True).to(device)
dt.hidden = dt.init_hidden()
odom_to_aff = GenerateAffineFromOdom(256, 0.1)

for i in range(100):
    x = torch.randn((batch_size, 2, 256, 256)).to(device)
    odom = np.array([[1, 1, np.pi/2]]*batch_size)
    aff = odom_to_aff(odom).to(device)
    y = dt(x, aff)
    print(y.size())

