import numpy as np
import torch
import time

from model import DeepTrackerGRU, DeepTrackerLSTM
from utils import GenerateAffineFromOdom

"""
Note on model performance

Testing bench:
    PyTorch 0.4.0
    CUDA V9.0.176
    GTX1070 8GB VRAM (laptop)
    CuDNN Benchmark = False
    
Hyperparams:
    batch_size = 2
    image_size = (256, 256)
    spatial_transform = True
    no of hidden state channels = 16

GRU:
    Max sequence length: 30
    Average forward time: 0.0064s
    
LSTM (no peephole):
    Max sequence length: 21
    Average forward time: 0.0088s
    
LSTM (with peephole):
    Max sequence length: 19
    Average forward time:  0.0094s  
"""

batch_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dt = DeepTrackerLSTM((3, batch_size, 8, 256, 256), True, True).to(device)
dt.hidden = dt.init_hidden()
odom_to_aff = GenerateAffineFromOdom(256, 0.1)

count = 0
ttime = 0
for i in range(100):
    x = torch.randn((batch_size, 2, 256, 256)).to(device)
    odom = np.array([[1, 1, np.pi / 2]] * batch_size)
    aff = odom_to_aff(odom).to(device)

    start = time.time()
    y = dt(x, aff)
    end = time.time()

    ttime += (end - start)
    count += 1

    print(ttime / count)
