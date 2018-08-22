import time

import numpy as np
import torch

from model import DeepTrackerLSTM, DeepTrackerGRU
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
    Max sequence length: 43
    Average forward time: 0.0044s
    
LSTM (no peephole):
    Max sequence length: 21
    Average forward time: 0.0078s
    
LSTM (with peephole):
    Max sequence length: 19
    Average forward time:  0.0084s  
"""

epochs = 5
seq_len = 200
bptt_len = 20

batch_size = 2
img_dim = 256

assert seq_len % bptt_len == 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = DeepTrackerGRU((3, batch_size, 16, img_dim, img_dim), True, True).to(device)
odom_to_aff = GenerateAffineFromOdom(device, img_dim, 0.1)
optimizer = torch.optim.Adam(dt.parameters())

dataset = torch.randn((1000, img_dim, img_dim)).to(device)
dataset_odom = torch.randn((1000, 3)).to(device)
dataset = dataset.view(-1, batch_size, 2, img_dim, img_dim)
dataset_odom = dataset_odom.view(-1, batch_size, 3)

bce_loss = torch.nn.BCELoss()
loss = 0

for i in range(epochs):
    epoch_loss = 0
    loss = 0
    dt.hidden = dt.init_hidden()

    for j in range(20):
        input = dataset[j]
        odom = dataset_odom[j]
        label = dataset[j + 1][:,0].unsqueeze(1)
        output = dt(input, odom_to_aff(odom.data))

        loss = bce_loss(output, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss)
