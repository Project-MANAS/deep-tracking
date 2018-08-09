from model import DeepTracker
import torch
import torch.nn as nn
import torch.nn.functional as F

dt = DeepTracker((3, 1, 16, 256, 256))
#(3, batch, 16, img_x, img_y)

dt.hidden = dt.init_hidden()
for i in range(10):
    x = torch.randn((1, 2, 256, 256))
    y = dt(x)
    print(y.size())

