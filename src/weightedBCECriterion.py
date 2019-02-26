import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class WeightedBCE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, label, logits):
        logits = torch.squeeze(logits, 0)
        logits = torch.squeeze(logits, 0)
        target, weights = label[0][0], label[0][1]

        true_buff = torch.mul(weights, torch.log(logits + 1e-12))
        neg_buff = torch.mul(weights, torch.log(1 - logits + 1e-12))

        true_loss = torch.mul(target, true_buff) + torch.mul(1 - target, neg_buff)

        return -torch.mean(true_loss)

