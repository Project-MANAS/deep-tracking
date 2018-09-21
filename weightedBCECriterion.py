import torch
import torch.nn as nn

class WeightedBCE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, label, logits):
        logits = torch.squeeze(logits, 0)
        logits = torch.squeeze(logits, 0)
        target, weights = label[0][0], label[0][1]
        true_buff = torch.mul(weights, torch.log(logits + 1e-12))
        true_loss = - torch.mm(target, true_buff)
        neg_buff = torch.mul(weights, torch.log(1 - logits + 1e-12))
        loss = true_loss - torch.sum(neg_buff) + torch.mm(target, neg_buff)
        return torch.mean(loss)

