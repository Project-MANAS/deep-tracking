import torch
import torch.nn as nn

class WeightedBCECriterion(nn.Module):

    def __init__(self, inp, target):
        self.input = inp
        self.target = target
        self.eps = 1e-12
        self.buffer = None
        self.output = None
        self.grad_input = None
    
    def update_output(self):
        target, weights = self.target[0], self.target[1]
        self.buffer = self.buffer or self.input.new()
        self.buffer = torch.mul(torch.log(torch.add(self.buffer, self.eps, self.input)), weights)
        self.output = torch.mm(target, self.buffer)
        self.buffer = torch.mul(torch.log(((self.input * -1) + 1 + self.eps)), weights)
        self.output = (self.output - torch.sum(self.buffer) + torch.mm(target, self.buffer)) / self.input.flatten()
        return self.output

    def update_grad_output(self):
        target, weights = self.target[0], self.target[1]
        self.buffer = self.buffer or self.input.new()
        self.buffer = torch.mul(torch.add(self.buffer, -1, self.input) - self.eps, self.input) - self.eps
        self.grad_input = torch.mul(torch.div(torch.add(self.target, -1, self.index), self.buffer), weights) / target.flatten()
        return self.grad_input
