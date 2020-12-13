import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def vae_loss(x, x_out, mu, logvar, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = 0.
    return BCE + KLD, BCE, KLD

def moses_loss(x, x_out, mu, logvar, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights, ignore_index=26)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = 0.
    return BCE + KLD, BCE, KLD

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = 0.
    return BCEmol + BCEmask + KLD, BCEmol, BCEmask, KLD

def stage2_loss(z, zs, mu, logvar, weights, beta=1):
    "Mean-Squared Error Loss + Kiebler-Lublach Divergence"
    z_out, z_in = zs
    MSE = F.mse_loss(z_out, z_in, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = 0.
    return MSE + KLD, MSE, KLD

class LabelSmoothing(nn.Module):
    "Implement label smoothing"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm
