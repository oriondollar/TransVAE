import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD + MSE, BCE, KLD, MSE

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence + Mask Length Prediction"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCEmol + BCEmask + KLD + MSE, BCEmol, BCEmask, KLD, MSE
