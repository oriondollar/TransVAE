import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
#             print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

class NoamOpt:
    "Optim wrapper that implements rate"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

class AdamOpt:
    def __init__(self, params, lr, optimizer):
        self.optimizer = optimizer(params, lr)
        self.state_dict = self.optimizer.state_dict()

    def step(self):
        self.optimizer.step()
        self.state_dict = self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
