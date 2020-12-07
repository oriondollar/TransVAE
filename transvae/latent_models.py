import os
import json
from time import perf_counter
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from tvae_util import *
from opt import NoamOpt, AdamOpt
from trans_models import VAEShell, LayerNorm

class LatentVAE(VAEShell):
    """
    Second stage VAE used to learn standard normal distribution of latent
    variable z. This may be useful when the latent manifold is well
    approximated but does not match the correct probability distribution.
    (https://arxiv.org/pdf/1903.05789.pdf)
    """
    def __init__(self, params, name=None, N=3, d_model=1024, d_latent=128):
        super().__init__(params, name)
        """
        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
        """

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store
        self.model_type = 'stage2'
        self.N = N
        self.d_model = d_model
        self.d_latent = d_latent

        ### Build model architecture
        encoder = ZEncoder(N, d_model, d_latent)
        decoder = ZDecoder(N, d_model, d_latent)
        self.model = ZEncoderDecoder(encoder, decoder)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)

class ZEncoderDecoder(nn.Module):
    """
    Fully Connected Feedforward Encoder-Decoder Architecture
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, z, tgt=None, src_mask=None, tgt_mask=None):
        u, mu, logvar = self.encoder(z)
        z_out = self.decoder(u)
        return z_out, mu, logvar

class ZEncoder(nn.Module):
    def __init__(self, N, d_model, d_latent):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.d_latent = d_latent

        linear_layers = []
        for i in range(N):
            in_d = self.d_model
            out_d = self.d_model
            if i == 0:
                in_d = self.d_latent
            elif i == N-1:
                out_d = self.d_latent
            linear_layers.append(nn.Linear(in_d, out_d))
        self.linear_layers = ListModule(*linear_layers)
        self.u_means = nn.Linear(d_latent, d_latent)
        self.u_var = nn.Linear(d_latent, d_latent)
        self.norm = LayerNorm(d_latent)

    def reparameterize(self, mu, logvar, eps_scale=1):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std

    def forward(self, z):
        for linear_layer in self.linear_layers:
            z = F.relu(linear_layer(z))
        z = self.norm(z)
        mu, logvar = self.u_means(z), self.u_var(z)
        u = self.reparameterize(mu, logvar)
        return u, mu, logvar

class ZDecoder(nn.Module):
    def __init__(self, N, d_model, d_latent):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.d_latent = d_latent

        linear_layers = []
        for i in range(N):
            in_d = self.d_model
            out_d = self.d_model
            if i == 0:
                in_d = self.d_latent
            elif i == N-1:
                out_d = self.d_latent
            linear_layers.append(nn.Linear(in_d, out_d))
        self.linear_layers = ListModule(*linear_layers)

    def forward(self, u):
        z_out = u.clone()
        for linear_layer in self.linear_layers:
            z_out = F.relu(linear_layer(z_out))
        return z_out
