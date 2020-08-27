import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from tvae_util import *

class Batch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, tgt, scores, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.scores = scores
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def data_gen(data, batch_size, char_dict, pad=0):
    n_batches = data.shape[0] // batch_size
    for i in range(n_batches):
        smiles = data[i*batch_size:(i+1)*batch_size,0]
        scores = torch.tensor(data[i*batch_size:(i+1)*batch_size,1].astype('float32'))
        smiles = [smi_tokenizer(x) for x in smiles]
        encoded_data = torch.empty((batch_size, 181))
        for j, smi in enumerate(smiles):
            encoded_smi = encode_smiles(smi, 180, char_dict)
            encoded_smi = [0] + encoded_smi
            encoded_data[j,:] = torch.tensor(encoded_smi)
        encoded_data = encoded_data.long()
        src = Variable(encoded_data, requires_grad=False)
        tgt = Variable(encoded_data, requires_grad=False)
        scores = Variable(scores, requires_grad=False)
        yield Batch(src, tgt, scores, pad)

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding"
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
