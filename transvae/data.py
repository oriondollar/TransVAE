import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from tvae_util import *

def vae_data_gen(data, char_dict):
    smiles = data[:,0]
    del data
    smiles = [smi_tokenizer(x) for x in smiles]
    encoded_data = torch.empty((len(smiles), 127))
    for j, smi in enumerate(smiles):
        encoded_smi = encode_smiles(smi, 126, char_dict)
        encoded_smi = [0] + encoded_smi
        encoded_data[j,:] = torch.tensor(encoded_smi)
    return encoded_data

def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
