import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *

def vae_data_gen(mols, props, char_dict):
    """
    Encodes input smiles to tensors with token ids

    Arguments:
        mols (np.array, req): Array containing input molecular structures
        props (np.array, req): Array containing scalar chemical property values
        char_dict (dict, req): Dictionary mapping tokens to integer id
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     SMILES string
    """
    smiles = mols[:,0]
    if props is None:
        props = np.zeros(smiles.shape)
    del mols
    smiles = [tokenizer(x) for x in smiles]
    encoded_data = torch.empty((len(smiles), 128))
    for j, smi in enumerate(smiles):
        encoded_smi = encode_smiles(smi, 126, char_dict)
        encoded_smi = [0] + encoded_smi
        encoded_data[j,:-1] = torch.tensor(encoded_smi)
        encoded_data[j,-1] = torch.tensor(props[j])
    return encoded_data

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
