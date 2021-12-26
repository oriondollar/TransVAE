import sys
sys.path.append('/gscratch/pfaendtner/orion/TransVAE')

import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from transvae.trans_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn
from scripts.parsers import model_init, train_parser

def train(rank, args):
    ### Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.n_gpus, rank=rank)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    ### Update beta init parameter
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        start_epoch = ckpt['epoch']
        total_epochs = start_epoch + args.epochs
        beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch
        args.beta_init = beta_init

    ### Build params dict
    params = {'ADAM_LR': args.adam_lr*args.n_gpus,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'LR_SCALE': args.lr_scale*args.n_gpus,
              'WARMUP_STEPS': args.warmup_steps}

    ### Load data, vocab and token weights
    if args.data_source == 'custom':
        assert args.train_mols_path is not None and args.test_mols_path is not None and args.vocab_path is not None,\
        "ERROR: Must specify files for train/test data and vocabulary"
        train_mols = pd.read_csv(args.train_mols_path).to_numpy()
        test_mols = pd.read_csv(args.test_mols_path).to_numpy()
        if args.property_predictor:
            assert args.train_props_path is not None and args.test_props_path is not None, \
            "ERROR: Must specify files with train/test properties if training a property predictor"
            train_props = pd.read_csv(args.train_props_path).to_numpy()
            test_props = pd.read_csv(args.test_props_path).to_numpy()
        else:
            train_props = None
            test_props = None
        with open(args.vocab_path, 'rb') as f:
            char_dict = pickle.load(f)
        if args.char_weights_path is not None:
            char_weights = np.load(args.char_weights_path)
            params['CHAR_WEIGHTS'] = char_weights
    else:
        train_mols = pd.read_csv('data/{}_train.txt'.format(args.data_source)).to_numpy()
        test_mols = pd.read_csv('data/{}_test.txt'.format(args.data_source)).to_numpy()
        if args.property_predictor:
            assert args.train_props_path is not None and args.test_props_path is not None, \
            "ERROR: Must specify files with train/test properties if training a property predictor"
            train_props = pd.read_csv(args.train_props_path).to_numpy()
            test_props = pd.read_csv(args.test_props_path).to_numpy()
        else:
            train_props = None
            test_props = None
        with open('data/char_dict_{}.pkl'.format(args.data_source), 'rb') as f:
            char_dict = pickle.load(f)
        char_weights = np.load('data/char_weights_{}.npy'.format(args.data_source))
        params['CHAR_WEIGHTS'] = char_weights

    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k

    params['CHAR_DICT'] = char_dict
    params['ORG_DICT'] = org_dict

    ### Train model
    vae = model_init(args, params, rank=rank)
    if args.checkpoint is not None:
        vae.load(args.checkpoint)
    torch.cuda.set_device(rank)
    with open('write_params.txt', 'a') as f:
        f.write('{}\n'.format(rank))
        for i, (name, param) in enumerate(vae.model.named_parameters()):
            if i > 0:
                break
            f.write('{}: {}\n'.format(name, param.data))
    vae.train(train_mols, test_mols, train_props, test_props,
              epochs=args.epochs, save_freq=args.save_freq)


if __name__ == '__main__':
    parser = train_parser()
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    args.n_gpus = torch.cuda.device_count()
    mp.spawn(train, nprocs=args.n_gpus, args=(args,))
