import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

from transvae.trans_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn
from scripts.parsers import model_init, train_parser

def train(args):
    ### Build params dict
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps}

    ### Load data, vocab and token weights
    if args.data_source == 'custom':
        assert args.train_path is not None and args.test_path is not None and args.vocab_path is not None,\
        "ERROR: Must specify files for train/test data and vocabulary"
        train_data = pd.read_csv(args.train_path).to_numpy()
        test_data = pd.read_csv(args.test_path).to_numpy()
        with open(args.vocab_path, 'rb') as f:
            char_dict = pickle.load(f)
        if args.char_weights_path is not None:
            char_weights = pd.read_csv(args.char_weights_path)
            params['CHAR_WEIGHTS'] = char_weights
    else:
        train_data = pd.read_csv('data/{}_train.txt'.format(args.data_source)).to_numpy()
        test_data = pd.read_csv('data/{}_test.txt'.format(args.data_source)).to_numpy()
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
    vae = model_init(args, params)
    vae.train(train_data, test_data, epochs=args.epochs, save_freq=args.save_freq)


if __name__ == '__main__':
    parser = train_parser()
    args = parser.parse_args()
    train(args)
