import os
import sys
sys.path.append('{}/transvae'.format(os.getcwd()))
sys.path.append(os.getcwd())
import pickle
import pkg_resources

import numpy as np
import pandas as pd

from trans_models import TransVAE
from rnn_models import RNN, RNNAttn
from parsers import train_parser

DATA_DIR = pkg_resources.resource_filename('transvae', '../data')

def train(args):
    # Build params dict
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps}

    # Load data, vocab and token weights
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
        train_data = pd.read_csv('{}/{}_train.txt'.format(DATA_DIR, args.data_source)).to_numpy()
        test_data = pd.read_csv('{}/{}_test.txt'.format(DATA_DIR, args.data_source)).to_numpy()
        with open('{}/char_dict_{}.pkl'.format(DATA_DIR, args.data_source), 'rb') as f:
            char_dict = pickle.load(f)
        char_weights = np.load('{}/char_weights_{}.npy'.format(DATA_DIR, args.data_source))
        params['CHAR_WEIGHTS'] = char_weights

    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k

    params['CHAR_DICT'] = char_dict
    params['ORG_DICT'] = org_dict

    # Build model
    if args.save_name is None:
        if args.model == 'transvae':
            save_name = '{}_{}x_latent{}_{}_{}'.format(args.d_model,
                                                       args.d_feedforward // args.d_model,
                                                       args.d_latent,
                                                       args.model,
                                                       args.data_source)
        else:
            save_name = '{}_latent{}_{}_{}'.format(args.d_model,
                                                   args.d_latent,
                                                   args.model,
                                                   args.data_source)
    else:
        save_name = args.save_name

    if args.model == 'transvae':
        vae = TransVAE(params=params, name=save_name, d_model=args.d_model,
                       d_ff=args.d_feedforward, d_latent=args.d_latent)
    elif args.model == 'rnnattn':
        vae = RNNAttn(params=params, name=save_name, d_model=args.d_model,
                      d_latent=args.d_latent)
    elif args.model == 'rnn':
        vae = RNN(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent)

    vae.train(train_data, test_data, epochs=args.epochs, save_freq=args.save_freq)


if __name__ == '__main__':
    parser = train_parser()
    args = parser.parse_args()
    train(args)
