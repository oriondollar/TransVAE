import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

from TransVAE.transvae.trans_models import TransVAE
from TransVAE.transvae.rnn_models import RNN, RNNAttn
from TransVAE.transvae.tvae_util import calc_entropy
from TransVAE.scripts.parsers import sample_parser

def sample(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)

    ### Calculate entropy depending on sampling mode
    if args.sample_mode == 'rand':
        sample_mode = 'rand'
        sample_dims = None
    else:
        entropy_data = pd.read_csv(args.smiles).to_numpy()
        _, mus, _ = vae.calc_mems(entropy_data, log=False, save=False)
        vae_entropy = calc_entropy(mus)
        entropy_idxs = np.where(np.array(vae_entropy) > args.entropy_cutoff)[0]
        sample_dims = entropy_idxs
        if args.sample_mode == 'high_entropy':
            sample_mode = 'top_dims'
        elif args.sample_mode == 'k_high_entropy':
            sample_mode = 'k_dims'

    ### Generate samples
    samples = []
    n_gen = args.n_samples
    while n_gen > 0:
        current_samples = vae.sample(args.n_samples_per_batch, sample_mode=sample_mode,
                                     sample_dims=sample_dims, k=args.k)
        samples.extend(current_samples)
        n_gen -= len(current_samples)

    samples = pd.DataFrame(samples, columns=['smile'])
    if args.save_path is None:
        os.makedirs('generated', exist_ok=True)
        save_path = 'generated/{}_{}.csv'.format(vae.name, args.sample_mode)
    else:
        save_path = args.save_path
    samples.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = sample_parser()
    args = parser.parse_args()
    sample(args)
