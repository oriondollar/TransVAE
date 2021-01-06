import argparse

def train_parser():
    parser = argparse.ArgumentParser()
    # Architecture Parameters
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type='str')
    parser.add_argument('--d_model', default=128, type='int')
    parser.add_argument('--d_feedforward', default=128, type='int')
    parser.add_argument('--d_latent', default=128, type='int')
    # Hyperparameters
    parser.add_argument('--batch_size', default=500, type='int')
    parser.add_argument('--batch_chunks', default=5, type='int')
    parser.add_argument('--beta', default=0.05, type='float')
    parser.add_argument('--beta_init', default=1e-8, type='float')
    parser.add_argument('--anneal_start', default=0, type='int')
    parser.add_argument('--adam_lr', default=3e-4, type='float')
    parser.add_argument('--lr_scale', default=1, type='float')
    parser.add_argument('--warmup_steps', default=10000, type='int')
    parser.add_argument('--eps_scale', default=1, type='float')
    parser.add_argument('--epochs', default=100, type='int')
    # Data Parameters
    parser.add_argument('--data_source', choices=['zinc', 'pubchem', 'custom'],
                        required=True, type='str')
    parser.add_argument('--train_path', default=None, type='str')
    parser.add_argument('--test_path', default=None, type='str')
    parser.add_argument('--vocab_path', default=None, type='str')
    parser.add_argument('--char_weights_path', default=None, type='str')
    # Save Parameters
    parser.add_argument('--save_name', default=None, type='str')
    parser.add_argument('--save_freq', default=5, type='int')

    return parser
