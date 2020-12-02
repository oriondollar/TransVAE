import os
from time import perf_counter
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from tvae_util import *
from opt import NoamOpt
from data import data_gen, make_std_mask
from loss import ce_loss, vae_ce_loss, trans_ce_loss


####### MODEL SHELL ##########

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """
    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            self.params['BATCH_SIZE'] = 500
        if 'BATCH_CHUNKS' not in self.params.keys():
            self.params['BATCH_CHUNKS'] = 5
        if 'LOSS_FUNC' not in self.params.keys():
            self.params['LOSS_FUNC'] = 'VAE_CE'
        if self.params['LOSS_FUNC'] == 'VAE_CE':
            self.loss_fn = vae_ce_loss
        elif self.params['LOSS_FUNC'] == 'TRANS_CE':
            self.loss_fn = trans_ce_loss
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 0
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'LR' not in self.params.keys():
            self.params['LR'] = 1
        if 'WARMUP_STEPS' not in self.params.keys():
            self.params['WARMUP_STEPS'] = 10000
        if 'EPS_SCALE' not in self.params.keys():
            self.params['EPS_SCALE'] = 1
        if 'CHAR_WEIGHTS' in self.params.keys():
            self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
        else:
            self.params['CHAR_WEIGHTS'] = torch.ones(src_vocab, dtype=torch.float)
        if 'CHAR_DICT' not in self.params.keys() or 'ORG_DICT' not in self.params.keys():
            print("WARNING: MUST PROVIDE VOCABULARY KEY PRIOR TO TRAINING")
        self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.pad_idx = self.params['CHAR_DICT']['_']

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}

    def save(self, state, fn, path='checkpoints'):
        """
        Saves current model state to .ckpt file

        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        os.makedirs(path, exist_ok=True)
        if os.path.splitext(fn)[1] == '':
            if self.name is not None:
                fn += '_' + self.name
            fn += '.ckpt'
        else:
            if self.name is not None:
                fn, ext = fn.split('.')
                fn += '_' + self.name
                fn += '.' + ext
        torch.save(state, os.path.join(path, fn))

    def load(self, checkpoint_path):
        """
        Loads a saved model state

        Arguments:
            checkpoint_path (str, required): Path to saved .ckpt file
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for k in self.current_state.keys():
            self.current_state[k] = loaded_checkpoint[k]

        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        for k, v in self.current_state['params'].items():
            if k not in self.params.keys():
                self.params[k] = v
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])

    def train(self, train_data, val_data, epochs=100, save=True, log=True, make_grad_gif=False):
        """
        Train model and validate

        Arguments:
            train_data (np.array, required): Numpy array containing columns with
                                             smiles and property scores (latter
                                             only used if predict_property=True)
            val_data (np.array, required): Same format as train_data. Used for
                                           model development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            log (bool): If true, writes training metrics to log file
        """
        ### Prepare data iterators
        train_data = data_gen(train_data, char_dict=self.params['CHAR_DICT'])
        val_data = data_gen(val_data, char_dict=self.params['CHAR_DICT'])

        train_iter = torch.utils.data.DataLoader(train_data,
                                                 batch_size=self.params['BATCH_SIZE'],
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False, drop_last=True)
        val_iter = torch.utils.data.DataLoader(val_data,
                                               batch_size=self.params['BATCH_SIZE'],
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']


        torch.backends.cudnn.benchmark = True

        ### Setup log file
        if log:
            os.makedirs('trials', exist_ok=True)
            if self.name is not None:
                log_fn = 'trials/log{}.txt'.format('_'+self.name)
            else:
                log_fn = 'trials/log.txt'
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,bce_loss,kld_loss\n')
            log_file.close()

        ### Gradient Gif
        if make_grad_gif:
            os.makedirs('gif', exist_ok=True)
            images = []
            frame = 0

        ### Initialize Annealer
        kl_annealer = KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                 epochs, self.params['ANNEAL_START'])

        ### Epoch loop
        for epoch in range(epochs):
            ### Train Loop
            self.model.train()
            losses = []
            beta = kl_annealer(epoch)
            for j, data in enumerate(train_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_kld_losses = []
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    if self.use_gpu:
                        batch_data = batch_data.cuda()

                    src = Variable(batch_data[:,:-1]).long()
                    tgt = Variable(batch_data[:,:-2]).long()
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)
                    scores = Variable(data[:,-1])

                    x_out, loss_items = self.model(src, tgt, src_mask, tgt_mask)
                    loss, bce, kld = self.loss_fn(src, x_out, loss_items,
                                                  self.params['CHAR_WEIGHTS'],
                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    loss.backward()
                if make_grad_gif and j % 100 == 0:
                    plt = uu.plot_grad_flow(self.model.named_parameters())
                    plt.title('Epoch {}  Frame {}'.format(epoch+1, frame))
                    fn = 'gif/{}.png'.format(frame)
                    plt.savefig(fn)
                    plt.close()
                    images.append(imageio.imread(fn))
                    frame += 1
                self.optimizer.step()
                self.model.zero_grad()
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                avg_kld = np.mean(avg_kld_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'train',
                                                                avg_loss,
                                                                avg_bce,
                                                                avg_kld))
                    log_file.close()
            train_loss = np.mean(losses)

            ### Val Loop
            self.model.eval()
            losses = []
            for j, data in enumerate(val_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_kld_losses = []
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    if self.use_gpu:
                        batch_data = batch_data.cuda()

                    src = Variable(batch_data[:,:-1]).long()
                    tgt = Variable(batch_data[:,:-2]).long()
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)
                    scores = Variable(data[:,-1])

                    x_out, loss_items = self.model(src, tgt, src_mask, tgt_mask)
                    loss, bce, kld = self.loss_fn(src, x_out, loss_items,
                                                  self.params['CHAR_WEIGHTS'],
                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                avg_kld = np.mean(avg_kld_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'test',
                                                                avg_loss,
                                                                avg_bce,
                                                                avg_kld))
                    log_file.close()

            self.n_epochs += 1
            val_loss = np.mean(losses)
            print('Epoch - {} Train - {} Val - {} KLBeta - {}'.format(self.n_epochs, train_loss, val_loss, beta))

            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.current_state['best_loss'] = self.best_loss
                if save:
                    self.save(self.current_state, 'best')

            if save:
                self.save(self.current_state, 'latest')

        if make_grad_gif:
            imageio.mimsave('grads.gif', images)
            shutil.rmtree('gif')

    ### Sampling and Decoding Functions
    def sample_from_latent(self, size):
        """
        Quickly sample from latent dimension
        """
        z = torch.randn(size, self.model.encoder.z_means.out_features)
        return z

    def greedy_decode(self, mem):
        """
        Greedy decode from model memory.
        """
        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        tgt = torch.ones(mem.shape[0],max_len).fill_(start_symbol).long()

        if self.use_gpu:
            tgt = tgt.cuda()

        for i in range(max_len-1):
            out, _ = self.model.decode(tgt, mem)
            out = self.model.generator(out)
            prob = F.softmax(out[:,i,:], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1
            tgt[:,i+1] = next_word
        decoded = tgt[:,1:]
        return decoded

    def decode_from_src(self, data, method='greedy', return_str=True):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding - 'greedy', 'beam search', 'top_k', 'top_p'
        """
        data = data_gen(data, char_dict=self.params['CHAR_DICT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=False)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']
        
        self.model.eval()
        decoded_smiles = []
        for j, data in enumerate(data_iter):
            log_file = open('accs/temp_log.txt', 'a')
            log_file.write('{}\n'.format(j))
            log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                if self.use_gpu:
                    batch_data = batch_data.cuda()

                ### Run through encoder to get memory
                mem, _, _ = self.model.encode(src)

                ### Decode logic
                if method == 'greedy':
                    decoded = self.greedy_decode(mem)
                else:
                    decoded = None

                if return_str:
                    decoded = decode_smiles(decoded, self.params['ORG_DICT'])
                decoded_smiles += decoded
        return decoded_smiles

    def decode_from_mem(self, n, method='greedy', return_str=True):
        """
        Method for decoding sampled memory back into smiles

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding - 'greedy', 'beam search', 'top_k', 'top_p'
        """
        mem = self.sample_from_latent(n)

        if self.use_gpu:
            mem = mem.cuda()

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem)
        else:
            decoded = None

        if return_str:
            decoded = decode_smiles(decoded, self.params['ORG_DICT'])
        return decoded


####### Encoder, Decoder and Generator ############

class TransVAE(VAEShell):
    """
    Transformer-based VAE class. Between the encoder and decoder is a stochastic
    latent space. "Memory value" matrices are convolved to latent bottleneck and
    deconvolved before being sent to source attention in decoder.
    """
    def __init__(self, params, name=None, N=3, d_model=128, d_ff=512,
                 d_latent=128, h=4, dropout=0.1, bypass_bottleneck=False):
        super().__init__(params, name)
        """
        Instatiating a TransVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_ff (int): Dimensionality of feed-forward layers
            d_latent (int): Dimensionality of latent space
            h (int): Number of heads per attention layer
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
        """

        ### Sequence length hard-coded into model
        self.src_len = 126
        self.tgt_len = 125

        ### Build model architecture
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        encoder = VAEEncoder(EncoderLayer(d_model, self.src_len, c(attn), c(ff), dropout), N, d_latent, bypass_bottleneck, self.params['EPS_SCALE'])
        decoder = VAEDecoder(EncoderLayer(d_model, self.src_len, c(attn), c(ff), dropout),
                             DecoderLayer(d_model, self.tgt_len, c(attn), c(attn), c(ff), dropout), N, d_latent, bypass_bottleneck)
        src_embed = nn.Sequential(Embeddings(d_model, self.vocab_size), c(position))
        tgt_embed = nn.Sequential(Embeddings(d_model, self.vocab_size), c(position))
        generator = Generator(d_model, self.vocab_size)
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = NoamOpt(d_model, self.params['LR'], self.params['WARMUP_STEPS'],
                                 torch.optim.Adam(self.model.parameters(), lr=0,
                                 betas=(0.9,0.98), eps=1e-9))

    def greedy_decode(self, mem, src_mask=None):
        """
        Greedy decode from model memory.
        """
        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long()
        if src_mask is None:
            src_mask = torch.ones(mem.shape[0],max_len+2).bool().unsqueeze(1)

        if self.use_gpu:
            src_mask = src_mask.cuda()
            decoded = decoded.cuda()

        for i in range(max_len):
            decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
            if self.use_gpu:
                decode_mask = decode_mask.cuda()
            out = self.model.decode(mem, src_mask, Variable(decoded),
                                    decode_mask)
            out = self.model.generator(out)
            prob = F.softmax(out[:,-1,:], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1
            next_word = next_word.unsqueeze(1)
            decoded = torch.cat([decoded, next_word], dim=1)

        decoded = decoded[:,1:]
        return decoded


    def decode_from_src(self, data, method='greedy', return_str=True):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding - 'greedy', 'beam search', 'top_k', 'top_p'
        """
        data = data_gen(data, char_dict=self.params['CHAR_DICT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=False)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']

        self.model.eval()
        decoded_smiles = []
        for j, data in enumerate(data_iter):
            log_file = open('accs/temp_log.txt', 'a')
            log_file.write('{}\n'.format(j))
            log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                if self.use_gpu:
                    batch_data = batch_data.cuda()

                src = Variable(data[:,:-1]).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory keys and values
                mem, _, _, _ = self.model.encode(src, src_mask)

                if method=='greedy':
                    decoded = self.greedy_decode(mem, src_mask=src_mask)
                else:
                    decoded = None

                if return_str:
                    decoded = decode_smiles(decoded, self.params['ORG_DICT'])
                decoded_smiles += decoded
        return decoded_smiles

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences"
        mem, mu, logvar, predicted_mask = self.encode(src, src_mask)
        x = self.decode(mem, src_mask, tgt, tgt_mask)
        x = self.generator(x)
        return x, [mu, logvar, predicted_mask, src_mask]

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, mem, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)

class VAEEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size)
        self.z_means, self.z_var = nn.Linear(576, d_latent), nn.Linear(576, d_latent)
        self.norm = LayerNorm(layer.size)
        self.learn_mask1 = nn.Linear(d_latent, d_latent*2)
        self.learn_mask2 = nn.Linear(d_latent*2, layer.src_len+1)

        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def reparameterize(self, mu, logvar, eps_scale=1):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu * eps*std

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        mem = self.norm(x)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
        predicted_mask = F.relu(self.learn_mask1(mu))
        predicted_mask = F.relu(self.learn_mask2(predicted_mask).unsqueeze(1))
        return mem, mu, logvar, predicted_mask

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class VAEDecoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck):
        super().__init__()
        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, N)
        self.norm = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck
        self.size = decoder_layers.size
        self.tgt_len = decoder_layers.tgt_len

        # Reshaping memory with deconvolution
        self.linear = nn.Linear(d_latent, 576)
        self.deconv_bottleneck = DeconvBottleneck(decoder_layers.size)

    def forward(self, x, mem, src_mask, tgt_mask):
        "Pass the memory and target into decoder"
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        mem = self.norm(mem)
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward"
    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory_key, memory_val, src_mask, tgt_mask):
        m_key = memory_key
        m_val = memory_val
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
        return self.sublayer[2](x, self.feed_forward)

############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


############## BOTTLENECKS #################

class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    """
    def __init__(self, size):
        super().__init__()
        conv_layers = []
        in_d = size
        first = True
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9
                first = False
            else:
                kernel_size = 8
            if i == 2:
                out_d = 64
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2)))
            in_d = out_d
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x

class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    def __init__(self, size):
        super().__init__()
        deconv_layers = []
        in_d = 64
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 4 - i
            kernel_size = 11
            if i == 2:
                out_d = size
                stride = 1
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size, stride=stride, padding=2)))
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x

############## Embedding Layers ###################

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function"
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        return self.bn(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))


################ STORAGE #####################

class Transformer():
    """
    Standard transfomer class
    """
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, d_ll=256,
                 h=8, dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        generator = Generator(d_model, tgt_vocab)
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
