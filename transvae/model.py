import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

import tvae_util as uu
from tvae_util import clones, attention
from opt import NoamOpt
from data import data_gen, make_std_mask
from loss import ce_loss

####### Encoder, Decoder and Generator ############

class TransVAE():
    """
    TransVAE class. Between the encoder and decoder is a stochastic
    latent space
    """
    def __init__(self, params, name=None, N=3, d_model=512, d_ff=2048, h=8, dropout=0.1):
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
            h (int): Number of heads per attention layer
            dropout (float): Rate of dropout
        """
        ### Initiate Default Parameters
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            self.params['BATCH_SIZE'] = 500
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.1
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

        ### Build model architecture
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        encoder = VAEEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        decoder = VAEDecoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
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
        self.optimizer = NoamOpt(d_model, 2, 4000,
                                 torch.optim.Adam(self.model.parameters(), lr=0,
                                 betas=(0.9,0.98), eps=1e-9))

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

    def train(self, train_data, val_data, epochs=100, save=True, log=True):
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

        ### Epoch loop
        for epoch in range(epochs):
            ### Train Loop
            self.model.train()
            for j, data in enumerate(train_iter):
                if self.use_gpu:
                    data = data.cuda()

                src = Variable(data[:,:-1], requires_grad=False).long()
                tgt = Variable(data[:,1:-1], requires_grad=False).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)
                tgt_mask = make_std_mask(tgt, self.pad_idx)
                scores = Variable(data[:,-1], requires_grad=False)

                x_out, mu, logvar = self.model(src, tgt, src_mask, tgt_mask)
                loss, bce, kld = ce_loss(src, x_out, mu, logvar,
                                         self.params['CHAR_WEIGHTS'])
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'train',
                                                                loss.item(),
                                                                bce.item(),
                                                                kld.item()))
                    log_file.close()

            ### Val Loop
            self.model.eval()
            losses = []
            for j, data in enumerate(val_iter):
                if self.use_gpu:
                    data = data.cuda()

                src = Variable(data[:,:-1], requires_grad=False).long()
                tgt = Variable(data[:,1:-1], requires_grad=False).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)
                tgt_mask = make_std_mask(tgt, self.pad_idx)
                scores = Variable(data[:,-1], requires_grad=False)

                x_out, mu, logvar = self.model(src, tgt, src_mask, tgt_mask)
                loss, bce, kld = ce_loss(src, x_out, mu, logvar,
                                         self.params['CHAR_WEIGHTS'])
                losses.append(loss.item())

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'test',
                                                                loss.item(),
                                                                bce.item(),
                                                                kld.item()))
                    log_file.close()

            self.n_epochs += 1
            val_loss = np.mean(losses)

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


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences"
        z, mu, logvar = self.encode(src, src_mask)
        x = self.decode(z, src_mask, tgt, tgt_mask)
        x = self.generator(x)
        return x, mu, logvar

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class VAEEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(VAEEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        # Adding Convolutional Bottleneck
        self.conv_layers = [self.conv1, self.conv2, self.conv3]
        in_d = layer.size
        first = True
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 21
                first = False
            else:
                kernel_size = 20
            if i == 2:
                out_d = 64
            self.conv_layers[i] = nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2))
            in_d = out_d
        # self.conv1 = self.conv_layers[0]
        # self.conv2 = self.conv_layers[1]
        # self.conv3 = self.conv_layers[2]
        self.z_means = nn.Linear(320, 128)
        self.z_var = nn.Linear(320, 128)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu * eps*std

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
        x = x.contiguous().view(x.size(0), -1)
        mu, logvar = self.z_means(x), self.z_var(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class VAEDecoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer, N):
        super(VAEDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        # Reshaping memory with deconvolution
        self.deconv_layers = []
        if layer.size == 256:
            self.deconv_layers.append(nn.Linear(128, 256))
        elif layer.size == 512:
            self.deconv_layers.append(nn.Linear(128, 512))
        self.deconv_layers.append(nn.ConvTranspose1d(1, 64, 9, padding=4))
        self.deconv_layers.append(nn.ConvTranspose1d(64, 128, 9, padding=4))
        self.deconv_layers.append(nn.ConvTranspose1d(128, 181, 9, padding=4))

    def forward(self, x, memory, src_mask, tgt_mask):
        memory = memory.unsqueeze(1)
        for deconv in self.deconv_layers:
            memory = F.relu(deconv(memory))
        memory = self.norm(memory)
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super(MultiHeadedAttention, self).__init__()
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
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

############## Embedding Layers ###################

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
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

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
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
        super(SublayerConnection, self).__init__()
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
