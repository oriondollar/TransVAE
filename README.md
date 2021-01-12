# Giving Attention to Generative VAE Models for _De Novo_ Molecular Design
![Attention Heads](https://raw.githubusercontent.com/oriondollar/TransVAE/master/imgs/attn_heads.png)
This repo contains the codebase for the attention-based implementations of VAE models for molecular design as described in this paper (note - link paper). The addition of attention allows models to learn longer range dependencies between input features and improves the quality and interpretability of learned molecular embeddings. The code is organized by folders that correspond to the following sections:

- **transvae**: code required to run models including model class definitions, data preparation, optimizers, etc.
- **scripts**: scripts for training models, generating samples and performing calculations
- **notebooks**: jupyter notebook tutorials and example calculations
- **checkpoints**: pre-trained model files
- **data**: token vocabularies and weights for ZINC and PubChem datasets (***note - full train and test sets for both [ZINC](https://drive.google.com/file/d/17kGpZOVwIGb_H57f4SvkPagdwqA8tADD/view?usp=sharing) and [PubChem](https://drive.google.com/file/d/1h0OhDtnkPl1FaqsouqiEJ14MqVzfwNJb/view?usp=sharing) are available for download)

## Installation

The code can be installed with pip using the following command `pip install transvae`. [RDKit](https://www.rdkit.org/docs/Install.html) and [tensor2tensor](https://github.com/tensorflow/tensor2tensor) are required for certain visualizations/property calculations and must also be installed (neither of these packages are necessary for training or generating molecules so if you would prefer not to install them then you can simply remove their imports from the source code).

## Training

![Model Types](https://raw.githubusercontent.com/oriondollar/TransVAE/master/imgs/model_types.png)

There are three model types - RNN (a), RNNAttn (b) and Transformer (c). If you've downloaded the ZINC or PubChem training sets from the drive link, you can re-train the models described in the paper with a command such as `python scripts/train.py --model transvae --data_source zinc`. The default model dimension is 128 but this can also be changed at the command line `python scripts/train.py --model rnnattn --d_model 256 --data_source pubchem`. You may also specify a custom train and test set like follows `python scripts/train.py --model transvae --data_source custom --train_path my_train_data.txt --test_path my_test_data.txt --vocab_path my_vocab.pkl --char_weights_path my_char_weights.npy --save_name my_model`. The vocabulary must be a pickle file that stores a dictionary that maps token -> token id and it must begin with the `<start>` or `<bos>` token. All modifiable hyperparameters can be viewed with `python scripts/train.py --help`.
