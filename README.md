# Giving Attention to Generative VAE Models for _De Novo_ Molecular Design
![Attention Heads](https://github.com/oriondollar/TransVAE/blob/master/imgs/attn_heads.png)
This repo contains the codebase for the attention-based implementations of VAE models for molecular design as described in this paper (note - link paper). The addition of attention allows models to learn longer range dependencies between input features and improves the quality and interpretability of learned molecular embeddings. The code is organized by folders that correspond to the following sections:

- **transvae**: code required to run models including model class definitions, data preparation, optimizers, etc.
- **scripts**: scripts for training models, generating samples and performing calculations
- **notebooks**: jupyter notebook tutorials and example calculations
- **checkpoints**: pre-trained model files
- **data**: token vocabularies and weights for ZINC and PubChem datasets (***note - full train and test sets for both [ZINC](https://drive.google.com/file/d/17kGpZOVwIGb_H57f4SvkPagdwqA8tADD/view?usp=sharing) and [PubChem](https://drive.google.com/file/d/1h0OhDtnkPl1FaqsouqiEJ14MqVzfwNJb/view?usp=sharing) are available for download)

## Installation

The code can be installed with pip using the following command `pip install transvae`. [RDKit](https://www.rdkit.org/docs/Install.html) and [tensor2tensor](https://github.com/tensorflow/tensor2tensor) are required for certain visualizations/property calculations and must also be installed (neither of these packages are necessary for training or generating molecules so if you would prefer not to install them then you can simply remove their imports from the source code).

# Training

There are three model types - RNN, RNNAttn and Transformer.
