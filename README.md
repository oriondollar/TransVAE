# Giving Attention to Generative VAE Models for _De Novo_ Molecular Design
![Attention Heads](imgs/attn_heads.png)
This repo contains the codebase for the attention-based implementations of VAE models for molecular design as described in this paper (note - link paper). The addition of attention allows models to learn longer range dependencies between input features and improves the quality and interpretability of learned molecular embeddings. The code is organized by folders that correspond to the following sections:

- **transvae**: code required to run models including model class definitions, data preparation, optimizers, etc.
- **scripts**: scripts for training models, generating samples and performing calculations
- **notebooks**: jupyter notebook tutorials and example calculations
- **checkpoints**: pre-trained model files
- **data**: token vocabularies and weights for ZINC and PubChem datasets (***note - full train and test sets for both [ZINC](https://drive.google.com/file/d/17kGpZOVwIGb_H57f4SvkPagdwqA8tADD/view?usp=sharing) and [PubChem](https://drive.google.com/file/d/1h0OhDtnkPl1FaqsouqiEJ14MqVzfwNJb/view?usp=sharing) datasets are available for download)

## Installation

There are three model types - RNN, RNNAttn and Transformer.
