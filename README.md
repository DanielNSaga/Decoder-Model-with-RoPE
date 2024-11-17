# Language Model Training with RoPE Attention

This repository contains code for training a transformer-based language model utilizing Rotary Position Embedding (RoPE) attention mechanism. The model is implemented using PyTorch and is trained on the English Wikipedia dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Tokenizer](#1-training-the-tokenizer)
  - [Preparing the Dataset](#2-preparing-the-dataset)
  - [Training the Model](#3-training-the-model)
- [Model Details](#model-details)
  - [Architecture](#architecture)
  - [Parameters](#parameters)
  - [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [Dataset](#dataset)
- [Training Details](#training-details)
  - [Hyperparameters](#hyperparameters)
  - [Optimizer and Scheduler](#optimizer-and-scheduler)
- [Evaluation](#evaluation)
- [Generating Text](#generating-text)

---

## Introduction

Transformer-based language models have revolutionized natural language processing tasks. This project implements a transformer model with the RoPE attention mechanism, which allows for better handling of positional information without relying on absolute positional encodings.

The model is trained from scratch on the English Wikipedia dataset and demonstrates how to implement custom attention mechanisms in PyTorch.

## Features

- **Custom Tokenizer**: Trains a Byte-Pair Encoding (BPE) tokenizer on the dataset.
- **Transformer Model with RoPE**: Implements a transformer model utilizing RoPE for positional embeddings.
- **Configurable Parameters**: Easy to adjust model size, number of layers, and other hyperparameters.
- **Training Scripts**: Provides scripts for training the tokenizer, preparing the dataset, and training the model.
- **In-Code Documentation**: Code is well-documented for better understanding and extensibility.

## Project Structure
- `tokenizer.py`: Script to train the tokenizer.
- `data_prep.py`: Script to tokenize and prepare the dataset.
- `model.py`: Contains the model architecture and related components.
- `train.py`: Script to train the model.
- `requirements.txt`: Lists all required Python packages.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 2.0 or higher
- CUDA-enabled GPU (recommended for training)

### Install Dependencies

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```
Install the required packages using:
```bash
pip install -r requirements.txt
```
## Usage

### 1. Training the Tokenizer

Run the tokenizer.py script to train the tokenizer on the English Wikipedia dataset.

```bash
python tokenizer.py
```

This script will:

Load the English Wikipedia dataset.
Train a BPE tokenizer with a vocabulary size of 32,768 tokens.
Save the tokenizer as tokenizer.json in the current directory.

### 2. Preparing the Dataset
Run the data_prep.py script to tokenize the dataset and prepare it for training
```bash
python data_prep.py
```

This script will:

Load the English Wikipedia dataset.
Split the dataset into training and validation sets (90% train, 10% validation).
Tokenize the datasets using the trained tokenizer.
Save the tokenized datasets to train_dataset and val_dataset directories.

### 3. Training the Model
Run the train.py script to train the model.
```bash
python train.py
```

This script will:

Load the tokenized training and validation datasets.
Initialize the model with the specified configuration.
Train the model for the specified number of epochs.
Save the best model checkpoint as checkpoint.pth.

## Model Details

### Architecture
The model is a transformer-based language model consisting of:

- An embedding layer for token embeddings.
- Multiple transformer blocks (n_layer), each containing:
Multi-head attention with RoPE.
Layer normalization.
Feed-forward neural network (MLP).
- A final layer normalization layer.
- A linear layer mapping to vocabulary size for language modeling.

## Parameters
The model configuration is defined in the `ModelConfig` class within `model.py`:

- `block_size`: Maximum sequence length the model can handle (default: 1024).
- `vocab_size`: Size of the vocabulary (default: 32768).
- `n_layer`: Number of transformer layers (default: 12).
- `n_head`: Number of attention heads (default: 12).
- `n_embd`: Embedding dimension (default: 768).
- `dropout`: Dropout probability (default: 0.1).
- `bias`: Whether to include bias terms in linear layers (default: True).
- `att_type`: Attention type (only 'RoPE-SM' is supported).
- `pad_token_id`: Token ID for padding (set based on tokenizer).

You can adjust these parameters in the `train.py` script when initializing `ModelConfig`.

## Rotary Position Embedding (RoPE)
RoPE introduces relative positional information to the model by applying a rotation to the queries and keys in the attention mechanism. This allows the model to generalize better to sequences longer than those seen during training.

In this implementation:
- RoPE embeddings are computed without using complex numbers for efficiency.
- The `build_rope_cache` function precomputes sine and cosine embeddings.
- The `apply_rotary_emb` function applies the rotation to queries and keys.

## Dataset
The dataset used is the English Wikipedia dataset available on Hugging Face Datasets:

- **Dataset Name**: `lucadiliello/english_wikipedia`
- **Content**: Contains English Wikipedia articles.
- **Usage**: Used for training and validating the language model.

## Training Details

### Hyperparameters
Training hyperparameters can be adjusted in the `train.py` script:

- `num_epochs`: Number of training epochs (default: 3).
- `batch_size`: Number of samples per batch (default: 8).
- `learning_rate`: Learning rate for the optimizer (default: 3e-4).
- `weight_decay`: Weight decay coefficient for regularization (default: 0.1).
- `betas`: Beta coefficients for the AdamW optimizer (default: (0.9, 0.95)).
- `max_grad_norm`: Maximum gradient norm for gradient clipping (default: 1.0).

### Optimizer and Scheduler
- **Optimizer**: Uses AdamW optimizer with weight decay.
- **Scheduler**: Uses CosineAnnealingLR learning rate scheduler to adjust the learning rate over time.
- **Gradient Clipping**: Gradient norms are clipped to prevent exploding gradients.

### Mixed Precision Training
The training script utilizes Automatic Mixed Precision (AMP) if CUDA is available, which can speed up training and reduce memory usage.

## Evaluation
During training, the model's performance is evaluated on the validation set after each epoch. The script reports:

- **Training Loss**: Average loss on the training set for each epoch.
- **Validation Loss**: Average loss on the validation set for each epoch.

The best model (with the lowest validation loss) is saved as `checkpoint.pth`.

## Generating Text
After training, you can use the model to generate text:

Parameters for text generation:

`max_new_tokens`: Maximum number of tokens to generate.
`temperature`: Controls randomness in sampling (higher values lead to more random outputs).
`top_k`: Limits sampling to the top K tokens with the highest probability.
`top_p`: Enables nucleus sampling, limiting tokens to a cumulative probability.


