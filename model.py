# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def build_rope_cache(seq_len, head_dim, dtype, device):
    """
    Builds the frequency components for Rotary Position Embedding (RoPE) without using complex numbers.

    Args:
        seq_len: Sequence length.
        head_dim: Dimension of each attention head.
        dtype: Data type for the tensors.
        device: Device to store the tensors.

    Returns:
        sin_emb: Sine embeddings.
        cos_emb: Cosine embeddings.
    """
    # Compute positions and inverse frequencies
    position = torch.arange(seq_len, dtype=dtype, device=device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    sinusoid_inp = torch.einsum('i,j->ij', position, inv_freq)

    # Compute sine and cosine embeddings
    sin_emb = torch.sin(sinusoid_inp)
    cos_emb = torch.cos(sinusoid_inp)

    return sin_emb, cos_emb


def apply_rotary_emb(x, sin_emb, cos_emb):
    """
    Applies rotary position embedding to the tensor x.

    Args:
        x: Input tensor of shape [batch_size, seq_len, n_head, head_dim].
        sin_emb: Sine embeddings.
        cos_emb: Cosine embeddings.

    Returns:
        x_rotated: Tensor after applying rotary embedding.
    """
    # Get dimensions
    batch_size, seq_len, n_head, head_dim = x.size()
    half_head_dim = head_dim // 2

    # Reshape embeddings for broadcasting
    sin_emb = sin_emb[None, :, None, :]  # Shape: [1, seq_len, 1, half_head_dim]
    cos_emb = cos_emb[None, :, None, :]  # Shape: [1, seq_len, 1, half_head_dim]

    # Split the head dimension
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices

    # Apply rotary transformation
    x_rotated_0 = x1 * cos_emb - x2 * sin_emb
    x_rotated_1 = x1 * sin_emb + x2 * cos_emb

    # Combine and reshape back to original shape
    x_rotated = torch.stack((x_rotated_0, x_rotated_1), dim=-1)
    x_rotated = x_rotated.flatten(-2)

    return x_rotated


class Attention(nn.Module):
    """
    Multi-head attention layer with RoPE embedding.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // self.n_head
        self.att_type = config.att_type
        self.dropout = config.dropout

        if self.att_type != "RoPE-SM":
            raise ValueError(f"Unsupported attention type: {self.att_type}. Only 'RoPE-SM' is supported.")

        # Linear projections for queries, keys, and values
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, sin_emb: torch.Tensor, cos_emb: torch.Tensor, mask):
        """
        Forward pass for the attention layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd].
            sin_emb: Sine embeddings for RoPE.
            cos_emb: Cosine embeddings for RoPE.
            mask: Attention mask tensor.

        Returns:
            Output tensor after attention.
        """
        bsz, seqlen, _ = x.shape

        # Linear projections
        queries = self.wq(x).view(bsz, seqlen, self.n_head, self.head_dim)
        keys = self.wk(x).view(bsz, seqlen, self.n_head, self.head_dim)
        values = self.wv(x).view(bsz, seqlen, self.n_head, self.head_dim)

        # Apply rotary position embedding
        queries = apply_rotary_emb(queries, sin_emb, cos_emb)
        keys = apply_rotary_emb(keys, sin_emb, cos_emb)

        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # Shape: [bsz, n_head, seqlen, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask
        mask = mask.expand(-1, self.n_head, -1, -1)  # Shape: [bsz, n_head, seqlen, seqlen]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Compute attention output
        output = torch.matmul(attention_weights, values)  # Shape: [bsz, n_head, seqlen, head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.resid_dropout(output)

        # Final linear projection
        return self.wo(output)


class MLP(nn.Module):
    """
    Feed-forward network (MLP) used within the transformer block.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass for the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after MLP.
        """
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block consisting of attention and MLP layers.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, sin_emb, cos_emb, mask):
        """
        Forward pass for the transformer block.

        Args:
            x: Input tensor.
            sin_emb: Sine embeddings for RoPE.
            cos_emb: Cosine embeddings for RoPE.
            mask: Attention mask tensor.

        Returns:
            Output tensor after the transformer block.
        """
        # Apply attention layer
        x = x + self.attn(self.ln_1(x), sin_emb, cos_emb, mask)
        # Apply MLP layer
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class ModelConfig:
    """
    Configuration class for the model.

    Attributes:
        block_size: Maximum sequence length.
        vocab_size: Vocabulary size.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to include bias terms in linear layers.
        att_type: Attention type (only 'RoPE-SM' is supported).
        pad_token_id: Token ID for padding.
    """
    block_size: int = 1024
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    att_type: str = "RoPE-SM"
    pad_token_id: int = None


class Model(nn.Module):
    """
    Transformer-based language model with RoPE attention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            word_token_embedding=nn.Embedding(config.vocab_size, config.n_embd),
            dropout=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layer_norm_final=nn.LayerNorm(config.n_embd)
        ))

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, idx, attention_mask=None, targets=None):
        """
        Forward pass for the model.

        Args:
            idx: Input token indices tensor of shape [batch_size, seq_len].
            attention_mask: Attention mask tensor.
            targets: Target token indices tensor for computing loss.

        Returns:
            logits: Output logits tensor.
            loss: Cross-entropy loss (if targets are provided).
        """
        device = idx.device
        bsz, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(idx)

        # Get token embeddings
        tok_emb = self.transformer.word_token_embedding(idx)
        x = self.transformer.dropout(tok_emb)

        # Build RoPE cache
        sin_emb, cos_emb = build_rope_cache(t, self.config.n_embd // self.config.n_head, dtype=x.dtype, device=device)

        # Create masks
        causal_mask = torch.tril(torch.ones((t, t), device=device)).bool()  # Causal mask
        attention_mask = attention_mask.bool()
        attention_mask = attention_mask.unsqueeze(1)  # [bsz, 1, t]
        attention_mask = attention_mask & attention_mask.transpose(1, 2)  # [bsz, t, t]
        mask = causal_mask.unsqueeze(0) & attention_mask  # [bsz, t, t]
        mask = mask.unsqueeze(1)  # [bsz, 1, t, t]

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x, sin_emb, cos_emb, mask)

        x = self.transformer.layer_norm_final(x)

        # Compute logits
        logits = self.lm_head(x)

        # Compute loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id
            )
        else:
            logits = logits[:, [-1], :]
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configures the optimizer for training.

        Args:
            weight_decay: Weight decay coefficient.
            learning_rate: Learning rate.
            betas: Betas for Adam optimizer.
            device_type: Type of device ('cuda' or 'cpu').

        Returns:
            optimizer: Configured optimizer.
        """
        import inspect

        # Separate parameters into decay and no-decay groups
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Check for fused optimizer support
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generates text by sampling from the model.

        Args:
            idx: Input token indices tensor.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Top-P (nucleus) sampling parameter.

        Returns:
            idx: Tensor containing the generated token indices.
        """
        for _ in range(max_new_tokens):
            # Truncate input to block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Get logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply top-P (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(-1, indices_to_remove, -float('Inf'))

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
