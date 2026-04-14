import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_head.rope import precompute_freqs_cis
from block.transformer_block import Block
from config.config import block_size, device, n_embd, n_head, n_layer, vocab_size


class LargeLanguageModel(nn.Module):
    """Causal Transformer language model."""

    def __init__(self):
        super().__init__()

        # Token and position embeddings.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks.
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        # Final layer norm and projection to vocab logits.
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Tính freqs_cos, freqs_sin cho RoPE
        head_size = n_embd / n_head
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            dim=head_size, seq_len=block_size
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Lookup token + position embeddings.
        tok_emb = self.token_embedding_table(idx)

        # Combine token and position information.
        x = tok_emb

        for block in self.blocks:
            x = block(x, self.freqs_cos[:T], self.freqs_sin[:T])

        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Autoregressive generation.
        for _ in range(max_new_tokens):
            # Keep context within block_size.
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            # Use the last token's logits.
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            # Sample next token id.
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
