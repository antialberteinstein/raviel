import torch.nn as nn
from mlp import FeedForward
from multihead import MultiHeadAttention
class Block(nn.Module):
    """Transformer block: multi-head attention + MLP with residuals."""

    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # Pre-norm for stability.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections with pre-norm.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x
