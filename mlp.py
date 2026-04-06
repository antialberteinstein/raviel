import torch.nn as nn

class FeedForward(nn.Module):
    """Token-wise MLP used inside each Transformer block."""

    def __init__(self, n_embd) -> None:
        super().__init__()

        # Expand -> activation -> project back.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
