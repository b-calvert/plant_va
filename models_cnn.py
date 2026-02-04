import torch
import torch.nn as nn


class CNN1DBinary(nn.Module):
    """
    Input:  x [B, C, T]
    Output: logits [B] 
    """
    def __init__(self, n_channels: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # -> [B, hidden*2, 1]
            nn.Flatten(),             # -> [B, hidden*2]
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1), # -> [B,1]
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
