import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.linear = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq_len=1, input_dim)
        
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        
        attn_scores = self.attention(gru_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden_dim)

        return self.classifier(context)