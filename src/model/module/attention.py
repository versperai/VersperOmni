import math
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings(action="ignore")


class SelfAttV1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Linear default have bias
        # input dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # X.shape = (batch_size, seq_len, hidden_dim)
        # the X's final dim can  different from hidden_dim but default same
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        # attention_value.shape = (batch_size, seq_len, seq_len)
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim),
            dim=-1,  # row attention-percentage
        )

        output = torch.matmul(attention_weight, V)

        return output
