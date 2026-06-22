import torch.nn as nn
import math


class InputEmbedding:
    def __init__(self, dim_model: int, voice_size: int):
        super().__init__()
        self.dim_model = dim_model
        self.voice_size = voice_size
        self.embedding = nn.Embedding(voice_size, dim_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim_model)

class 
