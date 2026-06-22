import torch
from dataclasses import dataclass


@dataclass
class PPL_EvalConfig:
    stride: int = 512
    max_length: int = 1024  # 根据你 VersperOmni 模型的原生窗口调整
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
