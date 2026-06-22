import torch
import math


class EvalMetrics:
    @staticmethod
    def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
        """防止 Float32 溢出的平移 Softmax"""
        norm_logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = norm_logits.exp()
        return probs / probs.sum(axis=-1, keepdim=True)

    @staticmethod
    def nats_to_bits(loss_value: float) -> float:
        """信息论转换：自然对数损失 -> 比特数"""
        return loss_value * math.log2(math.e)
