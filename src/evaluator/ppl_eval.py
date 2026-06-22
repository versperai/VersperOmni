import torch
import math
from .config import PPL_EvalConfig
from .metrics import EvalMetrics


class VersperOmniEvaluator:
    def __init__(self, model, tokenizer, config: PPL_EvalConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PPL_EvalConfig()
        self.model.eval()

    def compute_metrics(self, text: str) -> dict:
        # 1. 编码与基础分母统计
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(
            self.config.device
        )
        seq_len = input_ids.size(1)

        word_count = len(text.split())
        byte_count = len(text.encode("utf-8"))

        total_loss_sum = 0.0
        total_predicted_tokens = 0
        prev_end_loc = 0

        # 2. 滑动窗口无损跨越长文本
        for begin_loc in range(0, seq_len, self.config.stride):
            end_loc = min(begin_loc + self.config.max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # 本次窗口实际预测的全新 Token 数

            window_input_ids = input_ids[:, begin_loc:end_loc]

            # 关键工程点：构建 Target Labels，并将历史 Context 设为 -100 隔离 Loss
            target_ids = window_input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.inference_mode():
                # 兼容标准 HuggingFace CausalLM 接口，或你自定义的 model.forward(labels=...)
                outputs = self.model(window_input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            # 累加真实预测部分的 Nats 损耗
            total_loss_sum += neg_log_likelihood.item() * trg_len
            total_predicted_tokens += trg_len

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        # 3. 多维度评测指标输出
        avg_token_loss = total_loss_sum / total_predicted_tokens
        avg_word_loss = total_loss_sum / max(word_count, 1)
        total_bits = EvalMetrics.nats_to_bits(total_loss_sum)

        return {
            "token_ppl": math.exp(avg_token_loss),
            "word_ppl": math.exp(avg_word_loss),
            "bits_per_byte": total_bits / max(byte_count, 1),
            "stats": {
                "tokens": total_predicted_tokens,
                "words": word_count,
                "bytes": byte_count,
            },
        }
