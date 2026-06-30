# 模型评估指南 — PPL / CER / 说话人相似度

> 项目中使用的几种核心评估方法概览。

## 评估总览

| 指标                 | 说明                           | 适用场景             |
| -------------------- | ------------------------------ | -------------------- |
| **PPL** (Perplexity) | 语言建模质量，滑动窗口策略      | 文本生成，预训练质量 |
| **CER**              | 字符错误率，通过 ASR 转录计算  | 中文语音生成一致性   |
| **WER**              | 词错误率，通过 ASR 转录计算    | 英文语音生成一致性   |
| **Speaker Similarity** | 说话人相似度，CAM++ 余弦相似度 | 语音克隆质量         |

---

## 1. PPL（困惑度）评估

PPL 是衡量语言模型对文本预测能力的基础指标，值越低表示模型对测试文本的拟合越好。

### 使用方式

`src/evaluator/ppl_eval.py` 提供了滑动窗口 PPL 评估器：

```python
from versper.config import VersperConfig
from versper.model import VersperForCausalLM
from evaluator.ppl_eval import VersperOmniEvaluator
from evaluator.config import PPL_EvalConfig

model = VersperForCausalLM(VersperConfig()).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("./model")

eval_cfg = PPL_EvalConfig(stride=512, max_length=1024)
evaluator = VersperOmniEvaluator(model, tokenizer, eval_cfg)

results = evaluator.compute_metrics("Your evaluation text here...")
print(f"Token PPL: {results['token_ppl']:.2f}")
print(f"Word PPL: {results['word_ppl']:.2f}")
print(f"Bits/Byte: {results['bits_per_byte']:.4f}")
```

### 滑动窗口策略

由于长文本无法一次性塞入模型上下文窗口，PPL 评估采用**滑动窗口**方案：

1. 以 `stride` 为步长，`max_length` 为窗口大小，将长文本切分为重叠的窗口
2. 每个窗口中，仅计算**新出现 Token**（即未被前一个窗口覆盖的部分）的 loss
3. 历史上下文 Token 的 label 被设为 `-100`，在交叉熵中被忽略

对应代码片段：

```python
for begin_loc in range(0, seq_len, self.config.stride):
    end_loc = min(begin_loc + self.config.max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # 本次窗口实际预测的全新 Token 数
    window_input_ids = input_ids[:, begin_loc:end_loc]
    target_ids = window_input_ids.clone()
    target_ids[:, :-trg_len] = -100       # 屏蔽历史上下文 loss
    outputs = self.model(window_input_ids, labels=target_ids)
    total_loss_sum += neg_log_likelihood.item() * trg_len
```

### 输出指标

| 指标           | 含义                                 | 公式                  |
| -------------- | ------------------------------------ | --------------------- |
| `token_ppl`    | 逐 Token 困惑度                     | `exp(avg_token_loss)` |
| `word_ppl`     | 逐词困惑度（按空格分词）            | `exp(avg_word_loss)`  |
| `bits_per_byte` | 每字节信息量，与压缩率直接相关       | 基于信息论转换        |

### PPL_EvalConfig 参数

```python
@dataclass
class PPL_EvalConfig:
    stride: int = 512          # 滑动窗口步长
    max_length: int = 1024     # 窗口大小（需 <= 模型最大上下文）
    device: str = "cuda"       # 计算设备
```

### 工具函数

`src/evaluator/metrics.py` 提供了两个实用方法：

```python
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
```

- `stable_softmax`：通过减去最大值避免 FP32 溢出
- `nats_to_bits`：将 nats 单位的 loss 转换为 bits（乘以 log2(e)），用于 `bits_per_byte` 计算

---

## 2. CER / WER 评估

CER（字符错误率）和 WER（词错误率）用于衡量语音生成的质量。评估流程如下：

```
生成语音 → ASR 转录 → 与 ground truth 文本对齐 → 计算错误率
```

### 评估流程（基于 Versper-O 论文）

1. 使用模型生成语音（Text-to-Audio 或 Audio-to-Audio）
2. 使用 ASR 模型（如 Qwen3-ASR-Flash）将生成的音频转录为文本
3. 将转录结果与 ground truth 文本进行对齐
4. 计算错误率：

```
CER = (插入数 + 删除数 + 替换数) / 总字符数
WER = (插入数 + 删除数 + 替换数) / 总词数
```

### 评估数据集

| 数据集       | 类型         | 说明                           |
| ------------ | ------------ | ------------------------------ |
| T2A test set | Text-to-Audio | 给定文本，评估生成语音的可懂度 |
| A2A test set | Audio-to-Audio | 输入语音，评估语音转换/克隆质量 |

### 注意事项：CER 膨胀

ASR 系统在处理数字和符号时可能与 ground truth 格式不一致，导致 CER 被高估。例如：

| 生成语音 | ASR 转录 | Ground Truth | 问题         |
| -------- | -------- | ------------ | ------------ |
| "300元"  | "三百元" | "300元"      | 数字格式不一致 |

这种差异会**人为增加** CER，在解读指标时需要结合 WER 和其他指标综合判断。

---

## 3. 说话人相似度（Speaker Similarity）

用于评估语音克隆的说话人保真度。

### 评估方法

1. 从生成的音频中提取说话人嵌入（speaker embedding）
2. 从参考音频中提取目标说话人嵌入
3. 计算两个嵌入向量的**余弦相似度**

### CAM++ 模型

CAM++ 是一个说话人识别模型，能够提取 192 维的说话人嵌入向量。相似度计算公式：

```
similarity = cosine_similarity(emb_generated, emb_reference)
```

### 测试说话人

- **5 个内置说话人**（seen，训练阶段见过）
- **7 个留出说话人**（unseen，未见过的说话人）

对 seen 说话人的高相似度反映模型对训练数据的拟合能力；
对 unseen 说话人的高相似度则反映模型的泛化和语音克隆能力。
