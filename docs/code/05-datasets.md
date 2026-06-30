# 数据集实现 -- LM / VLM / Omni Dataset

## 概述

VersperOmni 的三种变体各有配套的 Dataset 实现，位于 `src/versper/dataset/` 目录下。所有 Dataset 均继承 `torch.utils.data.Dataset`。

| Dataset | 文件 | 用途 | 数据格式 |
|---------|------|------|----------|
| `PretrainDataset` | `lm_dataset.py` | 预训练 | JSONL |
| `SFTDataset` | `lm_dataset.py` | 指令微调 | JSONL |
| `DPODataset` | `lm_dataset.py` | DPO 偏好对齐 | JSONL |
| `VLMDataset` | `vlm_dataset.py` | 图文训练 | Parquet |
| `OmniDataset` | `omni_dataset.py` | 全模态训练 | Parquet |

---

## PretrainDataset -- 预训练数据

### 数据格式

JSONL 文件，每行一个 JSON 对象，必须包含 `"text"` 字段：

```json
{"text": "这是预训练语料中的一段文本..."}
```

### 处理流程

```python
def __getitem__(self, index):
    # 1. 读取文本
    text = sample["text"]
    
    # 2. Tokenize + truncate (max_length - 2, 留 BOS/EOS 位置)
    tokens = tokenizer(text, truncation=True, max_length=max_length-2).input_ids
    
    # 3. 添加 BOS 和 EOS
    tokens = [bos_token_id] + tokens + [eos_token_id]
    
    # 4. Padding 到固定长度
    input_ids = tokens + [pad_token_id] * pad_len
    
    # 5. Labels: 复制 input_ids，pad 位置设为 -100
    labels = input_ids.clone()
    labels[input_ids == pad_token_id] = -100
```

- 所有序列 padding 到 `max_length`（默认 512）
- Padding 位置在 labels 中设为 `-100`（CrossEntropy 忽略）

---

## SFTDataset -- 指令微调

### 数据格式

JSONL 文件，每行包含 `"conversations"` 字段：

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个AI助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
  ]
}
```

### 处理流程

```python
def __getitem__(self, index):
    # 1. 预处理对话 (可选添加 system prompt)
    conversations = pre_processing_chat(conversations)
    
    # 2. 应用 chat template → 纯文本
    prompt = tokenizer.apply_chat_template(conversations, tokenize=False)
    
    # 3. 随机移除空的 <think></think> 块
    prompt = post_processing_chat(prompt)
    
    # 4. Tokenize + padding
    input_ids = tokenizer(prompt).input_ids[:max_length]
    
    # 5. 生成 labels (只对 assistant 回答部分计算损失)
    labels = generate_labels(input_ids)
```

### Labels 生成逻辑

`generate_labels` 方法寻找 `bos_id` (`"<|bos|>assistant\n"`) 标记，只有 assistant 回答部分的 token 参与损失计算：

```python
# 搜索模式
bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n").input_ids
eos_id = tokenizer(f"{tokenizer.eos_token}\n").input_ids

# labels 中: 用户/系统部分 = -100, assistant 部分 = input_ids
```

### System Prompt 池

当对话没有 system prompt 时，以 20% 概率随机插入一条：

```python
SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI...",
    "你是minimind，一个小巧但有用的语言模型。",
    "You are a helpful AI assistant.",
    # ... 共 10 条
]
```

---

## DPODataset -- 偏好对齐

### 数据格式

```json
{
  "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "正确回答"}],
  "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "错误回答"}]
}
```

### 输出

返回包含 `x_chosen`, `y_chosen`, `mask_chosen`, `x_rejected`, `y_rejected`, `mask_rejected` 的字典，全部是 `[:-1]` / `[1:]` 的移位序列。

---

## VLMDataset -- 图文数据集

### 数据格式

Parquet 文件，每行包含：
- `"image_bytes"`: 图片的原始字节（单张或列表）
- `"conversations"`: JSON 格式的多轮对话

### 处理流程

```python
def __getitem__(self, index):
    # 1. 读取数据
    row = dataset[index]
    conversations = json.loads(row["conversations"])
    image_bytes = row["image_bytes"]  # bytes / list[bytes]
    
    # 2. 替换 <image> 占位符为 image_special_token × image_token_len
    #    "<|image_pad|>" × 64
    prompt = create_chat_prompt(conversations)
    
    # 3. Tokenize 文本
    input_ids = tokenizer(prompt).input_ids[:max_length]
    
    # 4. 生成 labels (同 SFT)
    labels = generate_labels(input_ids)
    
    # 5. 加载图片
    image_inputs = [
        VersperVLM.image2tensor(Image.open(io.BytesIO(img)), processor)
        for img in image_bytes
    ]
    
    return input_ids, labels, image_data
```

### vlm_collate_fn

```python
def vlm_collate_fn(batch):
    # 1. Stack input_ids 和 labels
    # 2. Stack pixel_values
    # 3. 返回 (input_ids, labels, pixel_values)
```

支持 `pixel_values` 为 dict（含 `pixel_attention_mask`）或 tensor 两种格式。

---

## OmniDataset -- 全模态数据集

最复杂的 Dataset，处理文本+音频+图片多模态数据的组装。

### 数据格式

Parquet 文件，可包含以下列：

| 列名 | 说明 |
|------|------|
| `conversations` | 多轮对话 JSON |
| `question_audios` | 用户问题音频 (list[bytes]) |
| `answer_audios` | 助手回答音频 codes (list[int]) |
| `image_bytes` | 图片字节 (list[bytes]) |
| `ref_audios` | 参考音频 codes (语音克隆用) |
| `spk_emb` | 说话人嵌入 (192维 float 数组) |

### 音频增广 (augment_wav)

`omni_dataset.py:96` 实现了丰富的音频增广：

| 增广 | 概率 | 参数 |
|------|------|------|
| Speed Perturbation | 50% | 0.7x ~ 1.6x |
| Additive Noise | 30% | 幅度 0.001 ~ 0.01 |
| Volume Change | 30% | 0.8x ~ 1.2x |
| Time Masking | 20% | 250ms 置零 |
| Lowpass Filter | 20% | 3/5/7 点移动平均 |
| Reverb (IR conv) | 30% | 指数衰减脉冲响应 |
| Pink Noise | 20% | 幅度 0.003 ~ 0.015 |

### SpecAugment (augment_mel)

在 Mel 频谱上做时频掩码：

```python
def augment_mel(self, fbank):
    # Frequency masking: 1~64 频带掩码
    # Time masking: 1~10 帧掩码
```

### 9 流输入构建

`__getitem__` 的核心输出是 9 流输入：

```python
# audio codes: 8 层 (codebook 0..7)
X_audio = [layer[:-1] for layer in Y_audio_layers]  # (8, T-1)
# text tokens
X_text = input_ids[:-1]                              # (1, T-1)

# 拼接为 9 流
input_ids = concat(X_audio, X_text, dim=0)           # (9, T-1)
```

这 9 流中的第 8 流（索引 8）是文本 token，0-7 流是音频 codebook token。

### 带交错目标的音频标签

音频标签使用**交错延迟目标**：

```python
# 第 layer_idx 层 code 从 assistant_start + layer_idx + 1 位置开始
for layer_idx in range(8):
    codes = last_audio_codes[layer_idx]
    start_pos = assistant_start + layer_idx + 1
    for i, code in enumerate(codes):
        audio_labels[layer_idx][start_pos + i] = code
```

这意味着第 0 层 codebook 比文本早 1 步，第 7 层延迟 8 步。

### 计划采样 (Scheduled Sampling)

`omni_dataset.py:222` 实现随机替换输入 token 为随机采样值：

```python
def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
    prob = self.scheduled_sampling_prob  # 默认 0.05
    # 对音频 tokens: 5% 概率替换为随机 code
    # 对文本 tokens: 5% 概率替换为随机 token (排除图像 token)
```

这有助于缓解训练-推理之间的 exposure bias。

### omni_collate_fn

```python
def omni_collate_fn(batch):
    # 1. Stack 9 流 input_ids, text_labels, audio_labels
    # 2. 音频 padding: 将不同长度的 Fbank 填充到相同时间步
    # 3. 图片 padding: stack 或 dict 合并
    # 4. 返回 (input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb)
```

---

## 公用工具

### pre_processing_chat

- 无 system prompt 时以 20% 概率随机插入
- 检测 `tools` 字段，有工具调用时跳过处理

### post_processing_chat

- 检测空的 `<think></think>` 块，以 `empty_think_ratio` 概率移除
- 默认 `empty_think_ratio=0.2`
