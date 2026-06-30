# 快速推理指南 — LM / VLM / Omni

> 本文提供 VersperOmni 三种模型的可运行推理示例。所有代码均基于 `src/versper/` 包路径。
>
> **前置条件**：已完成环境搭建（见 [00-setup.md](00-setup.md)）且模型权重已下载至 `./model/` 目录。

---

## 1. LM 文本生成

最基本的用法：加载 MiniMind 纯语言模型，给定 prompt 生成文本回复。

```python
import torch
from transformers import AutoTokenizer
from versper.config import MiniMindConfig
from versper.model import MiniMindForCausalLM

# 初始化模型
cfg = MiniMindConfig()
model = MiniMindForCausalLM(cfg).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("./model")

# 加载预训练权重
state_dict = torch.load("./model/pretrain_768.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)

# 构造 prompt
prompt = "介绍一下深度学习"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

# 生成
out = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.85,
    top_p=0.85,
)
response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### 交互式对话

项目自带一个简单的交互式 demo 脚本：

```bash
python src/versper/scripts/web_demo.py
```

该脚本会加载 `./model/pretrain_768.pth` 权重并在终端中持续接收用户输入。

---

## 2. VLM 图像描述

MiniMind-V 在 LM 基础上增加了 SigLIP2 视觉编码器，支持图像输入。

```python
import torch
from PIL import Image
from versper.config import VLMConfig
from versper.vlm import MiniMindVLM

# 初始化模型（指定视觉编码器路径）
vcfg = VLMConfig()
model = MiniMindVLM(vcfg, vision_model_path="./model/siglip2-base-p32-256-ve")
model = model.cuda().eval()

# 加载权重
state_dict = torch.load("./model/pretrain_768.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)

# 加载图像
image = Image.open("photo.jpg").convert("RGB")

# 处理输入：process 返回 SigLIP2 格式的 pixel_values
inputs = model.image2tensor(image, model.processor)
pixel_values = inputs["pixel_values"].cuda()

# 构造文本 prompt（图像占位符由模型自动处理）
tokenizer = model.processor  # SigLipImageProcessor，不包含 tokenizer
# 需手动导入 AutoTokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")

prompt = "Describe this image"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

# 生成时需要手动注入 vision token
# 在 input_ids 开头插入 image placeholder 标记
marker_id = vcfg.image_ids[0]  # 默认为 12
image_token_len = vcfg.image_token_len  # 默认为 64
image_placeholder = torch.full((1, image_token_len), marker_id, dtype=torch.long).cuda()
input_ids = torch.cat([image_placeholder, input_ids], dim=1)

# 生成
out = model.generate(
    input_ids,
    pixel_values=pixel_values,
    max_new_tokens=128,
    temperature=0.85,
    top_p=0.85,
)
response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

> **注意**：VLM 的 `forward` 方法在首步推理时会自动将 `pixel_values` 通过视觉编码器 → Projector 映射为视觉嵌入，并替换 input_ids 中的 `<|image_pad|>` 占位符标记（ID=12）。因此需要在 input_ids 开头预留 `image_token_len` 个占位符。

---

## 3. Omni 文本转语音 (TTS)

MiniMind-O 采用 Thinker-Talker 架构，可在生成文本的同时**流式输出** Mimi 音频编码。

```python
import torch
from transformers import AutoTokenizer
from versper.config import OmniConfig
from versper.omni import MiniMindOmni

# 初始化 Omni 模型
ocfg = OmniConfig()
model = MiniMindOmni(
    ocfg,
    audio_encoder_path="./model/SenseVoiceSmall",
    vision_model_path="./model/siglip2-base-p32-256-ve",  # 可选
)
model = model.cuda().eval()

# 加载权重
state_dict = torch.load("./model/sft_omni_768.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)

# 准备文本输入
tokenizer = AutoTokenizer.from_pretrained("./model")
prompt = "Say hello world"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# 流式生成（stream=True 返回 Python generator）
generator = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.75,
    top_p=0.90,
    stream=True,
    return_audio_codes=True,
)

# 逐帧读取生成结果
for text_tokens, audio_frame in generator:
    if text_tokens is not None:
        # 解码当前累积的文本（可选）
        decoded = tokenizer.decode(text_tokens[0].tolist(), skip_special_tokens=True)
        print(f"\r生成中: {decoded}", end="", flush=True)
    if audio_frame is not None:
        # audio_frame: list of 8 ints（8 个 codebook 当前时间步的 code）
        # 每个 int 范围为 [0, 2048)，表示 Mimi 编码的音频 token
        # 需使用 Mimi 解码器（即对应的 audio codec decoder）还原为 24kHz 波形
        pass

print("\n生成完成")

# 非流式模式：返回最终的 (text_tokens, audio_codes) 元组
# 注意：非流式模式下 stream=False，generate 不返回 generator
# 如需一次性获取完整结果：stream=False
result = model.generate(
    input_ids,
    max_new_tokens=256,
    stream=False,
    return_audio_codes=True,
)
# result 是 generator 最后一个 yield 值
final_text, final_audio_frame = result

# 解码音频：audio_codes 形状为 (8, frame_count)
# 需使用对应的 Mimi 音频编解码器将 code 还原为 24kHz 波形
```

### Omni 语音输入 (ASR)

Omni 也支持语音输入 + 文本输出：

```python
import torchaudio
from versper.dataset.omni_dataset import OmniDataset  # 可参考其音频处理逻辑

# 加载音频文件
waveform, sr = torchaudio.load("speech.wav")
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

# 使用模型的 audio_processor 提取特征
audio_features = model.audio_processor(
    waveform.squeeze(0).numpy(),
    sampling_rate=16000,
)

# 构造带音频占位的 input_ids
prompt = ""  # 或保留 prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
# 插入 audio 占位符
audio_marker_id = ocfg.audio_ids[0]  # 默认为 16
# ...（参照 OmniDataset 的数据构建方式）

# 前向传播
out = model.forward(
    input_ids=...,  # 需构造多流输入
    audio_inputs=audio_features.input_features,
    audio_lens=audio_features.attention_mask.sum(-1),
)
```

> **提示**：Omni 的语音输入处理较为复杂，建议直接参考 `src/versper/dataset/omni_dataset.py` 中的数据构造逻辑。

---

## 4. 加载预训练权重

### 从单文件 checkpoint 加载

```python
import torch

# LM / VLM 权重
state_dict = torch.load("./model/pretrain_768.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)

# Omni 权重（含 Thinker + Talker 及 Projector）
state_dict = torch.load("./model/sft_omni_768.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)
```

> 使用 `strict=False` 的原因是：LM 权重中不包含编码器/Projector/Talker 的键，而这些键在 VLM/Omni 模型中是必需的。设置为 `False` 可跳过缺失键的检查。

### 从 HuggingFace Hub 直接加载（推荐）

```python
from versper.config import MiniMindConfig
from versper.model import MiniMindForCausalLM

# 直接从 Hub 加载（需要 transformers>=4.50.0）
# 注意：当前版本尚未上传至 HF Hub，此方式暂不可用
# model = MiniMindForCausalLM.from_pretrained("jyaogong/minimind")
```

---

## 5. 生成参数指南

`model.generate()` 支持以下常用参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_new_tokens` | int | 8192 (LM) / 1024 (Omni) | 最多生成的新 token 数 |
| `temperature` | float | 0.85 (LM) / 0.75 (Omni) | 采样温度。越高越随机，越低越确定 |
| `top_p` | float | 0.85 (LM) / 0.90 (Omni) | Nucleus sampling 累积概率阈值 |
| `top_k` | int | 50 | 仅从概率最高的 k 个 token 中采样 |
| `repetition_penalty` / `rp` | float | 1.0 | 重复惩罚。>1.0 降低重复概率 |
| `do_sample` | bool | True | 是否使用随机采样（否则贪心解码） |
| `eos_token_id` | int | 2 | 结束符 token ID |
| `use_cache` | bool | True | 是否使用 KV cache（加速生成） |
| `stream` (Omni) | bool | False | 是否流式输出（Omni 独有） |
| `return_audio_codes` (Omni) | bool | False | 是否返回音频编码（Omni 独有） |

### Temperature 与 Top-p 调参建议

| 任务类型 | temperature | top_p | top_k | 说明 |
|----------|-------------|-------|-------|------|
| 创意写作/对话 | 0.85 | 0.90 | 50 | 兼顾多样性与连贯性 |
| 事实问答/摘要 | 0.3 | 0.80 | 40 | 降低随机性，减少幻觉 |
| 代码生成 | 0.2 | 0.70 | 30 | 追求确定性输出 |
| 翻译 | 0.1 | 0.60 | 20 | 力求精确 |
| Omni 语音生成 | 0.75 | 0.90 | 50 | Omni 默认参数 |

### 重复惩罚调参

- `repetition_penalty=1.0`: 不惩罚（默认）
- `repetition_penalty=1.05`: 轻微惩罚，适用于对话
- `repetition_penalty=1.15`: 较强惩罚，适用于长文本生成
- `repetition_penalty=1.2+`: 强惩罚，可能导致语义不连贯

---

## 6. 完整推理脚本参考

项目自带的 `web_demo.py` 是最简洁的推理参考实现：

```bash
cat src/versper/scripts/web_demo.py
```

核心流程始终为：

```
加载配置 → 实例化模型 → 加载权重 → model.cuda().eval() → tokenizer 编码 → model.generate → tokenizer 解码
```

三种模型的差异点总结：

| 方面 | MiniMind (LM) | MiniMind-V (VLM) | MiniMind-O (Omni) |
|------|---------------|------------------|--------------------|
| 配置类 | `MiniMindConfig` | `VLMConfig` | `OmniConfig` |
| 模型类 | `MiniMindForCausalLM` | `MiniMindVLM` | `MiniMindOmni` |
| 额外输入 | — | `pixel_values` | `pixel_values`, `audio_inputs`, `spk_emb` |
| 输出 | 文本 tokens | 文本 tokens | 文本 tokens + 音频 codes |
| 流式支持 | 否（内置 generate） | 否（内置 generate） | 是（`stream=True`） |
| 权重文件 | `pretrain_768.pth` | `pretrain_768.pth` | `sft_omni_768.pth` |
