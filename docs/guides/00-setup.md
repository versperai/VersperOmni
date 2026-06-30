# 环境搭建与安装指南

> 本文将指导您完成 VersperOmni 的本地开发/推理环境搭建，涵盖 MiniMind（文本 LM）、MiniMind-V（视觉语言模型）和 MiniMind-O（全模态模型）的安装与验证。

---

## 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Linux (Ubuntu 22.04+) | Linux (Ubuntu 22.04+) |
| Python | >= 3.12 | 3.12+ |
| GPU | CUDA-capable GPU (8 GB+) | CUDA-capable GPU (16 GB+, e.g. RTX 3090/4090) |
| 内存 | 16 GB | 32 GB+ |
| 磁盘 | 10 GB 可用空间 | 30 GB+ (含数据集和权重) |

**依赖项概览**（详见 `pyproject.toml`）：

- `torch>=2.12.0` — 深度学习框架
- `transformers>=4.50.0` — HuggingFace 模型/分词器工具
- `numpy`, `scipy`, `soundfile`, `librosa` — 音频/数值处理
- `Pillow` — 图像处理
- `sentencepiece` — 分词器后端
- `datasets`, `pyarrow` — 数据集加载

---

## 环境搭建

### 方式一：uv（推荐）

[uv](https://docs.astral.sh/uv/) 是 Rust 编写的极速 Python 包管理器，建议作为首选工具。

```bash
# 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 Python 3.12
uv python install 3.12

# 在项目根目录创建虚拟环境
uv venv
source .venv/bin/activate

# 安装核心依赖（含项目自身）
uv pip install -e .

# 可选：按需安装额外依赖组
uv pip install -e ".[omni]"     # Omni 模型（funasr, onnxruntime）
uv pip install -e ".[vlm]"      # VLM 模型（当前预留，无额外依赖）
uv pip install -e ".[train]"    # 训练（swanlab 实验追踪）
uv pip install -e ".[all]"      # 以上全部
```

### 方式二：conda

```bash
conda create -n versperomni python=3.12
conda activate versperomni
pip install -e .
pip install -e ".[all]"
```

---

## 验证安装

创建 Python 脚本或直接在交互式环境中运行以下代码，确认模型可正常实例化：

```python
from versper.config import MiniMindConfig
from versper.model import MiniMindForCausalLM

cfg = MiniMindConfig()
model = MiniMindForCausalLM(cfg)
print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
```

预期输出：

```
Model created: 105.86M params
```

如需同时验证 VLM 和 Omni 模型：

```python
from versper.config import VLMConfig, OmniConfig
from versper.vlm import MiniMindVLM
from versper.omni import MiniMindOmni

# VLM（不含视觉编码器，因未指定模型路径）
vcfg = VLMConfig()
vlm = MiniMindVLM(vcfg, vision_model_path=None)
print(f"VLM params: {sum(p.numel() for p in vlm.parameters())/1e6:.2f}M")

# Omni（不含编码器）
ocfg = OmniConfig()
omni = MiniMindOmni(ocfg, audio_encoder_path=None, vision_model_path=None)
print(f"Omni params: {sum(p.numel() for p in omni.parameters())/1e6:.2f}M")
```

---

## 模型权重下载与目录结构

### 下载地址

| 文件 | HuggingFace 源 |
|------|---------------|
| MiniMind 分词器 | [jyaogong/minimind](https://huggingface.co/jyaogong/minimind) |
| 预训练权重 (`pretrain_768.pth`) | [jyaogong/minimind](https://huggingface.co/jyaogong/minimind) |
| Omni SFT 权重 (`sft_omni_768.pth`) | [jyaogong/minimind-o](https://huggingface.co/jyaogong/minimind-o) |
| SenseVoice 音频编码器 | [iic/SenseVoiceSmall](https://huggingface.co/iic/SenseVoiceSmall) |
| SigLIP2 视觉编码器 | [google/siglip2-base-patch16-256](https://huggingface.co/google/siglip2-base-patch16-256) |

> 注意：MiniMind-O 中实际使用的视觉编码器是 SigLIP2 base-p32-256-ve 变体（patch size 32，输出 256 维）。如果 HuggingFace 上找不到完全一致的路径，可下载 `siglip2-base-patch16-256` 后自行配置，或使用 `vision_model_path=None` 跳过视觉编码器加载。

### 期望目录结构

```
model/
├── tokenizer.model                  # MiniMind sentencepiece 分词器
├── tokenizer_config.json            # HuggingFace 分词器配置
├── special_tokens_map.json          # 特殊 token 映射
├── pretrain_768.pth                 # MiniMind 预训练权重（768 hidden, 8层）
├── sft_omni_768.pth                 # Omni SFT 权重（含 Thinker + Talker）
├── SenseVoiceSmall/                 # SenseVoice 音频编码器（funasr 格式）
│   ├── model.pt
│   ├── config.yaml
│   └── ...
└── siglip2-base-p32-256-ve/         # SigLIP2 视觉编码器（HuggingFace 格式）
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    └── ...
```

加载权重示例：

```python
import torch
from versper.config import MiniMindConfig
from versper.model import MiniMindForCausalLM

cfg = MiniMindConfig()
model = MiniMindForCausalLM(cfg)
state_dict = torch.load("./model/pretrain_768.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)
print("Weights loaded successfully")
```

---

## 数据集准备

### Omni 训练数据（Parquet 格式）

Omni 的 SFT 数据以 Parquet 文件提供，支持三种模式：

```
dataset/
├── sft_t2a.parquet    # Text-to-Audio（文本输入 → 语音输出）
├── sft_a2a.parquet    # Audio-to-Audio（语音输入 → 语音输出）
├── sft_i2t.parquet    # Image-to-Text（图像输入 → 文本输出）
└── sft_vision.parquet # 纯视觉问答
```

每个 Parquet 文件包含的列由 `OmniDataset` 处理，典型字段包括：

- `input_ids` / `labels` — 文本 token 序列
- `audio_inputs` / `audio_lens` — 音频特征
- `pixel_values` — 图像像素值
- `spk_emb` — 说话人嵌入

### 文本训练数据（JSONL 格式）

```jsonl
# 预训练数据
{"text": "Attention Is All You Need 是 Transformer 架构的原始论文..."}

# SFT 数据
{"conversations": [{"from": "user", "value": "什么是深度学习？"}, {"from": "assistant", "value": "深度学习是机器学习的一个分支..."}]}
```

---

## 常见问题排查

### CUDA Out of Memory

```bash
# 降低推理批次大小（默认 1）
# 使用 torch.cuda.empty_cache() 释放缓存
# 或使用 CPU 推理进行初步调试
model = MiniMindForCausalLM(cfg).eval()  # 不调用 .cuda()
```

### 模型文件缺失或加载失败

- 确认 `model/` 目录在项目根目录下
- 检查文件名大小写（`pretrain_768.pth` vs `Pretrain_768.pth`）
- 使用 `strict=False` 忽略缺失的键（如编码器权重在 Omni 加载时可能不匹配）

```
RuntimeError: Error(s) in loading state_dict for MiniMindForCausalLM:
    Missing key(s) in state_dict: ...
```

解决：`model.load_state_dict(state_dict, strict=False)`

### Import 错误

```
ModuleNotFoundError: No module named 'versper'
```

解决：确保在项目根目录执行 `pip install -e .`，或在运行时添加 `sys.path`：

```python
import sys
sys.path.append(".")  # 或使用绝对路径
```

### omni 依赖安装失败

`funasr` 和 `onnxruntime` 为 Omni 模型的**可选依赖**。如果仅使用文本 LM 或 VLM，无需安装。安装 Omni 依赖时如遇问题，可尝试：

```bash
uv pip install funasr onnxruntime
# 或降级至特定兼容版本
uv pip install "funasr>=1.0.0" "onnxruntime>=1.16.0"
```

### 关于 torchvision

`torchvision` 并非项目直接依赖，但 SigLIP2 的 processor 在处理图像时可能会打印以下警告：

```
The torchvision module is not installed. ...
```

此警告不影响功能，可忽略。如需消除警告：

```bash
uv pip install torchvision
```

---

## 下一步

安装验证完成后，请参阅：

- **[01-inference.md](01-inference.md)** — 三种模型的快速推理示例
- `docs/architecture/` — 架构设计详解
- `docs/training/` — 训练指南与策略
