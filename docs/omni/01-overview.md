# VersperOmni 整体架构总览

> 基于 MiniMind-O Technical Report 整理，适用于学习和面试准备

## 1. 项目定位

VersperOmni 是一个**开源的小规模语音原生全模态模型**（Omni Model），核心思想源自 MiniMind-O。在**约 0.1B 活跃参数**的规模下，实现文本、语音、图像三种输入，以及文本 + 流式语音输出。

### 关键特点

| 特点 | 说明 |
|------|------|
| **参数规模** | 约 1 亿活跃参数（Dense: 115M / MoE: 115M active） |
| **输入模态** | Text + Speech + Image |
| **输出模态** | Text + Streaming Speech（24kHz 波形） |
| **训练成本** | 4× RTX 3090，单变体约 4 小时 |
| **开源程度** | 模型代码 + 权重 + Parquet 训练数据集全开源 |

## 2. 核心架构：Thinker–Talker

```
                         ┌─────────────────────┐
  Text ────────────────► │                     │
                         │      Thinker        │
  Audio ──► SenseVoice ──┤   (MiniMind 8层)    │
          (frozen)       │                     │
                         │  hidden[bridge]     ├────► Text Output
  Image ──► SigLIP2 ────┤                     │
           (frozen)      │  layer 3 (中层桥接)  │
                         └────────┬────────────┘
                                  │ bridge state
                                  ▼
                         ┌─────────────────────┐
                         │       Talker         │
  Codec History ────────►│   (4层 MiniMind)     │
                         │                     ├────► Mimi Codes ──► 24kHz Audio
  Speaker Ref ──────────►│  fusion: bridge +    │
  CAM++ ────────────────►│  codec history       │
                         └─────────────────────┘
```

### Thinker（思考器）
- 完整的 **MiniMind Transformer**：8 层、hidden 768、8 query heads、4 KV heads
- 词表大小 6400（轻量级 tokenizer）
- 负责语义理解、推理、文本生成
- **Dense 版**：63.91M 参数 / **MoE 版**：198.42M 总参数（活跃 ~同 Dense）

### Talker（说话器）
- 独立的 **4 层 MiniMind blocks**
- hidden 768，audio vocab 2112，8 codebook heads
- **rank-256 低秩 adapter**（输入输出各 8 份，共享底座）
- **Dense 版**：47.05M 参数 / **MoE 版**：114.30M 总参数
- 初始化：从 Thinker 最后 4 层拷贝权重（维度匹配时）

## 3. 输入处理管线

### 语音输入
| 模块 | 具体模型 | 参数 |
|------|---------|------|
| 编码器 | SenseVoice-Small | 50 encoder blocks，output 512 dim (frozen) |
| 投影器 | MMAudioProjector | LayerNorm(512) → Linear → GELU → Linear(768) |
| 前端 | 16kHz 音频特征 | — |

### 图像输入
| 模块 | 具体模型 | 参数 |
|------|---------|------|
| 编码器 | SigLIP2 base patch32-256 | 12 层 ViT，hidden 768 (frozen) |
| 投影器 | MMVisionProjector | LayerNorm(768) → Linear → GELU → Linear(768) |
| 图像 token | 64 个 placeholder 位置 | — |

### 语音输出（Codec）
| 模块 | 具体模型 | 参数 |
|------|---------|------|
| Codec | Mimi | 8 codebooks, vocab 2048, 12.5 Hz frames (frozen) |
| 解码输出 | 24kHz 波形 | 流式逐帧解码 |

### 说话人控制
| 模块 | 具体模型 | 参数 |
|------|---------|------|
| 声纹嵌入 | CAM++ | 192-dim 向量 (precomputed) |
| 参考提示 | Ref Mimi codes | 右对齐放置在目标音频之前 |

## 4. 数据流概要

1. **Text**: 直接通过 token embedding table → Thinker
2. **Audio**: SenseVoice 编码 → MLP 投影 → 替换 `<|audio_pad|>` 位置
3. **Image**: SigLIP2 编码 → MLP 投影 → 替换 `<|image_pad|>` 位置
4. **Speaker**: CAM++ 向量投影后填入 `<|audio_spk|>` 位置的所有 8 层
5. **Bridge**: Thinker 第 3 层 hidden state → embed_proj → Talker
6. **Talker fusion**: embed_proj(bridge) × text_scale + codec_proj(history) × audio_scale
7. **Output**: Talker → 8 个 codebook heads → Mimi 解码 → 24kHz waveform

## 5. 模型配置对比

| 配置项 | Dense | MoE |
|--------|-------|-----|
| Thinker 参数 | 63.91M | 198.42M (active ~63.91M) |
| Talker 参数 | 47.05M | 114.30M (active ~47.05M) |
| Audio 投影器 | 0.99M | 0.99M |
| Vision 投影器 | 1.18M | 1.18M |
| **总计活跃** | **~115.29M** | **~115.33M** |
| Avg CER ↓ | 0.0897 | 0.0900 |
| Voice Clone Overall ↑ | 0.5995 | 0.5937 |

## 6. 面试要点

### 问：为什么叫 Omni Model？
全模态模型（Omni Model）指一个模型能同时处理和理解多种模态（文本、语音、图像），并能以多种模态输出。VersperOmni 做到了 text→text, text→speech, speech→speech, image→speech 的完整交互闭环。

### 问：0.1B 这个规模有什么特殊之处？
- 可以在消费级 GPU（如 RTX 3090）上完整复现训练流程（约 4 小时）
- 使得全模态研究的门槛大幅降低
- 暴露了大规模下被掩盖的设计权衡（如 bridge layer 位置、低秩接口的 rank 选择）

### 问：Thinker-Talker 和传统的 Cascade 方案有什么区别？
- **Cascade 方案**：ASR → LLM → TTS，LLM 处于声学环路之外
- **Thinker-Talker**：Talker 共享 Thinker 的语义 hidden state，实现语义-声学联合建模
- 优势：错误可追溯、端到端优化、支持流式交互
