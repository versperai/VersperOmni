# Thinker–Talker 机制详解

> 核心架构：语义路径与声学路径的分离与融合

## 1. 设计动机

在全模态模型中，文本生成和语音生成有本质不同的需求：

| 维度 | 文本生成（Thinker） | 语音生成（Talker） |
|------|-------------------|-------------------|
| 目标 | 语义准确、推理正确 | 发音清晰、自然流畅 |
| 表征 | 离散 token | 离散 codec codes |
| 上下文 | 整个对话历史 | 最近的 codec 历史 |
| 输出 | 自回归文本 token | 8 层 codebook logits |

如果用一个模型同时做这两件事，两者会互相干扰。Thinker-Talker 分离方案让**语义路径**和**声学路径**各自独立，仅在桥接层交互。

## 2. Thinker（思考器）架构

```
                    Thinker — MiniMind Transformer
                    ┌───────────────────────────────────────┐
                    │  Embedding (vocab 6400 → 768)        │
                    │         ↓                            │
                    │  Layer 0: Attention + FFN             │
                    │         ↓                            │
                    │  Layer 1: Attention + FFN             │
                    │         ↓                            │
                    │  Layer 2: Attention + FFN             │
                    │         ↓                            │
        ┌───────────┤  Layer 3: Attention + FFN  ←── bridge layer
        │           │         ↓                            │
        │           │  Layer 4: Attention + FFN             │
        │           │         ↓                            │
        │           │  Layer 5: Attention + FFN             │
        │           │         ↓                            │
        │           │  Layer 6: Attention + FFN             │
        │           │         ↓                            │
        │           │  Layer 7: Attention + FFN             │
        │           │         ↓                            │
        │           │  RMSNorm → LM Head (vocab 6400)      │
        │           └───────────────────────────────────────┘
        │                            │
        │              ┌─────────────┘
        ▼              ▼
   ┌────────┐   ┌────────────┐
   │Text    │   │Bridge State│  ← Layer 3 的 hidden state
   │Logits  │   │(向 Talker) │
   └────────┘   └────────────┘
```

### Thinker 配置
- **层数**: 8
- **Hidden size**: 768
- **Attention heads**: 8 query heads / 4 KV heads (GQA)
- **词表**: 6400
- **参数量**: Dense 63.91M / MoE 198.42M
- **位置编码**: RoPE（Rotary Position Embedding）

## 3. Talker（说话器）架构

```
                    Talker — 独立 4 层模块
                    ┌───────────────────────────────────────┐
                    │  TalkerEmbedding (低秩, 见下文)         │
                    │         ↓                            │
                    │  ┌─ Fusion: bridge + codec ──┐       │
                    │  │  embed_proj(bridge) × α     │       │
                    │  │  + codec_proj(emb) × β     │       │
                    │  └─────────────────────────────┘       │
                    │         ↓                            │
                    │  Block 0: MiniMind Attention + FFN     │
                    │         ↓                            │
                    │  Block 1: MiniMind Attention + FFN     │
                    │         ↓                            │
                    │  Block 2: MiniMind Attention + FFN     │
                    │         ↓                            │
                    │  Block 3: MiniMind Attention + FFN     │
                    │         ↓                            │
                    │  RMSNorm                              │
                    │         ↓                            │
                    │  TalkerHead (低秩, 8 codebook heads)  │
                    │         ↓                            │
                    │  8 × codebook logits (vocab 2112)    │
                    └───────────────────────────────────────┘
```

### Talker 配置
- **层数**: 4（MiniMind blocks，从 Thinker 最后 4 层初始化）
- **Hidden size**: 768（ablated: 512/384 退化明显）
- **音频词表**: 2112（Mimi codec vocab 2048 + special tokens）
- **Codebook heads**: 8
- **参数量**: Dense 47.05M / MoE 114.30M

### 低秩接口 (TalkerEmbedding & TalkerHead)

```
传统方案（不采用）:
  8 个独立 embedding tables:  8 × vocab × hidden = 8 × 2112 × 768 = 12.97M
  8 个独立 output heads:     8 × hidden × vocab = 8 × 768 × 2112 = 12.97M

低秩方案（实际采用）:
  Shared base: 1 × vocab × hidden
  8 × low-rank adapters: 8 × (vocab × rank + rank × hidden)  

  以 rank=256 为例:
  Embedding: base(2112×768) + 8×[down(2112×256) + up(256×768)] = ~8.4M
  Head:      base(768×2112) + 8×[down(768×256) + up(256×2112)] = ~8.4M
```

**为什么输出头的 rank 比输入 embedding 的 rank 更重要？**
- Embedding: 主要读取近期 codec 历史，共享信息多
- Head: 需要区分 8 个 codebook 各自在完整音频词表上的分布
- 实验：Head rank 16→256 的收益 > Embedding rank 同样提升

## 4. 桥接机制 (Bridge)

### 桥接层选择
MiniMind-O 使用 `num_hidden_layers // 2 - 1 = 3`（第 3 层，从 0 开始索引）。

选择依据：
```
Layer 0 (Embedding): 
  ❌ 只包含 token 身份信息 + 多模态特征
  ❌ 没有积累足够的上下文（无法解决 地 → de/di 的发音歧义）

Layer 3 (Middle):
  ✅ 积累了足够的句法和跨模态上下文
  ✅ 尚未被 next-token 分类器过度塑造
  ✅ 包含发音歧义消解所需的信息

Layer 7 (Last):
  ❌ 已被 LM head 的几何结构强烈影响
  ❌ 带入了 token 选择噪声，不利于声学条件
  ❌ CER 指标上表现最差
```

### Embedding 与 Codec 的融合
Talker 每个位置接收两个输入流的和：

```python
# 伪代码
bridge_proj = embed_proj(thinker_hidden[layer_3])        # (B, T, 768)
codec_emb = codec_proj(talker_embedding(audio_tokens))    # (B, T, 768)
talker_input = bridge_proj * text_scale + codec_emb * audio_scale
# text_scale 和 audio_scale 均为可学习的标量参数
```

## 5. 面试要点

### 问：为什么要分 Thinker 和 Talker，而不是一个统一的模型？
- **目标冲突**：文本生成需要高层次的语义抽象，语音生成需要低层次的声学细节
- **历史长度不同**：文本需要整个对话上下文，语音只需近期 codec 帧
- **初始化知识**：Talker 可以从 Thinker 拷贝初始化（inductive bias 共享）
- **流式可行**：Thinker 先做完语义 prefill，Talker 再逐帧生成音频

### 问：Model Soufflé（model merging）和 MiniMind-O 的 Thinker-Talker 初始化有什么关系？
Talker 初始化是从 Thinker 最后 4 层拷贝权重（当 hidden size 匹配时）。这利用了 MiniMind 语言模型预训练获得的 Transformer 表示，使 Talker 不需要从零学习语义理解，只需在此基础上学习音频生成特有的模式。

### 问：低秩接口为什么有效？
8 个 codebook 共享大量统计信息（都建模同一段音频的不同量化残差），共享底座可以让它们共享这些统计信息，而低秩 adapter 只学习 codebook 特有的偏移。这是一种典型的**参数高效微调（PEFT）**思路在架构层面的应用。
