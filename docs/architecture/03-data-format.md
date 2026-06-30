# 序列格式与流式解码

> 多模态数据的组织方式是 Omni 模型训练的核心

## 1. 九流序列布局 (9-Stream Sequence)

每个训练样本是一个 **9 条并行流**的序列：8 条音频 codec 流 + 1 条文本流。

```
时间步 →        t0    t1    t2    t3    t4    t5    t6    t7    t8 ...

Text Stream:   [usr] [txt] [pad] [sys] [txt] [pad] [ast] [txt] [txt] ...
Audio Str 0:   [pad] [pad] [pad] [pad] [pad] [spk] [ref] [a0]  [a1]  ...
Audio Str 1:   [pad] [pad] [pad] [pad] [pad] [spk] [ref] [pad] [a0]  ...
Audio Str 2:   [pad] [pad] [pad] [pad] [pad] [spk] [ref] [pad] [pad] ...
...  (以此类推，共 8 层)
Audio Str 7:   [pad] [pad] [pad] [pad] [pad] [spk] [ref] [pad] [pad] ...

               ───  User / System  ───    ───  Assistant  ───
               ── 无损失  ──              ──  参考区域   ──  ── 目标区域 ──
                                          (loss masked)   (有音频监督)
```

### 占位符 Token

| Token | 位置 | 含义 |
|-------|------|------|
| `<|audio_pad|>` | Thinker 输入 | 将被 SenseVoice 投影特征替换 |
| `<|image_pad|>` | Thinker 输入 | 将被 SigLIP2 投影特征替换 |
| `<|audio_spk|>` | Talker 输入 | 将被 CAM++ 说话人嵌入替换（8 层全填） |
| `<|audio_ref|>` | Talker 输入 | 参考音频 codec 提示（loss masked） |

### 区域划分

| 区域 | Loss | 说明 |
|------|------|------|
| User/System 部分 | ❌ Masked | 对话上下文 |
| 参考音频区域 | ❌ Masked | 说话人参考提示 |
| 目标音频区域 | ✅ CE Loss | 8 层 codebook 均有监督信号 |
| 文本响应区域 | ✅ CE Loss | 标准的 next-token prediction |

## 2. 损失函数

$$
\mathcal{L} = \mathcal{L}_{\text{text}} + \lambda_{\text{audio}} \sum_{q=1}^{8} \mathcal{L}_{\text{audio}}^{(q)}
$$

- $\mathcal{L}_{\text{text}}$: 文本 next-token prediction 交叉熵损失
- $\mathcal{L}_{\text{audio}}^{(q)}$: 第 q 层 codebook 的交叉熵损失
- $\lambda_{\text{audio}}$: 音频损失的权重系数
- 无效位置和 conditioning 位置均被 mask

## 3. 延时对齐策略 (Staggered Targets)

Mimi codec 有 8 层，但音频生成需要按层逐级解码。因此目标音频在不同层之间有**延时偏移**：

```
Assistant 开始位置 = t_ast

Layer 0: t_ast + 1  开始有监督
Layer 1: t_ast + 2  开始有监督
Layer 2: t_ast + 3  开始有监督
...
Layer 7: t_ast + 8  开始有监督

即在 t_ast + q + 1 位置，第 q 层开始有监督信号
```

**为什么这样设计？**
- 第一帧语音需要等文本先输出一个 token
- 每层 codebook 需要前一层的输出才能解码
- 实现真正的流式：文本生成开始后，音频逐步可用

## 4. 流式推理

### 推理时序

```
时间 ───────────────────────────────────────────────►

Thinker Prefill:  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░
  (处理输入，生成文本)

Thinker Decode:   ░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  (逐 token 生成文本)

Talker Gen:       ░░░░░░████▒▒▒▒████▒▒▒▒████▒▒▒▒░░░░
  (逐帧生成 8 层 codec codes)

Mimi Decode:      ░░░░░░░░░░▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░
  (解码为 24kHz 波形)

用户听到音频:     ░░░░░░░░░░░░♫♫♫♫♫♫♫♫♫♫♫♫♫♫♫♫░░░░
```

### 关键点
1. Thinker 先完成语义 prefill（处理输入、生成首个文本 token）
2. 第一个文本 token 生成后，Talker 才开始生成第 0 层 codec
3. 当完整 8 层一帧就绪，Mimi 解码器立即输出波形
4. **用户可以在文本生成完成前就开始听到语音**
5. 支持 **barge-in**（打断）：VAD 检测到用户说话，立即停止生成

## 5. 训练数据集

| 数据集 | 样本数 | 输入 | 输出 | 总音频时长 | 用途 |
|--------|--------|------|------|-----------|------|
| sft_t2a | 1,248,923 | 文本指令 | 语音回答 | 1,636.01 h | 文本→语音对齐 |
| sft_i2t | ~100,000 | 图像+文本 | 文本回答 | - | 视觉→语言对齐 |
| sft_a2a | 414,024 | 语音+文本 | 语音回答 | 2,135.37 h | 语音→语音对话 |

### 语言分布

| 分割 | 中文 | 英文 | 混合 |
|------|------|------|------|
| T2A | 45.7% | 46.5% | 7.8% |
| A2A | 70.8% | 21.2% | 8.0% |

> 英文长回答是最容易出问题的场景（发音漂移、词汇遗漏）

## 6. 评估方法

### 一致性评估 (Consistency Evaluation)

核心思路：**Talker 生成的语音是否和 Thinker 写的文本一致？**

```
Prompt → Thinker → 文本 T
                 → Talker → 语音 A → ASR → 转录文本 T'

CER/WER = edit_distance(T, T') / len(T)
```

- **内一致性**: 使用 Qwen3-ASR-Flash 转录，与模型自己生成的文本比较
- **跨模型**: 与其他模型（Mini-Omni, Mini-Omni2）在相同 prompt 下比较
- **注意**: 数字格式（如 299,792,458）可能因 ASR 的格式差异导致 CER 虚高

### 说话人相似度
CAM++ 余弦相似度，比较生成语音和参考语音的声纹嵌入。

## 7. 面试要点

### 问：为什么需要 8 个 codebook 而不是 1 个？
Mimi codec 使用残差向量量化（RVQ），第一层量化主要残差，后续层逐步量化更精细的残差。8 层可以从粗糙到精细重建音频，用更少的 token 数实现高质量音频。

### 问：流式解码具体如何工作？
文本先输出→开始解码第一层 codebook→逐层解码→完整一帧后交给 Mimi 解码器→输出波形片段。每个时间步都有新音频帧可用，实现低延迟的播放体验。

### 问：为什么参考音频区域要 mask loss？
参考音频的作用是提供**说话人音色提示**，而不是让模型去重建它。如果对它计算 loss，模型会浪费容量去记住参考音频的精确 codec 序列，而不是学习从语义到语音的映射。
