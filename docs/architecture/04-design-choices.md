# 关键设计决策分析

> 为什么这样设计？面试中如何回答？

## 1. 中层桥接 (Middle-Layer Bridge)

### 问题
Talker 应该读取 Thinker 哪一层的 hidden state？

### 实验发现

| 桥接位置 | 表现 | 原因 |
|---------|------|------|
| Embedding (Layer 0) | ❌ 发音模糊 | 只有 token 身份信息，没有上下文 |
| Layer 3 (中层的) | ✅ 最佳 CER | 有足够上下文，未被 LM head 污染 |
| Layer 7 (最后层) | ❌ CER 上升 | 已被 next-token 分类器过度塑造 |

### 具体例子：汉字"地"

汉字 **"地"** 可以有多种发音：
- **de**: 轻轻地、慢慢地（助词）
- **di**: 地球、地图（名词）

```
Embedding 层:    只知道看到"地"这个 token
Layer 3 (中层):  看到了上下文"轻轻__" — 知道该读 de
Layer 7 (最后层): 已高度聚焦于下一个 token 的分类 — 掺杂了 LM head 的几何噪声
```

### 面试回答模板

> **Q: 为什么选择中间层作为桥接？**
>
> 这是一个 trade-off。Embedding 层太浅，没有上下文信息，无法消解发音歧义；最后一层被 next-token 分类器过度塑造，hidden state 的主要变化方向是区分不同输出 token，这对声学生成来说是噪声。中间层积累了足够的句法和跨模态上下文，但尚未被语言模型头主导，是最干净的声学条件信号。

---

## 2. 低秩 Codebook 接口

### 问题
8 个 codebook 的 embedding 和 output head 应该怎么做？

### 方案对比

```
方案 A: 8 个完全独立（简单直接）
  参数: 8 × vocab × hidden + 8 × hidden × vocab
  问题: 参数量太大 (约 26M 参数，在 0.1B 模型中不可接受)

方案 B: 共享底座 + 低秩 adapter（MiniMind-O 采用）
  Embedding: SharedBase(codebook_all) + ∑ rank_adapters
  Head:      SharedBase(codebook_all) + ∑ rank_adapters
  特点: 用少量 adapter 参数实现 codebook 特异性

方案 C: 完全共享 1 份（太粗暴）
  问题: 无法区分不同 codebook 的分布差异
```

### Rank 消融实验结果

| Rank | Audio Loss | Codebook Accuracy | 完整性 |
|------|-----------|------------------|--------|
| 16 | 较高 | 较低 | 欠适配 |
| 64 | 中等 | 中等 | — |
| 256 | 接近 full rank | 接近 full rank | ✅ 选用的配置 |
| Full (2112) | 最优 | 最优 | 参数浪费 |

**解耦实验更重要**：

| 实验 | Embedding rank | Head rank | 效果 |
|------|---------------|-----------|------|
| 变 Head rank | 固定 16 | 16→256 | ✅ **大幅提升** |
| 变 Emb rank | 16→256 | 固定 16 | △ 提升较小 |

**结论**：Head rank 比 Embedding rank 重要，因为 head 需要区分 8 个离散分布，而 embedding 主要是读取近期的共享 codec 历史。

### 面试回答模板

> **Q: 为什么用低秩接口而不是 8 个独立的 codebook 层？**
>
> 这是一个参数效率问题。8 个 codebook 共享同一段音频的不同量化残差，所以它们的统计结构高度相关。我们使用共享底座 + 低秩 adapter 的方式，让底座捕获共性，adapter 学习差异。实验表明，中等 rank（256）就能恢复大部分 full-rank 的收益。更重要的是，我们发现输出端的 rank 比输入端的 rank 更关键，因为输出需要区分 8 个不同的概率分布，而输入主要是读取共享的 codec 上下文。

---

## 3. 说话人控制策略

### 设计选择：In-Context Conditioning

**不做**：独立的 TTS 模块 + 固定的说话人 embedding
**做**：通过参考音频的 codec prompt + CAM++ embedding 实现说话人控制

### 控制信号的放置

```
序列中对齐方式:

         ... [pad] [<|audio_spk|>] [ref_code_0] [ref_code_1] ... [target codes]
                  ├── 8 layers ──┤  ├── 右对齐参考区 ──┤      ├── 目标 ──┤
                  │              │  │  (loss masked)  │      │ (有loss) │
                  │  CAM++ 192   │
                  │  → projected │
                  │  → 替换所有 8 层│
```

### 内置声音
| 类型 | 声音名称 | 数量 |
|------|---------|------|
| 内置（seen） | dylan, eric, serena, uncle_fu, vivian | 5 |
| 保留（unseen） | arthur, chelsie, cherry, ethan, jennifer, momo, moon | 7 |

### 推理时更换声音
只需要更改：`voices.pt` 中的参考 Mimi codes + CAM++ 向量
不需要更改：Thinker prompt、Talker weights

### 面试回答模板

> **Q: 说话人控制为什么用参考提示而不是独立的 TTS 模块？**
>
> 这保持了端到端的可追溯性。如果说话人控制是独立的 TTS 模块，发音错误无法回溯到共享表示。我们通过将参考 codec 提示和 CAM++ 声纹嵌入直接放在 Talker 的输入上下文中，让说话人音色成为音频编码上下文的一部分，而不是独立的条件输入。推理时更换说话人只需要换参考音频，不需要改模型权重。

---

## 4. Thinker-Talker 分离度

### 设计选择
Talker 从 Thinker copy 初始化，但训练时独立更新。

### 为什么不共享参数？
| 因素 | Thinker | Talker |
|------|---------|--------|
| 输入 | 文本 embedding + 多模态特征 | Mimi code embeddings |
| 注意力范围 | 全部上下文 | 近期 codec 帧 |
| 输出分布 | 6400 类文本 token | 8 × 2112 类 audio code |
| 功能目标 | 语义正确 | 发音流畅 |

共享参数会迫使单个 Transformer 同时优化两个不同的目标函数，实验上会导致性能下降。

### 为什么 Talker 只有 4 层？
Talker 的消融实验发现：

| Talker hidden | Avg CER | 说明 |
|-------------|---------|------|
| 768（默认） | 0.0897 | ✅ 最优 |
| 512 | 0.1745 | ❌ CER 翻倍 |
| 384 | 0.2767 | ❌ 严重退化 |

Talker 不能因为语义来自 Thinker 就被做得太薄。8 层的 codebook 预测本身就是个困难任务。

---

## 5. 模态 Projector 设计

### 设计选择：轻量 MLP（2 层）

```python
# MMAudioProjector
LayerNorm(encoder_dim) → Linear → GELU → Linear(hidden_dim)

# MMVisionProjector  
LayerNorm(encoder_dim) → Linear → GELU → Linear(hidden_dim)
```

**为什么这么简单？**
- 编码器（SenseVoice / SigLIP2）已经提供高质量特征
- Projector 只需要做**维度映射 + 非线性变换**
- 复杂的设计（Q-Former 等）在小规模下收益有限
- 保持最小配方（minimal recipe）的设计哲学

### 面试回答模板
> **Q: 为什么不用 Q-Former 或 Perceiver Resampler 作为多模态桥接？**
>
> Q-Former 这类复杂桥接器在 0.1B 规模下带来的额外参数量和训练难度超过了其收益。在中小规模下，感知任务应该由预训练编码器承担，桥接器只需要做维度对齐和轻量变换。这也是一致的最小配方设计哲学：只有那些被消融实验证明不可或缺的组件才应该保留。

---

## 6. 评估策略

### 为什么用一致性评估而不是 MOS？
| 指标 | 优点 | 缺点 |
|------|------|------|
| MOS（主观听感） | 最自然的评估 | 成本高、不可复现、不稳定 |
| CER/WER（一致性） | 自动、可复现、诊断性强 | 不评估自然度、对 ASR 有依赖 |
| CAM++ 相似度 | 自动评估说话人保持 | 对合成音频质量敏感 |

### 面试回答模板
> **Q: CER 和 WER 是 ASR 评估指标，为什么用来评估语音生成模型？**
>
> 我们不把它当作语音识别指标，而是**一致性的量化**。如果 Talker 生成的语音和 Thinker 写的文本不一致（遗漏词、读错词），ASR 转录后的 edit distance 就会扩大。这让我们能诊断出 Thinker-Talker 之间的不一致。当然，ASR 自身的错误（如数字格式）会导致假阳性，所以我们同时保留了人工检查和自然度评估作为补充。
