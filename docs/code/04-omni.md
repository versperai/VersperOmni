# 全模态模型 -- VersperOmni (Thinker-Talker 架构)

## 架构总览

`VersperOmni` (`omni.py:206`) 是 VersperOmni 的全模态模型，采用 **Thinker-Talker** 双模块架构，支持文本、语音、图像多模态输入，以及文本+流式语音输出。

```
VersperForCausalLM
  └── VersperOmni
        ├── thinker (alias of self.model) -- LM 骨干 (8× VersperBlock)
        ├── talker (TalkerModule) -- 语音生成模块 (4× VersperBlock)
        ├── audio_proj (MMAudioProjector) -- 音频特征投影
        ├── vision_proj (MMVisionProjector) -- 视觉特征投影
        ├── audio_encoder (SenseVoice-Small, 冻结)
        └── vision_encoder (SigLIP2, 冻结)
```

### Thinker-Talker 数据流

```
输入: [文本] / [音频] / [图片]
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  text_embed    audio_encoder   vision_encoder
     │              │              │
     │       audio_proj       vision_proj
     │              │              │
     └──────────────┼──────────────┘
                    ▼
         [注入音频/视觉特征]
                    │
         ┌──────────┴──────────┐
         │     Thinker (8层)    │
         │ 第 0..bridge_layer   │
         └──────────┬──────────┘
                    ▼ bridge_states
         ┌──────────┴──────────┐
         │ embed_proj * text_scale + codec_proj * audio_scale
         └──────────┬──────────┘
                    ▼
         ┌──────────┴──────────┐
         │  Talker (4层)       │
         └──────────┬──────────┘
                    ▼
         ┌──────────┴──────────┐
         │ TalkerHead (8头)     │ → 8×Mimi Codebook logits
         └─────────────────────┘
```

---

## TalkerModule -- 语音生成模块

定义在 `omni.py:136`，是一个独立的语音解码器。

### 结构

```python
class TalkerModule(nn.Module):
    def __init__(self, config):
        self.layers = [VersperBlock × num_talker_hidden_layers]  # 默认 4 层
        self.norm = RMSNorm(talker_hidden_size)                   # 输出归一化
        self.lm_head = TalkerHead(talker_hidden_size, 2112)       # 8 头 Codec 输出
        self.embed_tokens = TalkerEmbedding(2112, talker_hidden_size)  # 8 头 Codec 输入
        self.codec_proj = Linear → GELU → Linear → RMSNorm        # 音频 embedding 投影
        self.embed_proj = Linear → GELU → Linear → RMSNorm        # 文本 hidden 投影
        self.text_scale = nn.Parameter(3.0)                        # 文本融合缩放
        self.audio_scale = nn.Parameter(1.0)                       # 音频融合缩放
        self.spk_proj = Linear(192, talker_hidden_size)            # 说话人嵌入
        # 独立的 RoPE buffer
```

### TalkerHead -- 低秩 8 头输出

```python
TalkerHead(in_features=768, out_features=2112, num_layers=8, rank=256):
  base = Linear(768, 2112)              # 共享基座
  adapters = [
    Linear(768, 256) → GELU → Linear(256, 2112)  # 每层独立适配器
    for _ in range(8)
  ]
  
  def forward(x):
    base_out = base(x)
    return [base_out + adapter(x) for adapter in adapters]  # 8 组 logits
```

每个 codebook layer 的输出 = 共享基座 + 该层低秩适配器。这比 8 个独立 Linear 头更省参数。

### TalkerEmbedding -- 低秩 8 头输入

```python
TalkerEmbedding(num_embeddings=2112, embedding_dim=768, num_layers=8, rank=256):
  base = Embedding(2112, 768)           # 共享基座
  adapters = [
    Embedding(2112, 256) → GELU → Linear(256, 768)
    for _ in range(8)
  ]
  
  def forward(x):  # x: (B, 8, T)
    base_out = base(x)              # (B, 8, T, D)
    sum_out = base_out[:, 0, :, :]  # 从第 0 层开始
    for i in range(8):
        sum_out += adapters[i](x[:, i, :])
    return sum_out / 8
```

---

## 音频编码器 -- SenseVoice-Small

| 属性 | 值 |
|------|-----|
| 模型 | SenseVoice-Small |
| 来源 | FunASR (`funasr.AutoModel`) |
| 参数 | ~234M |
| Encoder 层数 | 50 |
| Frontend | Fbank 前端 |
| 输出特征维度 | 512 |
| 采样率 | 16000 Hz |
| 是否冻结 | 是 |

```python
# 加载逻辑
from funasr import AutoModel
m = AutoModel(model=path, trust_remote_code=True, ...)
encoder, frontend = m.model.encoder, m.kwargs["frontend"]
```

### MMAudioProjector

```python
MMAudioProjector(512 → 768):
  LayerNorm(512)
  → Linear(512, 768)
  → GELU
  → Linear(768, 768)
```

参数量: `512 + 512×768 + 768×768 = 983,040 ≈ 0.99M`

### SenseVoiceAudioProcessor

自定义处理器 (`omni.py:113`)，将原始音频波形转换为 Fbank 特征：

```python
processor = SenseVoiceAudioProcessor(frontend)
inputs = processor(wav, sampling_rate=16000)
# inputs.input_features: Fbank 特征
# inputs.attention_mask: 有效长度掩码
```

---

## 视觉编码器 -- SigLIP2

与 VLM 中的视觉编码器完全相同。当模型同时加载音频和视觉编码器时，参数量显著增加，但两者均在推理时冻结。

---

## 前向传播

`VersperOmni.forward` (`omni.py:416`) 是核心方法，处理三种输入模式。

### 输入格式

```
纯文本:  input_ids shape = (B, T)
多流:    input_ids shape = (B, 9, T)  # 8 层 audio codes + 1 层 text
```

### Thinker 阶段

```python
# 1. 文本 embedding
hidden_states = thinker.dropout(thinker.embed_tokens(text_ids))

# 2. 音频特征注入 (首次前向)
if audio_inputs is not None and start_pos == 0:
    audio_features = encode_audio_inputs(audio_inputs, audio_lens)
    hidden_states = inject_audio_features(text_ids, hidden_states, audio_features)

# 3. 视觉特征注入 (首次前向)
if pixel_values is not None and start_pos == 0:
    vision_tensors = encode_image_inputs(pixel_values)
    hidden_states = inject_vision_features(text_ids, hidden_states, vision_tensors)

# 4. Thinker 逐层前向，记录 bridge states
for i, layer in enumerate(thinker.layers):
    hidden_states = layer(hidden_states, ...)
    if i == bridge_layer:         # 默认第 3 层
        bridge_states = hidden_states
h_thinker = thinker.norm(hidden_states)
```

### Talker 阶段

```python
# 5. Audio embedding (8 层 codec)
talker_emb = talker.embed_tokens(audio_ids)

# 6. 融合: bridge_states(文本) + talker_emb(音频)
hidden_states = (
    talker.embed_proj(bridge_states) * talker.text_scale   # 文本路径
    + talker.codec_proj(talker_emb) * talker.audio_scale    # 音频路径
)

# 7. Talker 逐层前向
for layer in talker.layers:
    hidden_states = layer(hidden_states, ...)
h_talker = talker.norm(hidden_states)
```

### 输出

```python
text_logits = thinker.lm_head(h_thinker)      # (B, T, vocab_size)
audio_logits = talker.lm_head(h_talker)        # 8 × (B, T, 2112)
# audio_logits 作为额外属性返回
out.audio_logits = audio_logits
```

---

## 流式生成 -- _stream_generate

定义在 `omni.py:616`，同时生成文本和 8 层音频 codes。

### 核心逻辑

```python
def _stream_generate(self, input_ids, ...):
    while input_ids.shape[1] < start_pos + max_new_tokens:
        # 1. 多流前向: audio_buffer + text_token
        out = self.forward(
            torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1)
        )
        
        # 2. 文本 token 采样 (top-p, repetition penalty)
        text_token = sample(out.logits)
        
        # 3. 音频 codes 采样 (8 层交错更新)
        for i in range(8):
            code = sample(out.audio_logits[i])
            audio_codes[i].append(code)
        
        # 4. 更新 audio buffer
        audio_buffer[0, i, -1] = audio_codes[i][-1]
        
        # 5. 流式输出
        yield text_output, audio_frame
```

### 关键设计

- **Thinking 检测**: 检测 `think_end_ids` 序列（`["</think>\n\n"]`）来确定何时开始音频输出
- **延迟采样**: 第 `i` 层 codec 的输出比文本延迟 `i` 步，形成交错金字塔结构
- **Stop 检测**: 当所有 8 层 codec 均输出 `>= 2048`（stop token）时停止解码
- **音频帧**: 当 `audio_step >= 7` 时，每步输出一帧完整音频（8 层各一个 code）

### `generate()` 的 `**kwargs` 参数

`VersperOmni.generate()` 接受以下通过 `**kwargs` 传递的额外参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `spk_emb` | `torch.Tensor` (1, 192) | `None` | 说话人声纹嵌入，用于声音克隆 |
| `ref_codes` | `torch.Tensor` (1, 8, T) | `None` | 参考音频的 Mimi codec 码，用于上下文声音克隆 |
| `open_thinking` | `bool` | `False` | 是否跟踪 thinking token 序列以确定音频输出开始时机 |
| `enter_token_id` | `int` | 201 | 文本生成完成后，替换文本 token 的填充 token |
| `pad_token_id` | `int` | 0 | 文本完成后的填充 token ID |

使用时直接作为关键字参数传入：

```python
# 声音克隆示例：传入参考说话人嵌入
spk = torch.load("speaker_embedding.pt")  # shape (1, 192)
ref = torch.load("reference_codes.pt")     # shape (1, 8, T)
tokens, audio = model.generate(
    input_ids,
    spk_emb=spk,
    ref_codes=ref,
    stream=True,
).__next__()
```

### 非流式模式

当 `stream=False`（默认）时，`generate()` 内部消费整个 `_stream_generate()` 生成器并返回最后一个 `(token_ids, audio_frame)` 对：

```python
# 非流式：等生成完成后获取最终结果
tokens, audio = model.generate(input_ids, max_new_tokens=256)
```

---

## 权重初始化

从预训练 LM 权重初始化 Omni 模型时，Talker 模块可以拷贝 Thinker 的最后几层：

```python
if omni_config.talker_hidden_size == omni_config.hidden_size:
    n_talker = config.num_talker_hidden_layers
    n_thinker = len(model.thinker.layers)
    for i in range(n_talker):
        src = n_thinker - n_talker + i  # 取 Thinker 的最后 n_talker 层
        model.talker.layers[i].load_state_dict(
            model.thinker.layers[src].state_dict()
        )
```

这要求 `talker_hidden_size == hidden_size`，否则维度不匹配时会自动跳过。
