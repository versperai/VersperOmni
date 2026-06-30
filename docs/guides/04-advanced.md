# 高级特性 — 流式推理、语音活动检测、实时会话

> 本文档覆盖 `src/versper/omni.py` 和 `src/versper/model.py` 中几个未在基础文档中展开的高级特性。

---

## 1. Omni Generate 参数

`MiniMindOmni.generate()` 接受可选的 `**kwargs`，用于控制语音生成的行为：

```python
from versper.omni import MiniMindOmni

model = MiniMindOmni(config)

output = model.generate(
    input_ids,
    stream=True,
    spk_emb=speaker_embedding,      # (1) 说话人嵌入
    ref_codes=reference_codes,       # (2) 参考音频编码
    open_thinking=False,             # (3) 思考标记追踪
    enter_token_id=201,              # (4) 文本结束后的替换 token
    pad_token_id=0,                  # (5) 填充 token
)
```

### 参数详解

| 参数              | 类型               | 默认值   | 说明                                                                 |
| ----------------- | ------------------ | -------- | -------------------------------------------------------------------- |
| `spk_emb`         | `torch.Tensor`     | `None`   | 形状 `(1, 192)` 的说话人嵌入向量，用于语音克隆                       |
| `ref_codes`       | `torch.Tensor`     | `None`   | 形状 `(1, 8, ref_len)` 的参考音频 codec 编码，用于上下文语音克隆     |
| `open_thinking`   | `bool`             | `False`  | 若为 `True`，追踪 `think_end_ids`，决定音频输出何时开始               |
| `enter_token_id`  | `int`              | `201`    | 文本生成完成后，替换后续文本 token 的值                               |
| `pad_token_id`    | `int`              | `0`      | 文本生成完成后，`enter_token_id` 之后使用的填充 token                 |

### `spk_emb` — 说话人嵌入

CAM++ 等说话人编码器输出的 192 维向量。在 Talker 模块中通过 `spk_proj` 线性层投影后，替换音频输入中的 `audio_spk_token` 位置嵌入。

### `ref_codes` — 参考音频编码

用于 In-Context Voice Cloning（上下文语音克隆）。传入一段参考音频的 8 通道 Mimi Codec 编码，模型会利用该编码的上下文来匹配目标说话人的音色、语调和风格。

### `open_thinking` — 思考标记追踪

启用后，生成过程中会检测输出序列中是否出现 `think_end_ids` 标记。一旦检测到，`audio_step` 开始计数，从而控制音频生成的起始时刻，实现"思考后语音输出"的效果。

---

## 2. SileroVAD — 语音活动检测

`SileroVAD` 是基于 ONNX 的轻量级语音活动检测器，位于 `src/versper/omni.py:768`。

### 接口

```python
from versper.omni import SileroVAD

vad = SileroVAD("path/to/silero_vad.onnx")

prob = vad(audio_chunk, sr=16000)
if prob > 0.5:
    print("speech detected")
else:
    print("silence")

vad.reset()  # 重置内部状态
```

### 实现细节

- 基于 ONNX Runtime（`CPUExecutionProvider`），单线程推理
- 内部维护 LSTM 状态 `h` / `c`（形状 `(2, 1, 64)`），在连续音频流上保持状态
- `__call__(chunk, sr=16000)` → 返回 `float` 类型概率值（0~1）
- `reset()` 将内部状态清零

### 参数

| 参数    | 类型           | 默认值 | 说明                 |
| ------- | -------------- | ------ | -------------------- |
| `chunk` | `np.ndarray`   | —      | 输入音频块（1D 波形） |
| `sr`    | `int`          | 16000  | 采样率               |

---

## 3. RealtimeSession — 实时语音会话管理

`RealtimeSession` 是一个完整的 VAD 驱动的交互式语音会话管理器，位于 `src/versper/omni.py:798`。它管理从"静默监听 → 说话检测 → 语音结束 → 中断处理"的完整状态机。

### 初始化参数

```python
from versper.omni import RealtimeSession

session = RealtimeSession(
    vad_path="silero_vad.onnx",
    sr=16000,
    threshold=0.8,
    min_speech_ms=128,
    min_silence_ms=800,
)
```

| 参数             | 类型    | 默认值  | 说明                               |
| ---------------- | ------- | ------- | ---------------------------------- |
| `vad_path`       | `str`   | —       | Silero VAD ONNX 模型路径            |
| `sr`             | `int`   | 16000   | 音频采样率                         |
| `threshold`      | `float` | 0.8     | VAD 阈值，高于此值视为说话         |
| `min_speech_ms`  | `int`   | 128     | 最短说话时长（毫秒），避免误触发   |
| `min_silence_ms` | `int`   | 800     | 最短静音时长（毫秒），判定语音结束 |

### 状态机

```
listening  ──(检测到说话)──>  speaking  ──(检测到静音)──>  speech_end  ──(reset)──>  listening
                                  │
                                  └──(生成过程中检测到新语音)──>  interrupt
```

### push_chunk 返回值

| 返回值        | 说明                                                                 |
| ------------- | -------------------------------------------------------------------- |
| `"listening"` | 静默监听状态，未检测到有效语音                                       |
| `"speech_end"`| 检测到语音结束，可通过 `get_audio()` 获取完整语音段                  |
| `"interrupt"` | 生成过程中用户开始说话，需停止当前生成，返回监听状态                 |

### 示例用法

```python
session = RealtimeSession("silero_vad.onnx")

def audio_callback(audio_chunk):
    status = session.push_chunk(audio_chunk)
    if status == "speech_end":
        audio = session.get_audio()
        # 处理完整语音输入（例如 ASR → LLM → TTS）
        process_utterance(audio)
    elif status == "interrupt":
        # 用户打断了模型输出，立即停止生成
        stop_generation()
        session.reset()
        # 开始监听用户的新输入
```

### 音频环形缓冲区（Ring Buffer）

为了**防止语音起始被截断**，`RealtimeSession` 内部维护了一个小型环形缓冲区 `self.ring`。在未检测到说话时，持续的音频块会被暂存在该缓冲区中。一旦 VAD 确认说话状态（`speech_samples >= min_speech`），缓冲区中的历史音频会被拼接到 `buffer` 开头，确保语音开头不被系统前置静音过滤所剪切：

```python
if self.speech_samples >= self.min_speech and not self.speaking:
    self.speaking = True
    self.buffer = self.ring + self.buffer  # 拼接环缓冲历史
    self.ring = []
```

---

## 4. MiniMindForCausalLM.generate() return_kv

基础语言模型 `MiniMindForCausalLM` 的 `generate()` 支持通过 `return_kv=True` 返回 KV Cache，用于高效的多次续写（例如交互式译码、多轮对话的 prefix caching）。

```python
from versper.model import MiniMindForCausalLM

model = MiniMindForCausalLM(config).cuda().eval()

# 第一次生成，同时返回 KV Cache
result = model.generate(input_ids, return_kv=True)
tokens = result["generated_ids"]
past_kv = result["past_kv"]  # 可复用于续写

# 第二次生成，复用之前的 KV Cache，避免重复计算 prefix
continuation = model.generate(
    tokens[:, -1:],  # 仅输入最后一个 token
    past_key_values=past_kv,
)

# 或者使用更底层的循环逐 token 生成
for _ in range(max_new_tokens):
    out = model.forward(
        tokens[:, -1:],
        past_key_values=past_kv,
        use_cache=True,
    )
    past_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=-1)
```

---

## 5. load_weight_path — 权重路径工具

`src/versper/trainer/utils.py:190` 提供的路径拼接工具，用于自动生成 Checkpoint 文件名，支持 MoE 后缀处理。

```python
from versper.trainer.utils import load_weight_path

config = MiniMindConfig(use_moe=True)

# 返回: "./out/full_768_moe.pth"
path = load_weight_path(config, weight_name="full", save_dir="./out")

# 非 MoE 模型:
config.use_moe = False
# 返回: "./out/full_768.pth"
path = load_weight_path(config, weight_name="full", save_dir="./out")
```

### 实现逻辑

```python
def load_weight_path(config, weight_name, save_dir="../out"):
    moe_suffix = "_moe" if config.use_moe else ""
    return f"{save_dir}/{weight_name}_{config.hidden_size}{moe_suffix}.pth"
```

该函数被训练脚本内部用于自动查找 Checkpoint，配合 `save_checkpoint()` 保证权重命名的对称性，避免训练恢复时路径不匹配。
