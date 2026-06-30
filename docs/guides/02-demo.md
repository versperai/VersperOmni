# 交互式演示与 Web Demo

> 快速体验 VersperOmni 语言模型的文本对话功能。

## 1. 快速启动

项目内置了一个简单的交互式文本聊天脚本 `web_demo.py`，基于纯 `input()` 循环实现，无需额外依赖。

```bash
python -m versper.scripts.web_demo
```

启动后终端会进入交互循环：

- 输入 `exit` 或 `quit` 退出
- 支持多轮对话（无历史管理，目前仅单轮）

## 2. 源码走读

`src/versper/scripts/web_demo.py` 的实现极为精简，核心流程如下：

```python
import torch
from transformers import AutoTokenizer
from versper.config import MiniMindConfig
from versper.model import MiniMindForCausalLM

config = MiniMindConfig()
model = MiniMindForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("./model")

# 加载预训练权重
state_dict = torch.load("./out/pretrain_768.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()

while True:
    prompt = input("\nUser: ")
    if prompt.strip().lower() in ("exit", "quit"):
        break
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    out = model.generate(input_ids, max_new_tokens=1024, temperature=0.85, top_p=0.85)
    response = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Assistant: {response}")
```

### 关键参数说明

| 参数               | 值    | 说明                     |
| ------------------ | ----- | ------------------------ |
| `max_new_tokens`   | 1024  | 最长生成长度             |
| `temperature`      | 0.85  | 采样温度，控制随机性     |
| `top_p`            | 0.85  | 核采样阈值               |
| 权重路径           | `./out/pretrain_768.pth` | 预训练 checkpoint |

## 3. 扩展方向

当前 `web_demo.py` 仅提供最小可用的文本交互循环，可根据需要扩展：

### 3.1 添加 Gradio Web UI

```python
import gradio as gr

def chat(message, history):
    messages = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    out = model.generate(input_ids, max_new_tokens=1024, temperature=0.85, top_p=0.85)
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

gr.ChatInterface(chat, title="VersperOmni Chat").launch()
```

### 3.2 添加流式输出

利用 `MiniMindForCausalLM.generate()` 的 `streamer` 参数，或直接使用逐 token 生成循环：

```python
from versper.model import MiniMindForCausalLM

# ... 加载模型 ...

for token_id in model.generate(input_ids, max_new_tokens=256, stream=True):
    # 逐 token 处理（例如实时打印或用 yield 流式返回）
    pass
```

### 3.3 添加音频 I/O（Omni 模型）

使用 `MiniMindOmni.generate()` 时，流式生成的每次迭代会返回 `(tokens, audio_frame)` 二元组，其中 `audio_frame` 为 8 通道 Mimi 编码，可通过内置或外部 Neural Codec 解码为波形：

```python
from versper.omni import MiniMindOmni

model = MiniMindOmni(config)

for tokens, audio_frame in model.generate(input_ids, stream=True, return_audio_codes=True):
    if audio_frame is not None:
        # audio_frame: list of 8 ints → 解码为 PCM 播放
        pass
    if tokens is not None:
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(text, end="", flush=True)
```

详情参见 [04-advanced.md](./04-advanced.md)。
