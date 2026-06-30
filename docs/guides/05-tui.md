# TUI 终端界面使用指南

> 训练与推理一体的终端 UI，基于 Textual 框架
> 入口：`versper-tui` 或 `python -m versper.tui`

## 启动

```bash
# 方式一：CLI 入口（推荐）
versper-tui

# 方式二：Python 模块
python -m versper.tui

# 方式三：代码中启动
python -c "from versper.tui import VersperTUI; VersperTUI().run()"
```

## 界面概览

TUI 包含三个标签页，底部有全局快捷键提示：

```
┌──────────────────────────────────────────────────────┐
│  VersperOmni TUI                     🕐 14:30:00     │
├──────────────────────────────────────────────────────┤
│  🖥  Inference  │  🎯  Training  │  📋  Logs         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  [当前标签页内容...]                                    │
│                                                      │
├──────────────────────────────────────────────────────┤
│  • q:Quit  • h:Help  • ...                          │
└──────────────────────────────────────────────────────┘
```

## Inference 标签页

### 模型加载
1. 选择 **Model Type**（LM / VLM / Omni）
2. 选择 **Device**（auto / cpu / cuda:0）
3. 可选：填写权重路径（如 `./out/pretrain_768.pth`）
4. 点击 **Load Model**

### 使用
- 在输入框中输入提示词
- 点击 **Generate**（或 `Ctrl+G`）运行推理
- VLM 模式下，在提示词中使用 `![描述](image.jpg)` 语法传入图片路径
- 开启 **Stream output** 启用流式输出

### 快捷键
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+G` | 生成回复 |
| `Ctrl+C` | 清空输出 |


## Training 标签页

### 训练配置
- **Training Mode**: 选择训练类型（Pretrain / SFT-VLM / SFT-Omni）
- **Data path**: 训练数据路径
- **Weight path**: 预训练权重路径
- **Batch size / Learning rate**: 训练超参数

### 布局

Training 标签页采用 Burn TUI 风格的分栏布局：

```
┌───────────────────────────┬─────────────────────────────┐
│ 左侧面板 (38%)            │ 右侧面板 (62%)              │
│                           │                             │
│  Controls (快捷键提示)     │  折线图 (Braille 点阵渲染)   │
│  Status (当前指标/模式)    │  支持 ←/→ 切换 loss / lr    │
│  Metrics (实时训练日志)    │  ↑/↓ 切换 Full/Recent       │
│                           │                             │
├───────────────────────────┴─────────────────────────────┤
│  进度条：Epoch ████████░░  ETA: 5 mins                  │
│  进度条：Batch ██████░░░░                               │
└─────────────────────────────────────────────────────────┘
```

### 操作
1. 配置训练参数
2. 点击 **Start Training** 开始训练
3. 训练过程中实时显示：
   - Epoch 进度条 + Batch 进度条
   - ETA 预估时间（右下角）
   - Loss / LR 实时折线图（右侧面板，Braille 点阵渲染）
   - 训练日志（左侧面板 Metrics 区域）
   - 当前状态（左侧面板 Status 区域）
4. 使用 `←`/`→` 切换图表显示的指标（loss ↔ lr）
5. 使用 `↑`/`↓` 切换图表类型（Full / Recent / Summary）
6. 可随时点击 **Stop**（或 `Ctrl+S` / `q`→Stop）终止训练
7. 训练完成时弹出总结弹窗

### 训练输出解析
TUI 自动解析训练脚本的 stdout 输出：
- `Epoch:[1/2]` → 更新 epoch 进度条
- `loss: 3.2456` → 日志显示
- `lr: 0.00050` → 学习率记录


## Logs 标签页

显示所有日志消息，包括：
- 模型加载/卸载
- 推理日志
- 训练输出
- 错误和警告

支持 **Clear Logs** 和 **Copy Logs**（需安装 pyperclip）。


## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| `q` | 退出应用（训练中弹出确认框） |
| `h` | 显示帮助弹窗 |
| `Tab` | 切换标签页 |
| `Ctrl+I` | 切换到 Inference |
| `Ctrl+T` | 切换到 Training |
| `Ctrl+L` | 切换到 Logs |
| `Ctrl+G` | Inference 标签页中生成回复 |
| `Ctrl+C` | Inference 标签页中清空输出 |
| `Ctrl+S` | Training 标签页中启动/停止训练 |
| `←` / `→` | 切换图表显示的指标（loss ↔ lr） |
| `↑` / `↓` | 循环切换图表类型（Full / Recent / Summary） |

## 支持的模型

| 模型类型 | 配置类 | 模型类 | 输入 | 输出 |
|---------|--------|--------|------|------|
| LM | MiniMindConfig | MiniMindForCausalLM | 文本 | 文本 |
| VLM | VLMConfig | MiniMindVLM | 文本+图片 | 文本 |
| Omni | OmniConfig | MiniMindOmni | 文本/语音/图片 | 文本+音频 |


## 注意事项

- 首次加载模型需要下载 tokenizer 文件（放在 `./model/` 目录）
- VLM 和 Omni 模型需要额外的编码器权重（SigLIP2 / SenseVoice）
- 训练功能通过启动子进程实现，需要对应数据集和训练脚本
- 训练时 CUDA OOM 可通过降低 batch size 缓解
