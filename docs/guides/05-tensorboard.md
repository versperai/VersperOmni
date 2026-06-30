# TensorBoard 训练可视化

## 安装

```bash
# TensorBoard 作为可选依赖
pip install tensorboard

# 或安装完整的 train 依赖组
pip install versperomni[train]
```

## 使用

所有三个训练脚本（Pretrain / SFT-VLM / SFT-Omni）均支持 `--use_tensorboard` 标志：

```bash
# LM 预训练
python -m versper.trainer.pretrain \
    --data_path ./dataset/pretrain_t2t_mini.jsonl \
    --use_tensorboard

# VLM SFT
python -m versper.trainer.sft_vlm \
    --data_path ./dataset/sft_i2t.parquet \
    --use_tensorboard

# Omni SFT
python -m versper.trainer.sft_omni \
    --data_path ./dataset/sft_t2a.parquet \
    --mode all \
    --lr 5e-6 \
    --use_tensorboard
```

## 查看

TensorBoard 事件文件输出到 `{save_dir}/runs/{run_name}/`（默认 `./out/runs/`）。

```bash
# 启动 TensorBoard（默认端口 6006）
tensorboard --logdir ./out/runs

# 指定端口
tensorboard --logdir ./out/runs --port 8080
```

在浏览器中打开 `http://localhost:6006` 查看：

| 指标 | 说明 |
|------|------|
| `loss` | 训练损失（所有模型） |
| `lr` | 学习率 |
| `text_loss` | 文本损失（Omni 模型） |
| `audio_loss` | 音频损失（Omni 模型） |
| `eta_min` | 预估剩余时间（分钟） |

## 同时使用 WandB

--use_tensorboard 可与 --use_wandb 同时使用，两者互不冲突。
