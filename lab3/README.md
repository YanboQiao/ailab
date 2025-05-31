# MNIST-CNN Lab3 📚🧠

基于 **PyTorch + Apple Silicon M-series** 的轻量级卷积神经网络示例。
完成从 **数据增强 → 训练 → 评估 → 可视化** 的全流程，便于同学撰写实验报告、对比基线模型并分析超参数。

---

## 目录结构

```text
lab3/
├── data.py          # 数据读取 & 增强 (RandomAffine)
├── model.py         # CNN 模型 (Conv → Pool → FC)
├── train.py         # 训练/评估脚本，支持日志 & 曲线输出
├── visualize.py     # 测试样本预测可视化
├── requirements.txt # 建议依赖版本 (PyTorch 2.3.0+CPU/MPS)
└── runs/            # 训练输出 (best_model.pt / metrics.png / log.csv …)
```

---

## 1. 环境准备

> **默认 Python 3.11，Apple M-series 优先启用 MPS；Intel Mac / Windows / Linux 可自动回退 CPU。**

```bash
# 1⃣️ 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 2⃣️ 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
# 若想用官方 MPS 轮子，请参考 https://pytorch.org 获取对应 index-url
```

`requirements.txt`（示例）

```
torch>=2.3.0        # macOS MPS / CPU 均可
torchvision>=0.18.0
torchaudio>=2.3.0
matplotlib>=3.9
pandas>=2.2
```

---

## 2. 数据集准备

```bash
# 推荐目录：lab3/MNIST/
mkdir -p MNIST
```

运行脚本时会**自动下载** `train-images-idx3-ubyte.gz` 等四个文件；若校园网受限，可预先手动放入同名文件。

---

## 3. 训练模型

```bash
python train.py \
    --data-dir ./MNIST \
    --output-dir runs/mnist_cnn \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --save-metrics
```

| 输出                 | 说明                    |
| ------------------ | --------------------- |
| `best_model.pt`    | 测试集准确率最高的权重           |
| `training_log.csv` | 每个 epoch 的 loss / acc |
| `metrics.png`      | 训练 & 测试准确率曲线          |

**默认随机种子**已固定（42），确保结果可重复（±0.1% 浮动）。

---

## 4. 可视化预测

```bash
python visualize.py \
    --model-path runs/mnist_cnn/best_model.pt \
    --data-dir ./MNIST \
    --n-samples 36 \
    --rows 6
```

弹出 6×6 网格，标题展示 `GT / Pred` 方便人工检视。

---

## 5. 快速对比基线 (MLP) 【可选】

```bash
python train.py --model mlp --epochs 10
```

生成的 `runs/mlp_*` 日志可与 CNN 曲线、准确率直接对照，用于报告中体现 **卷积特征提取优势**。

---

## 6. 关键结果

| 模型               | 训练 Epoch | Test Acc (10k) |
| ---------------- | -------- | -------------- |
| **CNN (本项目)**    | 10       | **≈ 99.4 %**   |
| MLP (2 × FC 256) | 10       | ≈ 97.8 %       |

> 结论：卷积层显著提高 MNIST 识别性能，同时参数量仅 \~1.2 M，推理开销低。

---

## 7. 参数说明

| 参数               | 默认    | 作用               |
| ---------------- | ----- | ---------------- |
| `--epochs`       | 10    | 训练轮数             |
| `--batch-size`   | 128   | 批大小              |
| `--lr`           | 1e-3  | Adam 学习率         |
| `--num-workers`  | 2     | DataLoader 进程数   |
| `--model`        | `cnn` | 选择 `cnn` / `mlp` |
| `--save-metrics` | flag  | 保存 CSV + 曲线 PNG  |

---

## 8. TensorFlow 实现 【选学】

若需在报告中做 **框架对比**，可在 `tf_train.py` 里用 `tf.keras.Sequential` 复刻同结构。对比点可包括：

* API 设计（高阶封装 vs 动态编程）
* 训练流程（`model.compile/fit` vs 手写循环）
* MPS / GPU 支持
* 社区教程、生态工具等

---

## 9. 常见问题

| 问题                                      | 解决方案                                         |
| --------------------------------------- | -------------------------------------------- |
| `pin_memory` warning on MPS             | 特性未实现，无影响；脚本已默认 `pin_memory=False`。          |
| `linear(): shapes cannot be multiplied` | Conv/Pool 输出尺寸与 `fc1` 不匹配，确认两次池化后特征图为 `7×7`。 |
| 下载超时                                    | 先手动下载四个 `*.gz` 放 `MNIST/`。                   |
