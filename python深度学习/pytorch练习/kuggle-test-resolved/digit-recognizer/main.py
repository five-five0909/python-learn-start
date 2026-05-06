# ============================================================================
# main.py — Kaggle Digit Recognizer 主程序入口
# ============================================================================
#
# 使用方式：
#   python digit-recognizer/main.py
#
# 整体流程：
#   1. 加载数据        → DigitDataset 从 CSV 加载 + 像素归一化
#   2. 划分数据集      → random_split 按 90/10 划分训练集和验证集
#   3. 创建 DataLoader → 自动分 batch + 打乱顺序
#   4. 创建模型组件    → 模型、损失函数、优化器
#   5. 训练主循环      → 训练 → 验证 → 打印状态 → 早停检查
#   6. 加载最优模型    → 从文件恢复最佳 epoch 的权重
#   7. 验证集评估      → 计算准确率
#   8. 生成提交文件    → 用最优模型预测测试集，输出 CSV
#
# ============================================================================

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from .config import (
        device, SEED, TRAIN_PATH, TEST_PATH, MODEL_PATH, SUBMISSION_PATH,
        BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE, INPUT_DIM, NUM_CLASSES,
    )
    from .dataset import DigitDataset
    from .model import NeuralNetwork
    from .checkpoint import EarlyStopping
    from .metrics import evaluate_metrics
    from .trainer import train_loop, val_loop
except ImportError:
    from config import (
        device, SEED, TRAIN_PATH, TEST_PATH, MODEL_PATH, SUBMISSION_PATH,
        BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE, INPUT_DIM, NUM_CLASSES,
    )
    from dataset import DigitDataset
    from model import NeuralNetwork
    from checkpoint import EarlyStopping
    from metrics import evaluate_metrics
    from trainer import train_loop, val_loop


# ============================================================================
# 1. 加载数据
# ============================================================================

print("=" * 60)
print("【加载数据】")
print("=" * 60)

train_set = DigitDataset(TRAIN_PATH)
test_set = DigitDataset(TEST_PATH)


# ============================================================================
# 2. 划分训练集 / 验证集 & 创建 DataLoader
# ============================================================================

val_size = int(0.1 * len(train_set))
train_size = len(train_set) - val_size

train_subset, val_subset = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n数据划分：训练集 {train_size} 条，验证集 {val_size} 条")


# ============================================================================
# 3. 模型 & 损失函数 & 优化器
# ============================================================================

model = NeuralNetwork(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)

print(f"模型已初始化，输入维度={INPUT_DIM}，输出类别={NUM_CLASSES}，设备={device}")

loss_fn = nn.CrossEntropyLoss()
# ↑ 交叉熵损失函数，分类任务的标准选择
#   内部包含 Softmax，所以模型输出 logits 即可

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ============================================================================
# 4. 训练主循环
# ============================================================================

print("\n" + "=" * 60)
print("【开始训练】")
print("=" * 60)

early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_PATH)

for epoch in range(1, EPOCHS + 1):
    train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
    val_loss = val_loop(val_loader, model, loss_fn, device)

    # ---- 计算验证准确率 ----
    val_metrics = evaluate_metrics(val_loader, model, device)
    val_acc = val_metrics["accuracy"]

    # ---- 打印训练状态 ----
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3} | 训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")

    # ---- 早停检查 ----
    early_stopping(val_acc, model)

    if early_stopping.early_stop:
        print(f"\n连续 {PATIENCE} 个 epoch 无提升，训练终止于 Epoch {epoch}。")
        print(f"最佳验证准确率: {early_stopping.best_accuracy:.4f}")
        break

print(f"\n训练完成，最佳模型已保存至 {MODEL_PATH}")


# ============================================================================
# 5. 加载最佳模型 → 在验证集上评估
# ============================================================================

print("\n" + "=" * 60)
print("【验证集评估】")
print("=" * 60)

best_model = NeuralNetwork(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
best_model.eval()

val_metrics = evaluate_metrics(val_loader, best_model, device)
print(f"  验证准确率 = {val_metrics['accuracy']:.4f}")


# ============================================================================
# 6. 生成 Kaggle 提交文件
# ============================================================================

print("\n" + "=" * 60)
print("【生成提交文件】")
print("=" * 60)

all_preds = []

with torch.no_grad():
    for X in test_loader:
        pred = best_model(X.to(device))
        predicted = pred.argmax(dim=1).cpu()
        all_preds.append(predicted)

submission = pd.read_csv(TEST_PATH)
# ↑ 读取原始测试集 CSV

submission = pd.DataFrame({
    'ImageId': range(1, len(submission) + 1),
    'Label': torch.cat(all_preds).numpy()
})

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"提交文件已生成：{SUBMISSION_PATH}")
print(submission.head(10))

print("\n训练流程完成。")
