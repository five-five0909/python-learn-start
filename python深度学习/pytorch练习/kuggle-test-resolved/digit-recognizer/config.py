# ============================================================================
# config.py — 全局配置模块（Digit Recognizer）
# ============================================================================
#
# Kaggle Digit Recognizer 竞赛：手写数字识别（MNIST）
# 任务：根据 28x28 像素图片，识别数字 0~9（10 分类）
#
# ============================================================================

import os
import random
import numpy as np
import torch


# ============================================================================
# 超参数
# ============================================================================

SEED = 42

DATA_DIR = "../data/digit-recognizer"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

OUT_DIR = os.path.join(DATA_DIR, "out")
MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")
SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")

BATCH_SIZE = 128
# ↑ 每批次样本数
#   42000 条数据，batch_size=128 → 每 epoch 约 328 个 batch

EPOCHS = 50

PATIENCE = 10
# ↑ 早停耐心值：连续 10 个 epoch 验证准确率没提升就停

LEARNING_RATE = 1e-3

VAL_RATIO = 0.1
# ↑ 验证集比例 10%（42000 条中取 4200 条验证）

INPUT_DIM = 784
# ↑ 输入维度：28 x 28 = 784 像素

NUM_CLASSES = 10
# ↑ 输出类别数：数字 0~9


# ============================================================================
# 设备选择
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 创建输出文件夹
# ============================================================================

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# 固定随机种子
# ============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()
