# ============================================================================
# dataset.py — 数据集模块（Digit Recognizer）
# ============================================================================
#
# MNIST 手写数字数据集：
#   训练集：42000 条，每条 = 1 个标签（0~9）+ 784 个像素值（0~255）
#   测试集：28000 条，每条 = 784 个像素值（无标签）
#
# 数据处理：
#   1. 读取 CSV
#   2. 像素值归一化到 [0, 1]（除以 255）
#   3. 标签转为 long 类型（CrossEntropyLoss 要求）
#
# ============================================================================

import torch
import pandas as pd
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    """手写数字数据集：从 CSV 加载，自动归一化像素值。

    Args:
        csv_path:   CSV 文件路径
        label_col:  标签列名（默认 "label"）
                    测试集没有这列，会自动跳过
        has_label:  是否有标签列（训练集 True，测试集 False）
    """

    def __init__(self, csv_path, label_col='label'):
        df = pd.read_csv(csv_path)

        # ---- 分离标签 ----
        if label_col in df.columns:
            self.y = torch.tensor(df[label_col].values, dtype=torch.long)
            df = df.drop(columns=[label_col])
        else:
            self.y = None

        # ---- 像素值归一化到 [0, 1] ----
        X = df.values.astype('float32') / 255.0
        self.X = torch.tensor(X, dtype=torch.float32)

        print(f"加载完成：{len(self.X)} 条样本，{self.X.shape[1]} 个像素特征"
              f"{'，有标签' if self.y is not None else '，无标签（测试集）'}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
