# ============================================================================
# dataset.py — 数据集模块
# ============================================================================
#
# 职责：定义自定义 Dataset 类，从 CSV 文件加载数据并自动处理特征工程
#
# 和之前版本的区别：
#   之前：Dataset 只接收 numpy 数组，特征工程在 main.py 里做
#   现在：Dataset 自己从 CSV 加载，内部完成所有特征工程
#   这样更封装——外部只需要传一个 CSV 路径，不用管细节
#
# 特征工程流程（在 __init__ 中自动完成）：
#   1. 读取 CSV 文件
#   2. 分离标签列（SalePrice）
#   3. 排除无用列（Id）
#   4. 识别数值列 vs 类别列
#   5. 填充缺失值（数值列用中位数，类别列用 'Missing'）
#   6. 类别列 → LabelEncoder 转数字
#   7. 数值标准化（StandardScaler）
#
# 为什么用 LabelEncoder 而不是 One-Hot 编码？
#   LabelEncoder：把类别转成单个数字（如 "Ex"→0, "Gd"→1, "TA"→2）
#   One-Hot：把类别转成多个 0/1 列（如 "Ex"→[1,0,0], "Gd"→[0,1,0]）
#   LabelEncoder 更简单，特征维度更少，适合初步实验
#
# 自定义 Dataset 必须实现 3 个方法：
#   __init__    → 初始化：加载数据、特征工程、转成张量
#   __len__     → 返回数据总数
#   __getitem__ → 返回第 idx 条数据（特征 + 标签）
#
# ============================================================================

import numpy as np
# ↑ NumPy：科学计算库
#   这里不直接使用，但数据通常以 numpy 数组的形式传入

import torch
# ↑ PyTorch 核心库
#   torch.tensor() 把 numpy 数组转成 PyTorch 张量

import pandas as pd
# ↑ Pandas：数据分析库
#   用于读取 CSV 文件、DataFrame 操作、缺失值填充等

from torch.utils.data import Dataset
# ↑ Dataset 是 PyTorch 数据加载的基类
#   所有自定义数据集都要继承它
#   DataLoader 通过调用 Dataset 的 __len__ 和 __getitem__ 来获取数据

from sklearn.preprocessing import StandardScaler, LabelEncoder
# ↑ scikit-learn 的预处理工具
#   StandardScaler：把数据变成均值=0、标准差=1 的分布
#   LabelEncoder：把类别文字转成数字（如 "Ex"→0, "Gd"→1）


class HousePricesDataset(Dataset):
    """房价数据集：从 CSV 加载，自动处理特征工程。

    和之前版本的区别：
        之前：外部做特征工程，传 numpy 数组进来
        现在：传 CSV 路径，内部自动完成所有处理

    为什么测试集要传 scaler 和 encoders？
        训练集：fit_scaler=True, fit_encoders=True → 学习数据的分布和类别映射
        测试集：传入训练集的 scaler 和 encoders → 用同样的标准处理
        如果测试集也 fit → 用了测试集的信息 → 数据泄漏 → 结果不准！

    Args:
        csv_path:      CSV 文件路径（如 "data/.../train.csv"）
        label_col:     标签列名（默认 "SalePrice"）
                         测试集没有这列，会自动跳过
        exclude_cols:  要排除的列（如 ["Id"]）
                         Id 是编号，对预测没用
        scaler:        外部传入的 StandardScaler（测试集复用训练集的）
        fit_scaler:    是否拟合 scaler（训练集 True，测试集 False）
        encoders:      外部传入的 LabelEncoder 字典（测试集复用训练集的）
        fit_encoders:  是否拟合 encoders（训练集 True，测试集 False）

    使用示例：
        # 训练集：自己学习 scaler 和 encoders
        train_set = HousePricesDataset("train.csv", fit_scaler=True, fit_encoders=True)

        # 测试集：复用训练集的 scaler 和 encoders
        test_set = HousePricesDataset("test.csv",
                                       scaler=train_set.scaler,
                                       encoders=train_set.encoders)
    """

    def __init__(self, csv_path, label_col='SalePrice', exclude_cols=None,
                 scaler=None, fit_scaler=False, encoders=None, fit_encoders=False):
        """初始化数据集：从 CSV 加载数据，完成特征工程。

        Args:
            csv_path:      CSV 文件路径
            label_col:     标签列名
            exclude_cols:  要排除的列列表
            scaler:        外部传入的 StandardScaler
            fit_scaler:    是否拟合 scaler
            encoders:      外部传入的 LabelEncoder 字典
            fit_encoders:  是否拟合 encoders
        """

        # ---- 第 1 步：读取 CSV 文件 ----
        df = pd.read_csv(csv_path)
        # ↑ 读取 CSV 文件，得到一个 DataFrame（表格）
        #   DataFrame 就像 Excel 表格：有行（样本）和列（特征）

        # ---- 第 2 步：分离标签 ----
        if label_col in df.columns:
            self.y = torch.tensor(df[label_col].values, dtype=torch.float32)
            # ↑ 如果有标签列（训练集），提取出来转成张量
            #   df[label_col].values → 取出这一列的值，变成 numpy 数组
            #   torch.tensor(...)   → 转成 PyTorch 张量
            #   dtype=torch.float32 → 32 位浮点数

            df = df.drop(columns=[label_col])
            # ↑ 从 DataFrame 中删除标签列（特征里不能包含标签）
        else:
            self.y = None
            # ↑ 测试集没有标签列（我们要预测的就是这个标签）

        # ---- 第 3 步：排除无用列 ----
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
            # ↑ 删除指定的无用列（如 Id）
            #   Id 是编号，对预测房价没有帮助
            #   如果不删除，模型会把它当特征学，反而干扰

        # ---- 第 4 步：自动识别数值列 vs 类别列 ----
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # ↑ 选出所有数值类型的列（int64、float64）
        #   例：LotArea(地块面积)、YearBuilt(建造年份) 等

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        # ↑ 选出所有字符串类型的列（object）
        #   例：MSZoning(分区类型)、Street(街道类型) 等

        # ---- 第 5 步：填充缺失值 ----
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        # ↑ 数值列的缺失值用中位数填充
        #   为什么用中位数而不是均值？
        #     中位数对异常值更稳健
        #     例：[1, 2, 3, 100] → 均值=26.5，中位数=2.5
        #     如果用均值，异常值 100 会把填充值拉高

        df[cat_cols] = df[cat_cols].fillna('Missing')
        # ↑ 类别列的缺失值用字符串 'Missing' 填充
        #   相当于把"缺失"当作一个独立的类别

        # ---- 第 6 步：类别列 → LabelEncoder 转数字 ----
        # LabelEncoder 把类别文字映射成整数
        #   例：["Ex", "Gd", "TA", "Fa"] → [0, 1, 2, 3]
        #
        # 为什么测试集要用训练集的 encoders？
        #   保证同一个类别在训练集和测试集中被编码成同一个数字
        #   如果测试集重新 fit，可能出现：训练集 "Ex"→0，测试集 "Ex"→1
        #   这就是"数据泄漏"的一种形式

        if encoders is None:
            encoders = {}
            # ↑ 第一次调用时（训练集），创建空字典

        for col in cat_cols:
            # ↑ 遍历每个类别列

            if fit_encoders or col not in encoders:
                # ↑ 训练集（fit_encoders=True）：学习映射关系
                #   或者这个列还没有 encoder（新列）

                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                # ↑ fit_transform = 学习映射 + 应用映射
                #   例：["Ex", "Gd", "TA", "Ex"] → [0, 1, 2, 0]

                encoders[col] = le
                # ↑ 保存 encoder，测试集要用
            else:
                # ↑ 测试集（fit_encoders=False）：复用训练集的映射

                le = encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                # ↑ 用训练集的 encoder 转换测试集的类别
                #   如果类别在训练集中见过 → 正常转换
                #   如果没见过（新类别）→ 编码为 -1
                #
                #   为什么要判断 x in le.classes_？
                #     测试集可能有训练集没见过的类别
                #     直接 transform 会报错
                #     用 -1 表示"未知类别"

        self.encoders = encoders
        # ↑ 保存所有 encoders，测试集创建时需要传入

        # ---- 第 7 步：数值标准化 ----
        X = df.values.astype('float32')
        # ↑ 把 DataFrame 转成 numpy 数组
        #   df.values → numpy 二维数组
        #   .astype('float32') → 转成 32 位浮点数

        if scaler is None:
            scaler = StandardScaler()
            # ↑ 第一次调用时（训练集），创建 scaler

        if fit_scaler:
            scaler.fit(X)
            # ↑ 训练集：学习数据的均值和标准差
            #   scaler.fit(X) 之后，scaler 就记住了每列的均值和标准差
            #   后续 transform 时用这些值做标准化

        X = scaler.transform(X)
        # ↑ 应用标准化：(x - 均值) / 标准差
        #   标准化后每列的均值=0，标准差=1
        #   为什么标准化？
        #     不同特征的量级差异很大（LotArea=9000, YearBuilt=2000）
        #     不标准化 → 大数值特征主导梯度 → 训练不稳定

        self.scaler = scaler
        # ↑ 保存 scaler，测试集创建时需要传入

        # ---- 第 8 步：转成 PyTorch 张量 ----
        self.X = torch.tensor(X, dtype=torch.float32)
        # ↑ 把特征矩阵从 numpy 转为 torch 张量
        #   dtype=torch.float32 → 32 位浮点数（PyTorch 默认格式）

        self.feature_names = df.columns.tolist()
        # ↑ 保存特征名称列表，main.py 用来获取特征数量

        print(f"加载完成：{len(self.X)} 条样本，{len(self.feature_names)} 个特征"
              f"（数值列 {len(num_cols)}，类别列 {len(cat_cols)}）")

    def __len__(self):
        """返回数据集的总样本数。

        DataLoader 需要知道总共有多少数据，才能：
        1. 确定每个 epoch 有多少个 batch
        2. 确定最后一个 batch 有多少条数据
        3. 计算进度条的百分比

        Returns:
            int: 总样本数
        """
        return len(self.X)

    def __getitem__(self, idx):
        """返回第 idx 条数据。

        DataLoader 会随机选一些索引（如 [3, 17, 42, ...]）
        然后调用 __getitem__ 获取对应的数据
        最后把这些数据堆叠成一个 batch

        Args:
            idx: 索引（可以是单个数字，也可以是列表/切片）

        Returns:
            训练/验证集：(特征张量, 标签张量) 元组
            测试集：特征张量（没有标签）
        """

        if self.y is not None:
            return self.X[idx], self.y[idx]
            # ↑ 有标签时返回 (特征, 标签) 元组
            #   训练集和验证集走这个分支
            #   DataLoader 会自动把多个样本的 X 和 y 分别堆叠成 batch
        else:
            return self.X[idx]
            # ↑ 没标签时只返回特征
            #   测试集走这个分支（我们要预测的就是这个标签）
