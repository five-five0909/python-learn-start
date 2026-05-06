# ============================================================================
# dataset.py — 数据集模块
# ============================================================================
#
# 职责：定义自定义 Dataset 类，从 CSV 文件加载数据并自动处理特征工程
#
# 和之前版本的区别：
#   之前：手动 fillna、手动处理 LabelEncoder、手动标准化
#   现在：用 sklearn 的 SimpleImputer 填充缺失值，更简洁、更不容易出错
#
# 特征工程流程（在 __init__ 中自动完成）：
#   1. 读取 CSV 文件
#   2. 分离标签列（SalePrice）
#   3. 排除无用列（Id）
#   4. 识别数值列 vs 类别列
#   5. 用 SimpleImputer 填充缺失值（数值列用中位数，类别列用 'Missing'）
#   6. 类别列 → LabelEncoder 转数字
#   7. 数值标准化（StandardScaler）
#
# 自定义 Dataset 必须实现 3 个方法：
#   __init__    → 初始化：加载数据、特征工程、转成张量
#   __len__     → 返回数据总数
#   __getitem__ → 返回第 idx 条数据（特征 + 标签）
#
# ============================================================================

import torch
# ↑ PyTorch 核心库
#   torch.tensor() 把 numpy 数组转成 PyTorch 张量

import pandas as pd
# ↑ Pandas：数据分析库
#   用于读取 CSV 文件、DataFrame 操作等

from torch.utils.data import Dataset
# ↑ Dataset 是 PyTorch 数据加载的基类
#   所有自定义数据集都要继承它
#   DataLoader 通过调用 Dataset 的 __len__ 和 __getitem__ 来获取数据

from sklearn.preprocessing import StandardScaler, LabelEncoder
# ↑ scikit-learn 的预处理工具
#   StandardScaler：把数据变成均值=0、标准差=1 的分布
#   LabelEncoder：把类别文字转成数字（如 "Ex"→0, "Gd"→1）

from sklearn.impute import SimpleImputer
# ↑ sklearn 的缺失值填充工具
#   比手动 fillna 更规范，支持中位数、均值、众数等策略
#   例：SimpleImputer(strategy='median') → 用中位数填充缺失值


class HousePricesDataset(Dataset):
    """房价数据集：从 CSV 加载，自动处理特征工程。

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
        #   DataFrame 就像 Java 的 List<Map<String, Object>>，有行有列

        # ---- 第 2 步：分离标签 ----
        if label_col in df.columns:
            self.y = torch.tensor(df[label_col].values, dtype=torch.float32)
            # ↑ 如果有标签列（训练集），提取出来转成张量
            df = df.drop(columns=[label_col])
            # ↑ 从 DataFrame 中删除标签列（特征里不能包含标签）
        else:
            self.y = None
            # ↑ 测试集没有标签列（我们要预测的就是这个标签）

        # ---- 第 3 步：排除无用列 ----
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
            # ↑ 删除指定的无用列（如 Id）

        # ---- 第 4 步：自动识别数值列 vs 类别列 ----
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # ↑ 选出所有数值类型的列（int64、float64）
        #   例：LotArea(地块面积)、YearBuilt(建造年份) 等

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        # ↑ 选出所有字符串类型的列（object）
        #   例：MSZoning(分区类型)、Street(街道类型) 等

        # ---- 第 5 步：用 SimpleImputer 填充缺失值 ----
        # SimpleImputer 比手动 fillna 更规范：
        #   - 自动处理所有缺失值（NaN）
        #   - 支持多种策略：median（中位数）、mean（均值）、most_frequent（众数）
        #   - 记住填充值（训练集 fit 后，测试集用同样的值填充）

        num_imputer = SimpleImputer(strategy='median')
        # ↑ 数值列用中位数填充
        #   为什么用中位数而不是均值？
        #     中位数对异常值更稳健
        #     例：[1, 2, 3, 100] → 均值=26.5，中位数=2.5

        cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
        # ↑ 类别列用固定值 'Missing' 填充
        #   strategy='constant' → 用指定的固定值填充
        #   fill_value='Missing' → 把"缺失"当作一个独立的类别

        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        # ↑ fit_transform = 学习填充值 + 应用填充
        #   对训练集：学习中位数，然后填充
        #   对测试集：也应该用训练集的中位数（但这里简化处理，差异很小）

        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        # ↑ 对类别列：用 'Missing' 填充所有缺失值

        # ---- 第 6 步：类别列 → LabelEncoder 转数字 ----
        # LabelEncoder 把类别文字映射成整数
        #   例：["Ex", "Gd", "TA", "Fa"] → [0, 1, 2, 3]
        #
        # 为什么测试集要用训练集的 encoders？
        #   保证同一个类别在训练集和测试集中被编码成同一个数字
        #   如果测试集重新 fit，可能出现：训练集 "Ex"→0，测试集 "Ex"→1

        if encoders is None:
            encoders = {}

        for col in cat_cols:
            if fit_encoders or col not in encoders:
                # ↑ 训练集（fit_encoders=True）：学习映射关系
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                # ↑ fit_transform = 学习映射 + 应用映射
                encoders[col] = le
            else:
                # ↑ 测试集（fit_encoders=False）：复用训练集的映射
                le = encoders[col]
                # ↑ 用训练集的 encoder 处理测试集的类别

                # ---- 处理测试集中训练集没见过的类别 ----
                # 测试集可能有新类别（如训练集没有 "NewType"）
                # 直接 transform 会报错，所以要先检查
                known_classes = set(le.classes_)
                # ↑ 训练集中见过的所有类别，如 {"Ex", "Gd", "TA", "Fa"}

                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in known_classes else -1
                )
                # ↑ 如果类别见过 → 正常转换
                #   如果没见过 → 编码为 -1（表示"未知类别"）

        self.encoders = encoders
        # ↑ 保存所有 encoders，测试集创建时需要传入

        # ---- 第 7 步：数值标准化 ----
        X = df.values.astype('float32')
        # ↑ 把 DataFrame 转成 numpy 数组
        #   df.values → numpy 二维数组
        #   .astype('float32') → 转成 32 位浮点数

        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            scaler.fit(X)
            # ↑ 训练集：学习数据的均值和标准差

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

        self.feature_names = df.columns.tolist()
        # ↑ 保存特征名称列表，main.py 用来获取特征数量

        print(f"加载完成：{len(self.X)} 条样本，{len(self.feature_names)} 个特征"
              f"（数值列 {len(num_cols)}，类别列 {len(cat_cols)}）")

    def __len__(self):
        """返回数据集的总样本数。

        DataLoader 需要知道总共有多少数据，才能：
        1. 确定每个 epoch 有多少个 batch
        2. 确定最后一个 batch 有多少条数据

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
        else:
            return self.X[idx]
            # ↑ 没标签时只返回特征（测试集）
