# ============================================================================
# house_prices 包
# ============================================================================
#
# Kaggle House Prices Advanced Regression Techniques 竞赛的模块化实现
#
# 包结构：
#   __init__.py       → 包标识文件（让 Python 把这个文件夹当作"包"来导入）
#   config.py         → 全局配置：超参数、路径、设备选择、随机种子
#   dataset.py        → 数据集：从 CSV 加载，自动特征工程（LabelEncoder + 标准化）
#   model.py          → 模型：3 层 MLP（256 → 64 → 1）
#   checkpoint.py     → 早停机制 + 最佳模型自动保存
#   metrics.py        → 评估指标：MSE、RMSE、MAE 的计算函数
#   trainer.py        → 训练循环：train_loop（训练一个 epoch）、val_loop（验证一个 epoch）
#   main.py           → 主程序入口：数据加载 → 建模 → 训练 → 评估 → 生成提交文件
#
# 使用方式：
#   在项目根目录下运行：
#   python -m house_prices.main
#
# 依赖库：
#   torch, numpy, pandas, scikit-learn
#
# ============================================================================

# __init__.py 的作用：
#   1. 告诉 Python："这个文件夹是一个 Python 包"
#   2. 没有 __init__.py → Python 无法用 import 从这个文件夹导入模块
#   3. 可以为空，也可以在这里做包级别的初始化（本项目留空即可）
#
# 什么是"包"（Package）？
#   包 = 包含 __init__.py 的文件夹
#   模块 = 一个 .py 文件
#   包可以包含多个模块，实现代码的分层组织
#
# 导入示例：
#   from house_prices.config import SEED           → 从 config 模块导入 SEED 常量
#   from house_prices.model import NeuralNetwork   → 从 model 模块导入 NeuralNetwork 类
#   from house_prices.trainer import train_loop    → 从 trainer 模块导入 train_loop 函数
