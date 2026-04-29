# 引入依赖库
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from torch import nn
import os

# ============================================================================
# 开始工作前：创建输出目录、定义路径常量
# ============================================================================
os.makedirs('data/house-prices-advanced-regression-techniques/out', exist_ok=True)
MODEL_PATH = 'data/house-prices-advanced-regression-techniques/out/best_model.pt'

# 先检查一下数据结构
train_df = pd.read_csv('data/house-prices-advanced-regression-techniques/train.csv')
test_df  = pd.read_csv('data/house-prices-advanced-regression-techniques/test.csv')
print(train_df.head())
print(test_df.head())


# ============================================================================
# 0. 早停机制类
# ============================================================================
class EarlyStopping:
    """
    早停机制：当验证损失在连续 patience 个 epoch 内没有改善时，停止训练，
    并自动将最佳模型权重保存到本地文件。

    参数:
        patience (int): 允许验证损失不下降的最大 epoch 数。默认 20。
        path (str): 最佳模型权重的保存路径。
    """
    def __init__(self, patience=20, path=MODEL_PATH):
        self.patience  = patience
        self.path      = path
        self.counter   = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    @property
    def best_rmse(self):
        return self.best_loss ** 0.5


# ============================================================================
# 1. 数据集类
# ============================================================================
class HousePricesDataset(Dataset):
    def __init__(self, csv_path, label_col='SalePrice', exclude_cols=None,
                 scaler=None, fit_scaler=False, encoders=None, fit_encoders=False):
        df = pd.read_csv(csv_path)

        # 分离标签（测试集没有 SalePrice，做兼容处理）
        if label_col in df.columns:
            self.y = torch.tensor(df[label_col].values, dtype=torch.float32)
            df = df.drop(columns=[label_col])
        else:
            self.y = None

        # 排除无用列（如 Id）
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns])

        # 自动识别数值列 vs 类别列
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # 填充缺失值
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].fillna('Missing')

        # 类别列 → LabelEncoder 转数字
        if encoders is None:
            encoders = {}
        for col in cat_cols:
            if fit_encoders or col not in encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                le = encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        self.encoders = encoders

        # 数值标准化
        X = df.values.astype('float32')
        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            scaler.fit(X)
        X = scaler.transform(X)
        self.scaler = scaler

        self.X = torch.tensor(X, dtype=torch.float32)
        self.feature_names = df.columns.tolist()
        print(f"加载完成：{len(self.X)} 条样本，{len(self.feature_names)} 个特征"
              f"（数值列 {len(num_cols)}，类别列 {len(cat_cols)}）")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ============================================================================
# 2. 加载数据
# ============================================================================
train_set = HousePricesDataset(
    'data/house-prices-advanced-regression-techniques/train.csv',
    exclude_cols=['Id'],
    fit_scaler=True,
    fit_encoders=True
)
test_set = HousePricesDataset(
    'data/house-prices-advanced-regression-techniques/test.csv',
    exclude_cols=['Id'],
    scaler=train_set.scaler,
    encoders=train_set.encoders,
)

# 训练集切 8:2 作为训练/验证，测试集只用于生成提交文件
train_size = int(0.8 * len(train_set))
val_size   = len(train_set) - train_size
train_subset, val_subset = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,     batch_size=32, shuffle=False)


# ============================================================================
# 3. 模型定义
# ============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # 回归输出：1个值，无激活函数
        )

    def forward(self, x):
        return self.net(x)


input_dim = len(train_set.feature_names)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = NeuralNetwork(input_dim=input_dim).to(device)
print(f"模型已初始化，输入维度={input_dim}，运行设备={device}")


# ============================================================================
# 4. 损失函数与优化器
# ============================================================================
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============================================================================
# 5. 训练 / 验证函数
# ============================================================================
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y    = y.unsqueeze(1)           # [batch] → [batch, 1]
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)  # 返回 MSE 均值


def val_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y    = y.unsqueeze(1)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)  # 返回 MSE 均值


# ============================================================================
# 6. 训练主循环
# ============================================================================
early_stopping = EarlyStopping(patience=20, path=MODEL_PATH)

print("\n开始训练...")
for epoch in range(500):
    train_mse = train_loop(train_loader, model, loss_fn, optimizer)
    val_mse   = val_loop(val_loader,     model, loss_fn)

    train_rmse = train_mse ** 0.5   # ← 开根号才是 RMSE
    val_rmse   = val_mse   ** 0.5

    print(f"Epoch {epoch+1:>3} | 训练 RMSE: ${train_rmse:>10,.0f} | 验证 RMSE: ${val_rmse:>10,.0f}")

    early_stopping(val_mse, model)
    if early_stopping.early_stop:
        print(f"\n连续 {early_stopping.patience} 个 epoch 无改善，训练终止于 Epoch {epoch+1}。")
        print(f"最佳验证 RMSE: ${early_stopping.best_rmse:,.0f}")
        break

print(f"\n训练完成，最佳模型已保存至 {MODEL_PATH}")


# ============================================================================
# 7. 加载最佳模型 → 在验证集上评估
# ============================================================================
best_model = NeuralNetwork(input_dim=input_dim).to(device)
best_model.load_state_dict(torch.load(MODEL_PATH))
best_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in val_loader:
        pred = best_model(X.to(device)).squeeze(1).cpu()
        all_preds.append(pred)
        all_labels.append(y)

all_preds  = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
mae  = (all_preds - all_labels).abs().mean().item()
rmse = ((all_preds - all_labels) ** 2).mean().sqrt().item()
print(f"\n验证集评估结果：")
print(f"  MAE  = ${mae:,.0f}")
print(f"  RMSE = ${rmse:,.0f}")


# ============================================================================
# 8. 生成 Kaggle 提交文件
# ============================================================================
all_preds = []
with torch.no_grad():
    for X in test_loader:       # 测试集无标签，只有 X
        pred = best_model(X.to(device)).squeeze(1).cpu()
        all_preds.append(pred)

submission = pd.read_csv('data/house-prices-advanced-regression-techniques/test.csv')[['Id']]
submission['SalePrice'] = torch.cat(all_preds).numpy()
out_path = 'data/house-prices-advanced-regression-techniques/out/submission.csv'
submission.to_csv(out_path, index=False)
print(f"\n提交文件已生成：{out_path}")