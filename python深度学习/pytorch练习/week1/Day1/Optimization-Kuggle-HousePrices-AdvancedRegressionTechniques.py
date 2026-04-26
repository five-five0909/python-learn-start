# 引入依赖库
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 全局配置
# ============================================================================
SEED            = 42
DATA_DIR        = "data/house-prices-advanced-regression-techniques"
TRAIN_PATH      = os.path.join(DATA_DIR, "train.csv")
TEST_PATH       = os.path.join(DATA_DIR, "test.csv")
OUT_DIR         = os.path.join(DATA_DIR, "out")
MODEL_PATH      = os.path.join(OUT_DIR, "best_model.pt")
SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")
BATCH_SIZE      = 64
EPOCHS          = 500
PATIENCE        = 40
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
VAL_RATIO       = 0.2
device          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# 0. 早停机制类
# ============================================================================
class EarlyStopping:
    def __init__(self, patience=PATIENCE, path=MODEL_PATH, min_delta=1e-6):
        self.patience   = patience
        self.path       = path
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    @property
    def best_log_rmse(self):
        return self.best_loss ** 0.5

# ============================================================================
# 1. 读取数据 & 特征工程
# ============================================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(train_df.head())
print(test_df.head())

# 保存测试集 Id，生成提交文件时用
test_ids = test_df["Id"].copy()

# 标签做 log1p 变换（对应 Kaggle log RMSE 评分方式）
y_train = np.log1p(train_df["SalePrice"].values).astype(np.float32)

# 删除无用列
train_features = train_df.drop(columns=["SalePrice", "Id"])
test_features  = test_df.drop(columns=["Id"])

# 自动识别数值列 vs 类别列
num_cols = train_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = train_features.select_dtypes(include=["object"]).columns.tolist()

# 缺失值：数值列用训练集中位数，类别列填 'Missing'
medians = train_features[num_cols].median().fillna(0)
train_features[num_cols] = train_features[num_cols].fillna(medians)
test_features[num_cols]  = test_features[num_cols].fillna(medians)
train_features[cat_cols] = train_features[cat_cols].fillna("Missing")
test_features[cat_cols]  = test_features[cat_cols].fillna("Missing")

# 拼接后统一 One-Hot，保证训练/测试特征列完全一致
combined    = pd.concat([train_features, test_features], axis=0, ignore_index=True)
combined    = pd.get_dummies(combined, columns=cat_cols).astype(np.float32)
X_train_raw = combined.iloc[:len(train_df)].values
X_test_raw  = combined.iloc[len(train_df):].values

# 标准化（只在训练集上 fit）
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)

print(f"特征工程完成：训练集 {X_train.shape}，测试集 {X_test.shape}")

# ============================================================================
# 2. 构建 DataLoader
# ============================================================================
full_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
)
val_size   = int(len(full_dataset) * VAL_RATIO)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(
    torch.tensor(X_test, dtype=torch.float32),
    batch_size=BATCH_SIZE, shuffle=False
)

# ============================================================================
# 3. 模型定义
# ============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # 回归输出，不加激活函数
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
model     = NeuralNetwork(input_dim=input_dim).to(device)
print(f"模型初始化完成，输入维度={input_dim}，设备={device}")

# ============================================================================
# 4. 损失函数、优化器、学习率调度器
# ============================================================================
loss_fn   = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

# ============================================================================
# 5. 训练与验证函数
# ============================================================================
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

def val_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            total_loss += loss_fn(model(X), y).item() * X.size(0)
    return total_loss / len(dataloader.dataset)

# ============================================================================
# 6. 训练主循环
# ============================================================================
early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_PATH)

print("\n开始训练...")
for epoch in range(1, EPOCHS + 1):
    train_mse = train_loop(train_loader, model, loss_fn, optimizer)
    val_mse   = val_loop(val_loader,     model, loss_fn)
    scheduler.step(val_mse)

    print(f"Epoch {epoch:>3} | 训练 logRMSE: {train_mse**0.5:.5f} | 验证 logRMSE: {val_mse**0.5:.5f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

    early_stopping(val_mse, model)
    if early_stopping.early_stop:
        print(f"连续 {PATIENCE} 个 epoch 无改善，训练终止于 Epoch {epoch}。")
        print(f"最佳验证 logRMSE: {early_stopping.best_log_rmse:.5f}")
        break

print(f"\n训练完成，最佳模型已保存至：{MODEL_PATH}")

# ============================================================================
# 7. 加载最佳模型，在验证集上评估
# ============================================================================
best_model = NeuralNetwork(input_dim=input_dim).to(device)
best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
best_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in val_loader:
        all_preds.append(best_model(X.to(device)).squeeze(1).cpu())
        all_labels.append(y.squeeze(1))

all_preds  = np.clip(np.expm1(torch.cat(all_preds).numpy()), 0, None)
all_labels = np.expm1(torch.cat(all_labels).numpy())

mae  = np.mean(np.abs(all_preds - all_labels))
rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
print(f"\n验证集评估结果：")
print(f"  MAE  = ${mae:,.0f}")
print(f"  RMSE = ${rmse:,.0f}")

# ============================================================================
# 8. 生成 Kaggle 提交文件
# ============================================================================
all_test_preds = []
with torch.no_grad():
    for X in test_loader:
        all_test_preds.append(best_model(X.to(device)).squeeze(1).cpu().numpy())

all_test_preds = np.clip(np.expm1(np.concatenate(all_test_preds)), 0, None)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": all_test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\n提交文件已生成：{SUBMISSION_PATH}")
print(submission.head())