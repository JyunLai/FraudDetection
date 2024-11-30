import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import wandb

wandb.init(project="Data_Science")


model_path = 'Detection.ckpt'
predict_csv_path = 'Detection_result.csv'
# 讀取數據
data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# 取得 fraud == 1 和 fraud == 0 的資料
fraud_1 = data[data['fraud'] == 1]
fraud_0 = data[data['fraud'] == 0]

# 計算最常見的類別數量（以 fraud == 0 的數量為基準）
majority_class_count = len(fraud_0)

# 過採樣 fraud == 1 的資料，複製至與 fraud == 0 類別數量相同
fraud_1_oversampled = fraud_1.sample(majority_class_count, replace=True, random_state=42)

# 將原資料與過採樣資料合併
data = pd.concat([fraud_0, fraud_1_oversampled])


X = data.drop("fraud", axis=1)
X["att8"] = X["att8"].apply(lambda x: x[4:])
X["att9"] = X["att9"].apply(lambda x: x[5:])
X["att14"] = X["att14"].apply(lambda x: x[8:])
test = test_data.drop("Id", axis=1)
test["att8"] = test["att8"].apply(lambda x: x[4:])
test["att9"] = test["att9"].apply(lambda x: x[5:])
test["att14"] = test["att14"].apply(lambda x: x[8:])

y = data["fraud"].values


# 將 att1 的時間區分成「小時」、「分」兩個欄位
def split_time_feature(X, time_column):
    X_augmented = X.copy()
    X_augmented["hour"] = X[time_column].str.split(":").str[0].astype(int)
    X_augmented["minute"] = X[time_column].str.split(":").str[1].astype(int)
    X_augmented.drop(columns=[time_column], inplace=True)
    return X_augmented

X = split_time_feature(X, "att1")
test = split_time_feature(test, "att1")

# 對 att3 使用 LabelEncoder
label_encoder = LabelEncoder()
X["att3"] = label_encoder.fit_transform(X["att3"])
test["att3"] = label_encoder.transform(test["att3"])


# 數值與類別特徵
numerical_features = ["att4", "att5", "att10", "att12", "att13", "att15", "att16"]
categorical_features = ["att6", "att7", "att8", "att9", "att14"]

# 數據預處理
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])
test_numerical = scaler.transform(test[numerical_features])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test)


# 將處理好的 att3 特徵與其餘特徵合併
X_processed = np.hstack([X["att3"].values.reshape(-1, 1), X_processed.toarray()])
test_processed = np.hstack([test["att3"].values.reshape(-1, 1), test_processed.toarray()])

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


# 使用 SMOTE 增強Training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 轉換為 PyTorch 張量
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# 定義Network
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(128, 16), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(16, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 初始化模型
input_size = X_train_tensor.shape[1]
model = FraudDetectionModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)


# 訓練模型
epochs = 1000
batch_size = 64
best_acc = 0
best_loss = 1000
first = 1
early_stop = 200

for epoch in range(epochs):
    if first != 1:
        model.load_state_dict(torch.load(model_path))
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # 前向傳播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每個 epoch 的損失輸出
    print(f"Epoch {epoch+1}/{epochs}, Train_Loss: {loss.item()}")

    # 測試模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_classes = (y_pred > 0.5).float()
        val_loss = criterion(y_pred, y_test_tensor)
        accuracy = (y_pred_classes == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Epoch {epoch+1}/{epochs}, Valid_Loss: {val_loss.item()}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    if val_loss < best_loss or (val_loss == best_loss and accuracy > best_acc):
        best_acc = accuracy
        best_loss = val_loss
        torch.save(model.state_dict(), model_path)
        early_stop = 50
    else:
        early_stop -= 1
    if early_stop <= 0:
        print("early stop")
        break
    if first == 1:
        torch.save(model.state_dict(), model_path)
        first = 0
    scheduler.step()
    wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "accuracy": accuracy})
wandb.finish()


test_tensor = torch.tensor(test_processed, dtype=torch.float32)
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    test_pred = model(test_tensor)
    test_pred_classes = (test_pred > 0.5).float()

with open(predict_csv_path, "w") as f:

    f.write("Id,fraud\n")
    for index in range(len(test_pred_classes)):
        f.write(f"{index+1},{int(test_pred_classes[index].item())}\n")