import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# 数据预处理函数
def preprocess_data(df):
    # 复制数据避免修改原数据
    data = df.copy()

    # 处理缺失值
    data['bmi'] = data['bmi'].replace('N/A', np.nan)
    data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())

    # 处理分类变量
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # 选择特征列和目标列
    feature_columns = ['gender', 'age', 'hypertension', 'ever_married', 'work_type',
                       'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

    X = data[feature_columns]
    y = data['heart_disease']

    return X, y, label_encoders


# 定义神经网络模型
class HeartDiseaseClassifier(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # 将张量/数组中每个值与阈值 0.5 对比。若某位置的值大于 0.5，则该位置结果为 True（在PyTorch中等价于整数 1）；否则为 False（即 0）。
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels.unsqueeze(1)).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss / len(train_loader):.4f}, '
                  f'Val Loss: {val_loss / len(val_loader):.4f}, '
                  f'Train Acc: {train_accuracy:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


# 主函数
def main():

    df = pd.read_csv('D:\\360MoveData\\Users\\hm943\\Desktop\\healthcare-dataset-stroke-data.csv')

    # 数据预处理
    X, y, label_encoders = preprocess_data(df)

    print("心脏病预测神经网络模型")
    print("=" * 50)

    # 数据集划分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    input_size = X_train.shape[1]
    model = HeartDiseaseClassifier(input_size)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    # weight_decay是一种正则化手段，用于减少模型过拟合
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

    # 测试模型
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"\n测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, test_predictions))

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

    print("模型构建完成！")


# 如果直接运行此文件，执行主函数
if __name__ == "__main__":
    main()