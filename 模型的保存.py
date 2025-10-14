import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


class EcologicalFootprintDataset(Dataset):
    def __init__(self, filepath, target_column=-1):
        data = pd.read_csv(filepath)

        # 更严格的清洗：移除非数值列和无效值
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        xy = numeric_data.values.astype(np.float32)

        # 检查目标列
        target_idx = target_column if target_column >= 0 else xy.shape[1] + target_column
        assert -xy.shape[1] <= target_idx < xy.shape[1], "目标列索引无效"

        # 标准化特征和标签
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        x_data = xy[:, [i for i in range(xy.shape[1]) if i != target_idx]]
        x_data = self.x_scaler.fit_transform(x_data)
        y_data = self.y_scaler.fit_transform(xy[:, [target_idx]])

        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
        self.len = xy.shape[0]
        # 处理目标列
        if isinstance(target_column, str):
            target_idx = numeric_data.columns.get_loc(target_column)
        else:
            target_idx = target_column if target_column >= 0 else xy.shape[1] + target_column

        # 分离特征和标签
        self.scaler = StandardScaler()
        feature_indices = [i for i in range(xy.shape[1]) if i != target_idx]
        x_data = xy[:, feature_indices]
        x_data = self.scaler.fit_transform(x_data)
        y_data = xy[:, [target_idx]]

        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class FiveLayerNN(nn.Module):
    def __init__(self, input_size):
        super(FiveLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 7)  # 输入层到第一隐藏层
        self.layer2 = nn.Linear(7, 6)  # 第一隐藏层到第二隐藏层
        self.layer3 = nn.Linear(6, 5)  # 第二隐藏层到第三隐藏层
        self.layer4 = nn.Linear(5, 1)  # 第三隐藏层到输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)  # 回归问题，输出层不使用激活函数
        return x


def train_model():
    # 初始化数据集和数据加载器
    dataset = EcologicalFootprintDataset('D:\\360MoveData\\Users\\hm943\\Desktop\\countries.csv', target_column=-11)
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)

    # 初始化模型
    model = FiveLayerNN(dataset.x_data.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    num_epochs = 200
    train_losses = []
    best_loss = float('inf')

    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 记录平均损失
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pt')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 确保目录存在
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练完成! 最佳模型已保存为 'best_model.pt'")
    print(f"训练曲线已保存为 'results/training_curve.png'")


if __name__ == '__main__':
    train_model()