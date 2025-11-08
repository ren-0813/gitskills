import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差归一化
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FCNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout防止过拟合

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 第一个卷积层：1个输入通道，32个输出通道，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层：32个输入通道，64个输出通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 最大池化层，2x2窗口，步长2（尺寸减半）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout层防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 经过两次池化，图像尺寸为7x7
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 池化后: [batch, 32, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))  # 池化后: [batch, 64, 7, 7]
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout2(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 切换到评估模式

    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# 实例化模型
fc_model = FCNet()
cnn_model = CNN()

# 训练全连接网络
print("训练全连接神经网络:")
fc_losses = train_model(fc_model, train_loader, epochs=5)
fc_accuracy = evaluate_model(fc_model, test_loader)

# 训练卷积神经网络
print("\n训练卷积神经网络:")
cnn_losses = train_model(cnn_model, train_loader, epochs=5)
cnn_accuracy = evaluate_model(cnn_model, test_loader)

# 可视化训练过程
plt.plot(fc_losses, label='FC Network')
plt.plot(cnn_losses, label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()