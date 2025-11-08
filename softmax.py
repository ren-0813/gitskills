import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor，并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 2. 定义神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 10)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. 训练模型
def train_model(model, train_loader, epochs=10):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零
            output = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch + 1} 平均损失: {avg_loss:.4f}')

    return train_losses


print("开始训练模型...")
train_losses = train_model(model, train_loader)


# 4. 测试模型
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\n测试集性能: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


accuracy = test_model(model, test_loader)

# 5. 可视化结果
# 绘制训练损失
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 显示一些预测结果
model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    output = model(data)
    pred = output.argmax(dim=1)

plt.subplot(1, 2, 2)
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(data[i][0], cmap='gray')
    ax.set_title(f'真实:{target[i]}, 预测:{pred[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 6. 保存模型
torch.save(model.state_dict(), 'mnist_fc_model.pth')
print("模型已保存为 'mnist_fc_model.pth'")