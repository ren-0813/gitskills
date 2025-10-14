import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 加载数据集
try:
    # 假设train.csv中包含特征列'x'和目标列'y'
    df = pd.read_csv('train.csv')
    x_data = df['x'].values
    y_data = df['y'].values
except Exception as e:
    print(f"加载数据集失败: {e}")
    # 若加载失败，使用示例数据
    x_data = np.linspace(1, 10, 100)
    y_data = 2 * x_data + 3 + np.random.normal(0, 1, 100)

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x_data).unsqueeze(1)
y_tensor = torch.FloatTensor(y_data).unsqueeze(1)
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 2. 定义线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 正态分布初始化权重和偏置
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0, std=0.1)

    def forward(self, x):
        return self.linear(x)


# 3. 训练函数
def train_model(optimizer, epochs=100, lr=0.01):
    model = LinearModel()
    criterion = nn.MSELoss()

    # 根据选择的优化器初始化
    if optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=lr)
    elif optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(), lr=lr)

    # 记录训练过程
    w_history = []
    b_history = []
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            opt.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # 记录参数和损失
        w_history.append(model.linear.weight.item())
        b_history.append(model.linear.bias.item())
        loss_history.append(total_loss / len(dataloader))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_history[-1]:.4f}')

    return model, w_history, b_history, loss_history


# 4. 三种优化器对比
optimizers = ['sgd', 'adam', 'rmsprop']
results = {}

for opt in optimizers:
    print(f"\n使用{opt}优化器训练:")
    model, w_hist, b_hist, loss_hist = train_model(opt, epochs=100, lr=0.01)
    results[opt] = {
        'w': w_hist,
        'b': b_hist,
        'loss': loss_hist
    }

# 5. 可视化优化器性能对比
plt.figure(figsize=(12, 5))

# 损失对比
plt.subplot(1, 2, 1)
for opt in optimizers:
    plt.plot(results[opt]['loss'], label=opt)
plt.title('不同优化器的损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 权重变化对比
plt.subplot(1, 2, 2)
for opt in optimizers:
    plt.plot(results[opt]['w'], label=opt)
plt.title('不同优化器的权重变化')
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.legend()
plt.tight_layout()
plt.show()

# 6. 参数w和b的调节过程可视化
plt.figure(figsize=(12, 5))

# 权重w的变化
plt.subplot(1, 2, 1)
for opt in optimizers:
    plt.plot(results[opt]['w'], label=opt)
plt.title('权重w的调节过程')
plt.xlabel('Epoch')
plt.ylabel('w值')
plt.legend()

# 偏置b的变化
plt.subplot(1, 2, 2)
for opt in optimizers:
    plt.plot(results[opt]['b'], label=opt)
plt.title('偏置b的调节过程')
plt.xlabel('Epoch')
plt.ylabel('b值')
plt.legend()
plt.tight_layout()
plt.show()

# 7. 不同学习率对比
lrs = [0.001, 0.01, 0.1]
lr_results = {}

for lr in lrs:
    print(f"\n使用学习率{lr}训练:")
    model, w_hist, b_hist, loss_hist = train_model('adam', epochs=100, lr=lr)
    lr_results[lr] = loss_hist

plt.figure(figsize=(10, 6))
for lr in lrs:
    plt.plot(lr_results[lr], label=f'学习率={lr}')
plt.title('不同学习率的损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8. 不同迭代次数对比
epochs_list = [50, 100, 200]
epoch_results = {}

for epochs in epochs_list:
    print(f"\n使用{epochs}次迭代训练:")
    model, w_hist, b_hist, loss_hist = train_model('adam', epochs=epochs, lr=0.01)
    epoch_results[epochs] = loss_hist

plt.figure(figsize=(10, 6))
for epochs in epochs_list:
    plt.plot(epoch_results[epochs], label=f'迭代次数={epochs}')
plt.title('不同迭代次数的损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 9. 参数调节动态可视化
def animate_param_changes(param_history, param_name):
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(param_history))
    ax.set_ylim(min(param_history) * 0.9, max(param_history) * 1.1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(param_name)
    ax.set_title(f'{param_name}参数调节动态过程')

    line, = ax.plot([], [], 'b-')

    def update(frame):
        line.set_data(range(frame + 1), param_history[:frame + 1])
        return line,

    anim = FuncAnimation(fig, update, frames=len(param_history),
                         interval=50, blit=True)
    return anim


# 以Adam优化器为例展示动态过程
adam_w = results['adam']['w']
adam_b = results['adam']['b']

print("生成权重w的动态调节过程...")
w_anim = animate_param_changes(adam_w, '权重w')
plt.show()

print("生成偏置b的动态调节过程...")
b_anim = animate_param_changes(adam_b, '偏置b')
plt.show()