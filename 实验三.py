import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('train.csv')

if data.isnull().values.any():
    data = data.fillna(0) 

if np.isinf(data.values).any():
    data = data.replace([np.inf, -np.inf], np.finfo(np.float32).max)  

X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

input_size = X.shape[1]
model = LinearModel(input_size)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

epochs = 100
w_history = []
loss_history = []

for epoch in range(epochs):

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    w = model.linear.weight.detach().numpy().copy()
    w_history.append(w)
    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

w_history = np.array(w_history)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
for i in range(w_history.shape[1]):
    plt.plot(w_history[:, 0, i], label=f'w_{i}')
plt.title('权重 (w) 的变化')
plt.xlabel('轮次')
plt.ylabel('权重值')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(loss_history, label='损失')
plt.title('损失的变化')
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.legend()

plt.tight_layout()
plt.show()