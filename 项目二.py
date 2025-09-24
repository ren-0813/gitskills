import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\HuaweiMoveData\Users\86158\Desktoptrain.csv")

df_clean = df.dropna(subset=["x", "y"])
x_data = df_clean["x"].values
y_data = df_clean["y"].values
print(f"成功读取数据，共{len(x_data)}个有效样本")


def forward(x, w, b):
    """正向传播：计算模型预测值 y_pred = w*x + b"""
    return w * x + b

def loss(x, y, w, b):
    """计算损失：使用均方误差（MSE），loss = (y_pred - y)² 的平均值"""
    y_pred = forward(x, w, b)
    return np.mean((y_pred - y) ** 2)

w_list = np.arange(-2.0, 4.1, 0.1)
w_loss_list = []

for w in w_list:
    current_loss = loss(x_data, y_data, w, b=0)
    w_loss_list.append(current_loss)

b_list = np.arange(-10.0, 10.1, 0.5)
b_loss_list = []

for b in b_list:
    current_loss = loss(x_data, y_data, w=1, b=b)
    b_loss_list.append(current_loss)

plt.figure(figsize=(10, 12))

plt.subplot(2, 1, 1)
plt.plot(w_list, w_loss_list, color="#1f77b4", linewidth=2)
plt.xlabel("Weight (w)", fontsize=12)
plt.ylabel("Mean Squared Error (Loss)", fontsize=12)
plt.title("(1) Weight (w) vs Loss Relationship", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(b_list, b_loss_list, color="#ff7f0e", linewidth=2)
plt.xlabel("Bias (b)", fontsize=12)
plt.ylabel("Mean Squared Error (Loss)", fontsize=12)
plt.title("(2) Bias (b) vs Loss Relationship", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_w = w_list[np.argmin(w_loss_list)]
best_b = b_list[np.argmin(b_loss_list)]
min_w_loss = np.min(w_loss_list)
min_b_loss = np.min(b_loss_list)

print(f"\n最优参数（固定b=0时）：w={best_w:.2f}，最小损失={min_w_loss:.4f}")
print(f"最优参数（固定w=1时）：b={best_b:.2f}，最小损失={min_b_loss:.4f}")