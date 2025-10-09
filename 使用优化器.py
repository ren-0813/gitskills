import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time


# 从CSV文件读取数据
def load_data(filename):
    df = pd.read_csv(filename, sep='\s+', skiprows=1, header=None, names=['x', 'y'])
    # 确保数据转换为数值类型
    return df['x'].astype(float).values, df['y'].astype(float).values

# 加载数据
try:
    x_data, y_data = load_data('C:\\Users\\hm943\\Downloads\\train.csv')
    print(f"成功加载 {len(x_data)} 个数据点")
except Exception as e:
    print(f"数据加载错误: {e}")
    # 使用示例数据作为备用
    np.random.seed(42)
    x_data = np.random.rand(100, 1) * 10
    y_data = 2.5 * x_data + 3.8 + np.random.randn(100, 1) * 0.5
    print("使用生成的示例数据")

# 数据预处理：转换为二维数组便于矩阵运算
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

print(f"数据统计: x范围[{x_data.min():.2f}, {x_data.max():.2f}], y范围[{y_data.min():.2f}, {y_data.max():.2f}]")

class LinearRegression:
    def __init__(self, optimizer_type='sgd', learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):

        # 使用正态分布初始化权重和偏置
        self.w = np.random.normal(0, 0.01, (1, 1)).astype(np.float64)  # 均值0，标准差0.01
        self.b = np.random.normal(0, 0.01, (1, 1)).astype(np.float64)

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # 优化器状态变量
        self.v_w, self.v_b = 0, 0  # 动量项
        self.m_w, self.m_b = 0, 0  # Adam的一阶矩估计
        self.v_hat_w, self.v_hat_b = 0, 0  # Adam的二阶矩估计
        self.t = 0  # 时间步

    def forward(self, x):
        """前向传播：计算预测值 y = w*x + b"""
        return np.dot(x, self.w) + self.b

    def compute_loss(self, y_pred, y_true):
        """计算均方误差损失"""
        m = y_true.shape[0]
        return np.sum((y_pred - y_true) ** 2) / (2 * m)

    def compute_gradients(self, x, y_true, y_pred):
        """计算梯度"""
        m = y_true.shape[0]
        dw = np.dot(x.T, (y_pred - y_true)) / m
        db = np.sum(y_pred - y_true, axis=0, keepdims=True) / m
        return dw, db

    def sgd_step(self, dw, db):
        """标准SGD优化器更新"""
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def momentum_step(self, dw, db):
        """带动量的SGD优化器更新"""
        self.v_w = self.momentum * self.v_w + self.learning_rate * dw
        self.v_b = self.momentum * self.v_b + self.learning_rate * db
        self.w -= self.v_w
        self.b -= self.v_b

    def adam_step(self, dw, db):
        """Adam优化器更新"""
        self.t += 1

        # 更新一阶矩估计
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db

        # 更新二阶矩估计
        self.v_hat_w = self.beta2 * self.v_hat_w + (1 - self.beta2) * (dw ** 2)
        self.v_hat_b = self.beta2 * self.v_hat_b + (1 - self.beta2) * (db ** 2)

        # 偏差校正
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_hat_w / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_hat_b / (1 - self.beta2 ** self.t)

        # 更新参数
        self.w -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def backward(self, x, y_true, y_pred):
        """反向传播：根据选择的优化器更新参数"""
        dw, db = self.compute_gradients(x, y_true, y_pred)

        if self.optimizer_type == 'sgd':
            self.sgd_step(dw, db)
        elif self.optimizer_type == 'momentum':
            self.momentum_step(dw, db)
        elif self.optimizer_type == 'adam':
            self.adam_step(dw, db)

        return dw, db


def train_model(optimizer_type, x_data, y_data, learning_rate=0.01, epochs=1000):
    """训练模型并返回训练历史"""
    print(f"\n=== 使用 {optimizer_type.upper()} 优化器训练 ===")

    # 创建模型实例
    model = LinearRegression(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=0.9 if optimizer_type == 'momentum' else 0,
        beta1=0.9,
        beta2=0.999
    )

    # 记录训练过程
    history = {
        'loss': [],
        'w': [],
        'b': [],
        'time': 0
    }

    start_time = time()

    print(f"初始参数: w={model.w[0][0]:.4f}, b={model.b[0][0]:.4f}")

    for epoch in range(epochs):
        # 前向传播
        y_pred = model.forward(x_data)

        # 计算损失
        current_loss = model.compute_loss(y_pred, y_data)

        # 反向传播并更新参数
        dw, db = model.backward(x_data, y_data, y_pred)

        # 记录历史
        history['loss'].append(current_loss)
        history['w'].append(model.w[0][0])
        history['b'].append(model.b[0][0])

        # 打印训练进度
        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {current_loss:.6f}, w = {model.w[0][0]:.4f}, b = {model.b[0][0]:.4f}")

    history['time'] = time() - start_time
    print(f"训练时间: {history['time']:.2f}秒")
    print(f"最终参数: w={model.w[0][0]:.4f}, b={model.b[0][0]:.4f}")
    print(f"最终损失: {history['loss'][-1]:.6f}")

    return model, history


# 超参数设置
epochs = 1000
learning_rate = 0.01

# 训练三种不同的优化器
optimizers = ['sgd', 'momentum', 'adam']
models = {}
histories = {}

for optimizer in optimizers:
    model, history = train_model(optimizer, x_data, y_data, learning_rate, epochs)
    models[optimizer] = model
    histories[optimizer] = history

# 可视化比较三种优化器的性能
plt.figure(figsize=(18, 12))

# 1. 损失函数下降曲线对比
plt.subplot(2, 3, 1)
for optimizer in optimizers:
    plt.plot(histories[optimizer]['loss'], label=f'{optimizer.upper()}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 使用对数坐标更好地观察损失下降

# 2. 参数w的变化对比
plt.subplot(2, 3, 2)
for optimizer in optimizers:
    plt.plot(histories[optimizer]['w'], label=f'{optimizer.upper()}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight (w)')
    plt.title('Weight Evolution Comparison')
    plt.legend()
    plt.grid(True)

# 3. 参数b的变化对比
plt.subplot(2, 3, 3)
for optimizer in optimizers:
    plt.plot(histories[optimizer]['b'], label=f'{optimizer.upper()}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Bias (b)')
    plt.title('Bias Evolution Comparison')
    plt.legend()
    plt.grid(True)

# 4. 收敛速度对比（前100个epoch）
plt.subplot(2, 3, 4)
for optimizer in optimizers:
    plt.plot(histories[optimizer]['loss'][:100], label=f'{optimizer.upper()}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Early Training Convergence (First 100 Epochs)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

# 5. 最终拟合效果对比
plt.subplot(2, 3, 5)
plt.scatter(x_data.flatten(), y_data.flatten(), color='black', label='True data', alpha=0.6)

colors = ['red', 'blue', 'green']
for i, optimizer in enumerate(optimizers):
    x_test = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
    y_test = models[optimizer].forward(x_test)
    plt.plot(x_test.flatten(), y_test.flatten(), color=colors[i], linewidth=2,
             label=f'{optimizer.upper()} (w={models[optimizer].w[0][0]:.3f}, b={models[optimizer].b[0][0]:.3f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression Lines Comparison')
plt.legend()
plt.grid(True)

# 6. 性能统计
plt.subplot(2, 3, 6)
optimizer_names = [opt.upper() for opt in optimizers]
final_losses = [histories[opt]['loss'][-1] for opt in optimizers]
training_times = [histories[opt]['time'] for opt in optimizers]

x_pos = np.arange(len(optimizer_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

bars1 = ax1.bar(x_pos - width / 2, final_losses, width, label='Final Loss', alpha=0.7, color='skyblue')
bars2 = ax2.bar(x_pos + width / 2, training_times, width, label='Training Time (s)', alpha=0.7, color='lightcoral')

ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Final Loss')
ax2.set_ylabel('Training Time (seconds)')
ax1.set_title('Optimizer Performance Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(optimizer_names)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 在柱状图上添加数值标签
for bar, value in zip(bars1, final_losses):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}',
             ha='center', va='bottom', fontsize=9)

for bar, value in zip(bars2, training_times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}s',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 性能总结
print("\n" + "=" * 50)
print("优化器性能总结")
print("=" * 50)

for optimizer in optimizers:
    history = histories[optimizer]
    print(f"\n{optimizer.upper()}优化器:")
    print(f"  最终损失: {history['loss'][-1]:.6f}")
    print(f"  训练时间: {history['time']:.2f}秒")
    print(f"  最终参数: w={models[optimizer].w[0][0]:.4f}, b={models[optimizer].b[0][0]:.4f}")
    print(f"  收敛epoch数: {np.argmin(history['loss'])}")


# 预测新数据
def predict(x_new, model):
    """使用训练好的模型预测新数据"""
    x_new = np.array([[x_new]])  # 转换为二维数组
    return model.forward(x_new)[0][0]


# 测试预测
print("\n=== 预测测试 ===")
test_x = np.mean(x_data)  # 使用数据均值进行测试
for optimizer in optimizers:
    predicted_y = predict(test_x, models[optimizer])
    print(f"{optimizer.upper()}预测: x={test_x:.2f}, y_pred={predicted_y:.4f}")

print("\n=== 优化器特点比较 ===")
print("SGD: 简单但可能收敛慢，容易震荡[1](@ref)")
print("Momentum: 加入动量项，加速收敛，减少震荡")
print("Adam: 自适应学习率，通常收敛最快且稳定[6](@ref)")