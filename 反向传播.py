import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 从CSV文件读取数据
def load_data(filename):
    # 使用推荐的sep参数代替已弃用的delim_whitespace
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
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
    print("使用示例数据")

# 数据预处理：转换为二维数组便于矩阵运算
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)


# 简单的线性回归模型（使用反向传播）
class LinearRegression:
    def __init__(self):
        # 随机初始化参数
        self.w = np.random.randn(1, 1) * 0.01
        self.b = np.zeros((1, 1))

    def forward(self, x):
        """前向传播：计算预测值 y = w*x + b"""
        return np.dot(x, self.w) + self.b

    def compute_loss(self, y_pred, y_true):
        """计算均方误差损失"""
        m = y_true.shape[0]
        return np.sum((y_pred - y_true) ** 2) / (2 * m)

    def backward(self, x, y_true, y_pred, learning_rate):
        """反向传播：计算梯度并更新参数[1](@ref)"""
        m = y_true.shape[0]

        # 计算梯度[1,6](@ref)
        dw = np.dot(x.T, (y_pred - y_true)) / m
        db = np.sum(y_pred - y_true, axis=0, keepdims=True) / m

        # 更新参数[1](@ref)
        self.w = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db

        return dw, db


# 超参数设置
learning_rate = 0.01
epochs = 1000
print_interval = 100

# 创建模型实例
model = LinearRegression()

# 记录训练过程
loss_history = []
w_history = []
b_history = []

print("开始训练线性回归模型（使用反向传播）...")
print(f"初始参数: w={model.w[0][0]:.4f}, b={model.b[0][0]:.4f}")

# 训练循环[1](@ref)
for epoch in range(epochs):
    # 前向传播
    y_pred = model.forward(x_data)

    # 计算损失
    current_loss = model.compute_loss(y_pred, y_data)

    # 反向传播并更新参数
    dw, db = model.backward(x_data, y_data, y_pred, learning_rate)

    # 记录历史
    loss_history.append(current_loss)
    w_history.append(model.w[0][0])
    b_history.append(model.b[0][0])

    # 打印训练进度
    if epoch % print_interval == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss = {current_loss:.6f}, w = {model.w[0][0]:.4f}, b = {model.b[0][0]:.4f}")

print("训练完成!")
print(f"最终参数: w={model.w[0][0]:.4f}, b={model.b[0][0]:.4f}")
print(f"最终损失: {loss_history[-1]:.6f}")

# 可视化训练过程
plt.figure(figsize=(15, 5))

# 1. 损失函数下降曲线
plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# 2. 参数w的变化
plt.subplot(1, 3, 2)
plt.plot(w_history)
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('Weight Evolution')
plt.grid(True)

# 3. 参数b的变化
plt.subplot(1, 3, 3)
plt.plot(b_history)
plt.xlabel('Epoch')
plt.ylabel('Bias (b)')
plt.title('Bias Evolution')
plt.grid(True)

plt.tight_layout()
plt.show()

# 显示最终拟合效果
plt.figure(figsize=(10, 6))
plt.scatter(x_data.flatten(), y_data.flatten(), color='blue', label='True data', alpha=0.7)

# 生成预测线
x_test = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
y_test = model.forward(x_test)

plt.plot(x_test.flatten(), y_test.flatten(), color='red', linewidth=2, label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Backpropagation')
plt.legend()
plt.grid(True)
plt.show()


# 预测新数据
def predict(x_new, model):
    """使用训练好的模型预测新数据"""
    x_new = np.array([[x_new]])  # 转换为二维数组
    return model.forward(x_new)[0][0]


# 测试预测
test_x = 6.0
predicted_y = predict(test_x, model)
print(f"预测结果: x={test_x}, y_pred={predicted_y:.4f}")

# 对比原始方法和反向传播方法的效果
print("\n=== 方法对比 ===")
print("原始方法: 遍历所有可能的w值，找到使MSE最小的w")
print("反向传播方法: 通过梯度下降自动优化参数w和b")
print(f"反向传播找到的参数: w={model.w[0][0]:.4f}, b={model.b[0][0]:.4f}")