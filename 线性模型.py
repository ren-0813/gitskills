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
    x_data, y_data = load_data('D:\\360MoveData\\Users\\hm943\\Desktop\\train.csv')
    print(f"成功加载 {len(x_data)} 个数据点")
except Exception as e:
    print(f"数据加载错误: {e}")
    # 使用示例数据作为备用
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_data = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    print("使用示例数据")


# 模型定义（含偏置项b）
def forward(x, w, b):
    # 确保x是数值类型，而不是序列
    if isinstance(x, (list, np.ndarray)):
        # 如果x是数组，确保使用元素级乘法
        return w * np.array(x, dtype=np.float64) + b
    else:
        # 如果x是单个数值
        return w * x + b


# 损失函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


# 参数范围设置
w_range = np.arange(0.0, 4.1, 0.1)  # w范围
mse_list = []

# 计算不同w值的MSE（固定b=0简化模型）
for w in w_range:
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        # 确保x_val是单个数值而不是序列
        if isinstance(x_val, (list, np.ndarray)) and len(x_val) > 0:
            # 如果x_val是序列，取第一个元素
            x_val = x_val[0] if hasattr(x_val, '__getitem__') else float(x_val)

        # 将x_val转换为浮点数以确保类型匹配
        x_val_float = float(x_val)
        loss_val = loss(x_val_float, y_val, w, 0)  # 简化模型，设b=0
        l_sum += loss_val

    mse = l_sum / len(x_data)
    mse_list.append(mse)
    print(f"w={w:.1f}, MSE={mse:.4f}")

# 找到使MSE最小的w值
min_mse = min(mse_list)
min_w_index = mse_list.index(min_mse)
min_w = w_range[min_w_index]

print(f"\n最佳参数: w={min_w:.2f}")
print(f"最小MSE: {min_mse:.4f}")

# 绘制MSE随w变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(w_range, mse_list, 'b-', label='MSE')
plt.scatter(min_w, min_mse, color='red', s=100, label=f'Min MSE (w={min_w:.2f})')
plt.xlabel('Weight (w)')
plt.ylabel('MSE Loss')
plt.title('MSE vs Weight Parameter')
plt.legend()
plt.grid(True)
plt.show()
