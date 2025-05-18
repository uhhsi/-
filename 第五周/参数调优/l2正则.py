import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import font
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

root = tk.Tk()
default_font = font.nametofont("TkDefaultFont")
default_font.configure(family="Microsoft YaHei")
root.destroy()  # 立即关闭Tkinter窗口，避免干扰

# 生成非线性数据
np.random.seed(42)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.3 * np.random.randn(100, 1)

# 标准化
X = (X - X.mean()) / X.std()

# 神经网络参数
input_size = 1
hidden_size = 50
output_size = 1
learning_rate = 0.01
epochs = 5000
lambda_ = 0.1  # L2正则化系数
# 初始化权重
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))
def relu(x):
    return np.maximum(0, x)
# 训练网络
losses = []
for epoch in range(epochs):
    # 前向传播
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2

    # 计算损失 (MSE + L2正则化)
    mse_loss = np.mean((z2 - y) ** 2)
    l2_penalty = 0.5 * lambda_ * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    total_loss = mse_loss + l2_penalty
    losses.append(total_loss)

    # 反向传播
    dz2 = (z2 - y) / len(X)
    dW2 = np.dot(a1.T, dz2) + lambda_ * W2  # 添加L2正则化项
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (z1 > 0)  # ReLU导数
    dW1 = np.dot(X.T, dz1) + lambda_ * W1  # 添加L2正则化项
    db1 = np.sum(dz1, axis=0, keepdims=True)
    # 更新权重
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
# 预测
z1 = np.dot(X, W1) + b1
a1 = relu(z1)
predictions = np.dot(a1, W2) + b2

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label="真实数据")
plt.plot(X, predictions, 'r', label="预测结果")
plt.title("神经网络拟合效果")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("训练损失曲线")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.show()