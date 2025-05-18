import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def create_data(n=1000):
    X = torch.rand(n, 10) * 5  # 10维特征，范围[0,5)
    y = X.sum(dim=1, keepdim=True) + torch.randn(n, 1)  # 目标值 = 特征和 + 噪声
    return X, y
X, y = create_data()
X_train, y_train = X[:800], y[:800]  # 训练集
X_test, y_test = X[800:], y[800:]  # 测试集
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),  # 输入层→隐藏层
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout层
            nn.Linear(64, 1)  # 输出层（回归任务）
        )
    def forward(self, x):
        return self.net(x)

model = SimpleMLP()
print(model)
def train(model, X_train, y_train, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        model.train()  # 开启Dropout
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            model.eval()  # 关闭Dropout
            with torch.no_grad():
                test_loss = loss_fn(model(X_test), y_test)
            print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss with Dropout')
    plt.show()

train(model, X_train, y_train)
model.eval()
with torch.no_grad():
    sample = torch.rand(1, 10)  # 随机生成一个样本
    pred = model(sample)
    print(f"\nSample Input: {sample.squeeze()}")
    print(f"Predicted Output: {pred.item():.2f}, True Value: {sample.sum().item():.2f}")
