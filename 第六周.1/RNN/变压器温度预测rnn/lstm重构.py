import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

data = pd.read_csv('./datasets/ETTm2.csv')

start_time = time.time()
# 将数据以列的形式提取出来 处理也许更简单
f1 = data['HUFL'].values
f2 = data['HULL'].values
f3 = data['MUFL'].values
f4 = data['MULL'].values
OT = data['OT'].values
print(f1.shape)
history_size = 10
future_size = 1
# 采样为2 降低一下样本数量
step_size = 2
features = []
labels = []
for i in range(history_size, len(f1) - 2, step_size):
    ''' 从10 开始采样 一直到len-2
    
    '''
    history_f1 = f1[i - history_size:i]
    history_f2 = f2[i - history_size:i]
    history_f3 = f3[i - history_size:i]
    history_f4 = f4[i - history_size:i]
    # shape: (10, )
    feature = np.stack((history_f1, history_f2, history_f3, history_f4), axis=1)
    features.append(feature)
    label = OT[i:i + future_size]
    labels.append(label)

features = np.array(features)
print(features.shape)
labels = np.array(labels)

# 归一化
scaler_x = StandardScaler()
scaler_y = StandardScaler()

features_nomalized = scaler_x.fit_transform(features.reshape(-1, 4))  # 转换成二维数组进行归一化（归一化只能对二维数组进行）
labels_nomalized = scaler_y.fit_transform(labels.reshape(-1, 1))  # 这里的1表示只有一列标签

features = features_nomalized.reshape(-1, history_size, 4)
labels = labels_nomalized.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

print("训练集特征形状:", x_train[1])  # (31350, 10, 4) 90%
print("测试集特征形状:", x_test.shape)  # (3484, 10, 4)   10%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动检测GPU

x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


# 创建LSTM模型（原RNN模型改为LSTM）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # input_size: 输入特征维度 (例如4，对应HUFL/HULL/MUFL/MULL四个特征)
        # hidden_size: 隐藏层神经元数量 (例如64)
        # num_layers: LSTM堆叠层数 (例如2)
        # batch_first: 输入/输出张量形状为 (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 数据变化：定义了一个可处理 (batch, seq_len, 4) -> (batch, seq_len, 64) 的LSTM层

        # 初始化全连接层
        # hidden_size: 输入维度 (LSTM输出的隐藏状态维度)
        # output_size: 输出维度 (例如1，预测OT值)
        self.fc = nn.Linear(hidden_size, output_size)
        # 数据变化：定义了一个可处理 (batch, 64) -> (batch, 1) 的线性层

    def forward(self, x):
        # x 输入形状: (batch_size, seq_len=10, input_size=4)
        # 例如: (32, 10, 4) 表示32个样本，每个样本10个时间步，每个时间步4个特征

        # 初始化隐藏状态和细胞状态
        # self.lstm.num_layers: LSTM层数 (例如2)
        # x.size(0): 当前batch大小 (例如32)
        # self.lstm.hidden_size: 隐藏层维度 (例如64)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # h0和c0形状: (num_layers=2, batch_size=32, hidden_size=64)
        # 数据变化：创建全零初始隐藏状态和细胞状态

        # LSTM前向传播（相比RNN多了细胞状态c）
        # x: (32, 10, 4)
        # h0: (2, 32, 64)
        # c0: (2, 32, 64)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out形状: (32, 10, 64)  (每个时间步的输出)
        # hn形状: (2, 32, 64)    (最终隐藏状态)
        # cn形状: (2, 32, 64)    (最终细胞状态)
        # 数据变化：LSTM处理时序数据，输出每个时间步的隐藏状态
        # 取最后一个时间步的输出并通过全连接层
        # out[:, -1, :] 选取所有batch的最后一个时间步输出
        # 形状: (32, 64) -> 经过fc层 -> (32, 1)
        return self.fc(out[:, -1, :])
        # 数据变化：最终输出形状 (batch_size=32, output_size=1)
# 实例化模型
input_size = 4  # 使用特征的列数作为输入特征数
hidden_size = 64  # 隐藏层大小
num_layers = 2  # LSTM层数
output_size = 1  # 输出特征数
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
#  设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):

    model.train()  # 将模型调至训练模式
    # 前向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#模型预测
model.eval()
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor).cpu()

y_pred = scaler_y.inverse_transform(y_pred_tensor.numpy())  # 将预测值逆归一化
y_test = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())  # 逆归一化真实值

# 评估指标
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mae_score = mae(y_test, y_pred)
print(f"R^2 score: {r2:.4f}")
print(f"MAE: {mae_score:.4f}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码运行时间: {elapsed_time:.6f} 秒")
# cpu运算 682.566581 秒
# Gpu运算
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))  # 创建一个新的 Matplotlib 图形，并设置图形的宽高尺寸
sample_indices = np.arange(len(y_test))  # 用于创建一个[0, len(y_test)]的等差数列的数组
plt.plot(sample_indices, y_test, color='blue', alpha=0.5, label='真实值')  # 绘制折线图，sample_indices为x轴数据，y_test为y轴数据
plt.plot(sample_indices, y_pred, color='red', alpha=0.5, label='预测值')
plt.xlabel('样本索引')
plt.ylabel('OT值')
plt.title('OT的实际值与预测值对比图(LSTM)')
plt.legend(['实际数据', '预测数据'])  # 显示图例
plt.grid(True)  # 显示网格线
plt.show()

elapsed_time = end_time - start_time
print(f"代码运行时间: {elapsed_time:.6f} 秒")