import torch
import torch.nn as nn

# 初始化RNN层和输入数据
input_size = 4
hidden_size = 64
batch_size = 5
seq_len = 10
rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,batch_first=True)
# 生成随机输入和初始隐藏状态
x = torch.randn(batch_size, seq_len, input_size)
print('this is x',x)
h0 = torch.zeros(1, batch_size, hidden_size)  # (num_layers, batch, hidden)
#  PyTorch原生计算结果
output, hn = rnn(x, h0)
print("PyTorch输出形状:", output.shape)  # 应输出 (5,10,64)
# 手动验证第一个时间步 -----------------------------------
# 获取RNN的内部参数
W_ih = rnn.weight_ih_l0  # 输入到隐藏的权重 (64,4)
W_hh = rnn.weight_hh_l0  # 隐藏到隐藏的权重 (64,64)
b_ih = rnn.bias_ih_l0    # 输入偏置 (64,)
b_hh = rnn.bias_hh_l0    # 隐藏偏置 (64,)
# 提取第一个时间步的输入和初始隐藏状态
x_t = x[:, 0, :]
print('this is x_t.shape',x_t.shape)# 取每个样本的第一个时间步
h_prev = h0[0]
#手动计算
h_t = torch.tanh(
    torch.mm(x_t, W_ih.T) + b_ih +  # 第一次循环 第一个时间步于权重矩阵相乘
    torch.mm(h_prev, W_hh.T) + b_hh  # 隐藏状态转换 (32,64) @ (64,64) -> (32,64)
)
print('\n手动计算结果验证:')
print("手动计算h_t形状:", h_t.shape)  # 应输出 (5,64)
print("PyTorch输出第一个时间步的形状:", output[:,0,:].shape)  # (5,64)
