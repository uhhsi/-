import os
import numpy as np
import pandas
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import LinearNet as LN
import torch.utils.data
# 设置训练设备 cuda好像出问题了只能在cpu 上训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读取数据
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# 输出训练集基本信息
train.head()
train.info()
# 合并训练和测试数据
all_data = pd.concat([train.drop("Survived", axis=1), test], axis=0)
# 删除无用列
all_data.drop(["PassengerId", "Cabin"], axis=1, inplace=True)
# 填补缺失值
all_data["Embarked"] = all_data["Embarked"].fillna("Not find")
# 归一化
num_feature = all_data.select_dtypes(include=[np.number]).columns
all_data[num_feature] = all_data[num_feature].apply(lambda x: (x - x.mean()) / (x.std()))
# 0填补
all_data[num_feature] = all_data[num_feature].fillna(0)
# 把数据集中的object转化为数值,使用独热编码  列数会剧增
all_data = pd.get_dummies(all_data, dummy_na=True)
# 转换 bool 类型为 int
bool_cols = all_data.select_dtypes(include=['bool']).columns
all_data[bool_cols] = all_data[bool_cols].astype(int)
# 获取列标签
cols = all_data.columns
print("cols:", cols)
# 把处理好的数据重新分为训练数据集和测试数据集,并创建对应的张量
train_feature = torch.tensor(all_data[:train.shape[0]].values, dtype=torch.float32, device=device)
test_feature = torch.tensor(all_data[train.shape[0]:].values, dtype=torch.float32, device=device)
# 提取标签并创建张量
train_label = torch.tensor(train["Survived"].values, dtype=torch.float32, device=device)
train_label = train_label.view(-1, 1)

# 定义模型的超参数
net_input = all_data.shape[1]
hidden0 = 32
hidden1 = 16
net_output = 1
model = LN.LinearNet(net_input, hidden0, hidden1, net_output).to(device)
lr = 0.01
batch_size = 32
# 损失函数
loss_fn = nn.MSELoss()
# 优化器
optimizer = optim.Adam(params=model.parameters(), lr=lr)
# 将训练特征和其标签合并，构成新的数据集
dataset = torch.utils.data.TensorDataset(train_feature, train_label)
# 拆分成小批次
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(1000):
    loss_epoch = 0.0
    for train_x, label_y in dataloader:
        train_x = train_x.to(device)
        label_y = label_y.to(device)
        # 确保数据没有nan
        assert not torch.isnan(train_x).any()
        assert not torch.isnan(label_y).any()

        # 前向传播 计算损失
        output = model(train_x)
        loss = loss_fn(output, label_y)

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 用优化器优化模型参数
        optimizer.step()
        # 如果损失为 NaN，终止
        if torch.isnan(loss):
            print("Training stopped due to NaN loss.")
            break
        loss_epoch += loss.item()
    # 输出损失值，
    avg_loss = loss_epoch / len(dataloader)
    print(f"Epoch: {epoch+1}, Loss: {avg_loss:.6f}")
test_feature = test_feature.to(device)
# 神经网络预测结果
test_result = model(test_feature).detach().to('cpu').numpy()
# 将结果压缩为1维张量
test_result = test_result.reshape(-1)
# 将结果输出二分类
test_result = np.where(test_result > 0.5, 1, 0)
test_result_series = pd.Series(test_result, name='Survived')
result = pandas.concat((test['PassengerId'], test_result_series), axis=1)
result.to_csv('submission.csv', index=False)
