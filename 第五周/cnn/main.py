import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
import CNN

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
# 输出训练集基本信息
train.head()
train.info()

# 合并训练数据和测试数据
all_data = pd.concat([train.drop("label", axis=1), test], axis=0, ignore_index=True)
# 对像素数据归一化
all_data = all_data / 255.0  # 像素值的范围是0~255
# 个数自动填充 主要是规定张量大小
all_data = all_data.values.reshape(-1, 1, 28, 28)
# 将数据集划分为训练集和测试集
train_data = all_data[:train.shape[0]]
test_data = all_data[train.shape[0]:]
# 为训练数据和测试数据创建张量
train_image = torch.tensor(train_data, dtype=torch.float32)
test_image = torch.tensor(test_data, dtype=torch.float32)
# 为训练标签创建张量
train_label = torch.tensor(train["label"].values, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(train_image, train_label)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# 定义模型
model = CNN.CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 训练模型
for epoch in range(200):
    loss_epoch = 0.0
    for image_x, label_y in train_loader:
        # 将数据移动到设备上
        image_x, label_y = image_x.to(device), label_y.to(device)
        # 前向传播
        output = model(image_x)
        # 计算损失函数
        loss = criterion(output, label_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {loss_epoch / len(train_loader)}")
test_image = test_image.to(device)
output = model(test_image)
test_output = pd.Series(output.argmax(dim=1).cpu().numpy())
test_result = pd.concat([pd.Series(range(1, 28001), name="ImageId"), test_output.rename("label")], axis=1)
# 将结果保存为csv文件
test_result.to_csv('submission.csv', index=False)
