import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.datasets
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 将数据集处理封装为LightningDataModule
class PetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir='./PetImages', batch_size=16):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        # 数据集分隔
        self.dataset = torchvision.datasets.ImageFolder(
            root=self.root_dir,
            transform=self.transform
        )
        train_len = int(0.8 * len(self.dataset))
        test_len = len(self.dataset) - train_len
        self.train_dataset, self.test_dataset = random_split(
            self.dataset, [train_len, test_len]
        )

    #DataLoader参数
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )


# 将模型封装为LightningModule
class MyNet(pl.LightningModule):
    def __init__(self, learning_rate=0.1):
        super().__init__()
        self.save_hyperparameters()  # 保存超参数


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    #训练循环
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == targets).float().mean()

        # 自动记录日志
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    # 测试逻辑
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    # 配置优化器
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate
        )


#训练代码
if __name__ == '__main__':
    # 初始化数据模块
    dm = PetDataModule(batch_size=16)

    # 模型初始化
    model = MyNet(learning_rate=0.1)

    #创建Trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",  # 自动检测
        devices="auto",
        enable_progress_bar=True
    )

    # 开始训练
    trainer.fit(model, datamodule=dm)