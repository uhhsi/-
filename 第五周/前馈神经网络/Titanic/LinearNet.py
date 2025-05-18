import torch
import torch.nn as nn
class LinearNet(nn.Module):
    def __init__(self, net_input, hidden0, hidden1, net_output):
        super(LinearNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(net_input, hidden0),
            nn.ReLU(),
            nn.Linear(hidden0, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, net_output),
        )

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
