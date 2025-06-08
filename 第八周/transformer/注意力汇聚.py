from torch.nn import functional as F
import torch
from torch import Tensor
import matplotlib.pyplot as plt

def plot_regression_comparison(x: Tensor, *, y_truth: Tensor, y_noisy: Tensor, y_pred: Tensor):
    """绘制回归对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_noisy, alpha=0.5, color='#FAC21E', edgecolor='none')
    plt.plot(x, y_truth, alpha=0.85, linewidth=3, color='#E43D30', label='Truth')
    plt.plot(x, y_pred, alpha=0.85, linewidth=3, color='#269745', label='Pred', linestyle='-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()

def nonlinear_dataset_generator(size: int, start: float = 0, end: float = 5):
    r"""
    基于非线性函数 y = 2 \sin(x) + x^{0.8} 生成数据集

    :param size: 样本数
    :param start: 特征的起始范围，在 [start, end) 区间内均匀采样
    :param end: 特征的结束范围，在 [start, end) 区间内均匀采样
    """
    features = torch.rand(size) * (end - start) + start
    features, _ = features.sort()  # 升序，以便注意力权重的可视化
    targets = 2 * torch.sin(features) + pow(features, 0.8)

    return features, targets


def gaussian_noise(size: int, mean: float = 0, std: float = 0.5) -> Tensor:
    r"""
    噪声 \epsilon \sim N(0, 0.5^2)

    :param size: 样本数
    :param mean: 噪声的均值
    :param std: 噪声的标准差
    """
    return torch.normal(mean=mean, std=std, size=(size,))

def nadaraya_watson_weights(queries: Tensor, keys: Tensor, h: float = 1.0) -> Tensor:
    print(queries)
    print(keys)
    r"""
    Nadaraya-Watson 核回归中的注意力权重 w_i(x, x_i) = Softmax(-\frac{(x - x_i)^2}{2h^2})

    :param queries: 查询 x
    :param keys: 键 x_i
    :param h: 带宽参数，控制注意力分布的聚焦程度
    :return: 注意力权重矩阵，形状为 (n_queries, n_keys)
    """
    # (n_queries * n_keys) -> (n_queries, n_keys)
    n= keys.shape[0]
    queries = queries.repeat(keys.shape[0]).reshape(-1,keys.shape[0])

    weights = F.softmax(-((queries - keys.reshape(n,-1)) ** 2) / (2 * h ** 2), dim=1)
    print(weights.shape)
    # weight.shape(50,50)
    return weights


if __name__ == '__main__':

    SAMPLE_NUM = 50
    x, truth = nonlinear_dataset_generator(size=SAMPLE_NUM)
    noisy = truth + gaussian_noise(size=SAMPLE_NUM)

    attention_weights = nadaraya_watson_weights(queries=x, keys=x)

    pred = attention_weights @ noisy

    plot_regression_comparison(x, y_truth=truth, y_noisy=noisy, y_pred=pred)
