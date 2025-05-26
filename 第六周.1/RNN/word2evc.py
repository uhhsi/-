import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "I enjoy playing football with my friends",
    "natural language processing is fascinating"
]
# 构建词汇表
def build_vocab(corpus, min_freq=1):
    words = []
    for sentence in corpus:
        words.extend(sentence.split())
    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items())}
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word
vocab, idx2word = build_vocab(corpus)
vocab_size = len(vocab)
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, target, context):
        # 输入层到隐藏层
        in_emb = self.in_embed(target)  # (batch_size, embed_dim
        # 隐藏层到输出层
        out_emb = self.out_embed(context)  # (batch_size, embed_dim)
        # 计算分数
        scores = torch.matmul(in_emb, out_emb.t())  # (batch_size, batch_size)
        return scores
# 训练数据生成
def get_training_data(corpus, vocab, window_size=2):
    data = []
    for sentence in corpus:
        words = sentence.split()
        indices = [vocab[word] for word in words]

        for center_pos in range(len(indices)):
            for offset in range(-window_size, window_size + 1):
                context_pos = center_pos + offset

                if (context_pos < 0) or (context_pos >= len(indices)) or (center_pos == context_pos):
                    continue

                data.append((indices[center_pos], indices[context_pos]))
    return data


train_data = get_training_data(corpus, vocab)

embed_dim = 50
model = SkipGram(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 训练循环
def train(model, data, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for target, context in data:
            # 准备数据
            target_tensor = torch.LongTensor([target])
            context_tensor = torch.LongTensor([context])
            scores = model(target_tensor, context_tensor)
            loss = criterion(scores, torch.LongTensor([0]))  # 负采样简化处理

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data):.4f}')
train(model, train_data, epochs=50)
# 获取词向量
def get_word_vectors(model, vocab):
    word_vectors = {}
    for word, idx in vocab.items():
        word_vectors[word] = model.in_embed(torch.LongTensor([idx])).detach().numpy()[0]
    return word_vectors
word_vectors = get_word_vectors(model, vocab)

print("'fox'的向量表示:", word_vectors['fox'][:5])  # 打印前5维
print("'dog'的向量表示:", word_vectors['dog'])