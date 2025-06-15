import torch
from torch import nn
from d2l import torch as d2l
batch_size = 64
# imdb 评论数据集
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
class BiRNN(nn.Module):
    def __init__(self,vocab_size, embed_size, num_hiddens,num_layers):
        super(BiRNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)

        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,bidirectional=True)
        #拼接前向RNN的初始状态（编码序列起始特征）和反向RNN的最终状态（编码序列末尾特征）
        #前向RNN的最终状态（编码序列末尾特征）和反向RNN的初始状态（编码序列起始特征）
        # 共4 个隐藏状态
        self.decoder = nn.Linear(4*num_hiddens,2)

    def forward(self,inputs):
        # 转置 修改矩阵形状
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

#参数初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
# print(embeds.shape)

net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,devices)

# 预测
def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
print(predict_sentiment(net, vocab, 'this movie is so great'))
print(predict_sentiment(net, vocab, 'this movie is so bad'))
print(predict_sentiment(net, vocab, 'fucking great'))



