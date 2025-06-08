from time import time
import  torch
from typing import Tuple, Optional, Iterable
from typing import Optional
import torch

from torch import nn, Tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoder_decoder import AbstractEncoder, AbstractDecoder, EncoderDecoder

from seqtoseq.文本预处理 import ST

VOCAB_SIZE = 10
EMBED_DIM = 3
input_indices = torch.LongTensor([[1, 2, 4, 5],
                                  [4, 3, 2, 9]])
embedding_layer  = torch.nn.Embedding(VOCAB_SIZE,EMBED_DIM)
embedd_vectors = embedding_layer(input_indices)
print(embedd_vectors.shape)
class Seq2SeqEncoder(AbstractEncoder):
    def __init__(self,vocab_size,embed_dim,hidden_dim,num_layers,dropout):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size,embed_dim)
        self.rnn =nn.GRU(embed_dim,hidden_dim,num_layers,dropout=dropout)

    def forward(self,input_seq,valid_lengths,**kwargs):
        """

        :param input_seq: 输入数据(经过padd打包过后的)
        :param valid_lengths: 每个时间步 长度
        :param kwargs:
        :return:
        """
        if input_seq.dim() != 2: raise ValueError(f'input_seq 应为二维张量！')
        embedded= self.embedding_layer(input_seq).contiguous()
        if valid_lengths is None:
            output,state = self.rnn(embedded)
        else:
            packed = pack_padded_sequence(input=embedded,
                                          lengths=valid_lengths.cpu(),
                                          enforce_sorted=False,
                                          batch_first=True)

            output,state = self.rnn(packed)
            output,_ = pad_packed_sequence(output)
        return  output,state
class Seq2SeqDecoder(AbstractDecoder):
    def __init__(self,vocab_size,embed_dim,hidden_num,num_layers,drop_out):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size,embed_dim)
        self.rnn =nn.GRU(input_size=embed_dim+hidden_num,hidden_size=hidden_num,num_layers=num_layers,dropout=drop_out)
        self.output_layer = nn.Linear(hidden_num,vocab_size)
        self.hidden_num = hidden_num
    def init_state(self,enc_output,**kwargs):
        """
        获取编码器输入
        :param enc_output:  encode ouput 编码器输出
        :param kwargs: 解码器的初始隐状态，形状为：(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
        :return:
        """
        return enc_output[1]

    def forward(self,input_seq,state):
        """

        :param input_seq:  解码器输入形状为：输入第一个时间步(BATCH_SIZE, SEQ_LENGTH)
        :param state: 编码器最后一层赢状态(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
        :return:
        """
        # (BATCH_SIZE, SEQ_LENGTH) -> (BATCH_SIZE, SEQ_LENGTH, EMBED_DIM) -> (SEQ_LENGTH, BATCH_SIZE, EMBED_DIM)
        embedded = self.embedding_layer(input_seq).permute(1,0,2).contiguous()

        #取最后一行
        context  = state[-1:].expand(embedded.shape[0],-1,-1)
        # 拼接上下文 与隐藏状态
        rnn_input = torch.cat([embedded,context],dim=2)
        output,state = self.rnn(rnn_input,state)
        return (output,), state
class SequenceLengthCrossEntropyLoss(nn.Module):
    """基于有效长度的交叉熵损失函数"""

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, label_smoothing: float = 0.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 size_average=size_average,
                                                 ignore_index=-100,  # 使用 PyTorch 默认值
                                                 reduce=reduce,
                                                 reduction='none',  # 设置为 'none' 以便后续手动应用掩码
                                                 label_smoothing=label_smoothing)


    def forward(self,inputs,targets,valid_lengths):
        """"
        在三维状态下的 向量交叉熵计算
        :param inputs: 模型预测的输出，形状为：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        :param targets: 目标标签，形状为：(BATCH_SIZE, SEQ_LENGTH)
        :param valid_lengths: 各序列的有效长度，形状为：(BATCH_SIZE,)
        :return: 掩码后的损失平均值
        """
        # 首先对向量的维度进行变换
        inputs =inputs.permute(1,2,0) # (SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE) -> (BATCH_SIZE, VOCAB_SIZE, SEQ_LENGTH)

        seq_length = targets.shape[1]
        
        """
        通过广播机制实现mask矩阵 bool掩码的实现 以seq_length = 5 shape为（5，） valid_lengths=tensor(2,3,5)shape为(3,)举例
        step1 生成 tensor( 0,1,2,3,4,) 并在0维增加一维 shape变成(1，seq_length) [[0, 1, 2, 3, 4]]
    
        step2 valid_lengths 在一维增加一维 shape变成(3,1)
        step3 广播 通过广播机制，(1, seq_length) 的张量和 (batch_size, 1) 的张量比较，会得到形状为 (batch_size, seq_length) 的布尔张量
        """
        mask = torch.arange(seq_length,device=targets.device).unsqueeze(0)<valid_lengths.unsqueeze(1)
        loss = self.cross_entropy(inputs,targets)
        #生成一个 batch_size,seq_length 的矩阵
        masked_mean_losses = (loss*mask.float()).mean(dim=1)
        return masked_mean_losses


class MultiIgnoreIndicesCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self,
                 ignore_indices: Iterable,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super().__init__(
            weight=weight,
            ignore_index=-100,  # 使用 PyTorch 默认值
            size_average=size_average,
            reduce=reduce,
            reduction='none',  # 设置对多个样本的损失值聚合的方式为 'none'，以便应用掩码
            label_smoothing=label_smoothing
        )
        self.ignore_indices = set(ignore_indices)
        self.reduction = reduction

    def forward(self, inputs, targets):
        mask = torch.ones_like(targets, dtype=torch.bool)  # 初始化掩码张量（全为 True）
        for idx in self.ignore_indices:
            mask = mask & (targets != idx)

        losses = super().forward(inputs, targets)  # 首先计算每个位置的损失
        masked_losses = losses * mask.float()  # 掩码后的损失值

        if self.reduction == 'sum':
            return masked_losses.sum()
        elif self.reduction == 'mean':
            return masked_losses.sum() / mask.sum().float().clamp(min=1.0)  # 防止极端情况下的除零错误
        else:
            return masked_losses  # 'none'

def train_one_epoch(
        module: EncoderDecoder,
        data_iter,
        optimizer:optim.Optimizer,
        criterion:nn.Module,
        tgt_vocab,
        device
):
    """
      一个迭代周期内 Seq2Seq 模型的训练

      :param module: 序列到序列模型
      :param data_iter: 数据集加载器
      :param optimizer: 优化器
      :param criterion: 损失函数
      :param tgt_vocab: 目标语言词表
      :param device: 计算设备
      :return: 平均损失, 训练速度 (tokens/sec)
      """
#  获取sos位置 索引 sos_idx = 0
    sos_idx = tgt_vocab.get_index(ST.SOS)
    total_loss = 0
    total_tokens  = 0

    module.train()
    start_time =time()
    for src,src_valid_len,tgt,tgt_valid_len  in data_iter:
        """
     src: 英文句子（比如"hello world"）

    tgt: 正确的中文翻译（比如"你好 世界"）

    src_valid_len: 英文句子的真实长度（去掉填充符后）

    tgt_valid_len: 中文句子的真实长度"""
        optimizer.zero_grad()
        src = src.to(device)
        tgt = tgt.to(device)  # 形状为：(BATCH_SIZE, SEQ_LENGTH)
        tgt_valid_len = tgt_valid_len.to(device)

        #填充
        dec_input = torch.cat([torch.full((tgt.shape[0],1),sos_idx,device=device),tgt[:,:-1]
                               ],dim=1)

        tgt_pred = module(src, dec_input, valid_lengths=src_valid_len)
        loss = criterion(inputs=tgt_pred[0], targets=tgt, valid_lengths=tgt_valid_len)  # 计算损失

        loss.sum().backward()  # 反向传播
        clip_grad_norm_(module.parameters(), max_norm=0.5)  # 梯度裁剪
        optimizer.step()  # 更新参数

        num_tokens = tgt_valid_len.sum().item()
        total_loss += loss.sum().item()
        total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    tokens_per_sec = total_tokens / (time() - start_time)

    return avg_loss, tokens_per_sec



if __name__=='__main__':
    from translation_dataset_loader import nmt_eng_fra_dataloader
    BATCH_SIZE = 128
    SEQ_LENGTH = 20
    EMBED_DIM = 256
    HIDDEN_NUM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.0000005
    EPOCHS_NUM = 100

    data_iter, eng_vocab, fra_vocab = nmt_eng_fra_dataloader(BATCH_SIZE, SEQ_LENGTH, num_workers=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 编码器集合
    nmt_model = EncoderDecoder(encoder=Seq2SeqEncoder(len(eng_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT),
                               decoder=Seq2SeqDecoder(len(fra_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT),
                               device=device
                              )  # 使用默认的模型参数初始化方法，不手动初始化
    optimizer = optim.Adam(nmt_model.parameters(), lr=LEARNING_RATE)
    criterion = SequenceLengthCrossEntropyLoss()
    for epoch in range(EPOCHS_NUM):
        loss, speed = train_one_epoch(nmt_model, data_iter, optimizer, criterion, fra_vocab, device)
        ""
        print(f'第 {epoch + 1:03} 轮：损失为 {loss:.3f}，速度为 {speed:.1f} tokens/sec')