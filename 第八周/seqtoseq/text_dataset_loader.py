import torch
from torch.utils.data import Dataset, DataLoader

from seqtoseq.文本预处理 import get_vocab_corpus_from_timemachine, Vocabulary


class TextDataset(Dataset):
    def __init__(self, corpus, seq_length: int):
        self.corpus = corpus
        self.seq_length = seq_length

    def __len__(self):
        return len(self.corpus) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.corpus[idx: idx + self.seq_length]),
                torch.tensor(self.corpus[idx + 1: idx + self.seq_length + 1]))


def timemachine_data_loader(
        batch_size: int, seq_length: int, shuffle=False, max_token_num=10_000
) :
    vocab, corpus = get_vocab_corpus_from_timemachine(token_type='char', max_token_num=max_token_num)
    vocab: Vocabulary
    corpus: 'list'

    data_iter = DataLoader(TextDataset(corpus, seq_length), batch_size, shuffle, drop_last=True)
    return data_iter, vocab