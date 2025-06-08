from collections import Counter
from enum import Enum
from typing import Literal, Iterable, Callable, Optional


class ST(str, Enum):
    """特殊词元 (Special Token)"""

    UNK = '<UNK>'
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    SEP = '<SEP>'
    MASK = '<MASK>'

    @property
    def description(self):
        describe = {
            ST.UNK: '未被纳入词表的未知词元 (unknown token)',
            ST.PAD: '用于填充序列长度的标记 (padding token)',
            ST.SOS: '序列的开始标记 (start of sequence)',
            ST.EOS: '序列的结束标记 (end of sequence)',
            ST.SEP: '段落的分隔标记 (separator token)',
            ST.MASK: '掩盖以用于预测的标记 (mask token)',
        }
        return describe[self.value]


class Vocabulary:
    def __init__(self, flat_tokens , special_tokens: Iterable[ST], min_freq: int = 1):
        """
        :param flat_tokens: 被展平后的词元列表
        :param min_freq: 纳入词表的最小词频
        :param special_tokens: 由特殊词元组成的可迭代对象
        """
        self.__unk_token = '<UNK>'
        tokens  = list(dict.fromkeys([self.__unk_token] + [i for i in special_tokens]))
        self.__valid_token_freqs = {token: freq for token, freq in Counter(flat_tokens).items() if freq >= min_freq}
        tokens.extend([
            token for token, index in
            sorted(self.__valid_token_freqs.items(), key=lambda pair: pair[1], reverse=True)
        ])

        self.__token_to_index = {token: index for index, token in enumerate(tokens)}
        self.__index_to_token = {index: token for token, index in self.__token_to_index.items()}

    def __len__(self):
        """词表大小"""
        return len(self.__token_to_index)

    def __repr__(self) -> str:
        return f'Vocabulary(len={len(self)})'

    def get_index(self, token: str) -> int:
        """根据词元获取索引值"""
        return self.__token_to_index.get(
            token,  # 当词元纳入了词汇表，返回索引值
            self.__token_to_index[self.__unk_token]  # 若词元未纳入词汇表，则返回 `<UNK>` 的索引值
        )

    def get_token(self, idx: int) -> str:
        """根据索引值获取词元"""
        return self.__index_to_token.get(
            idx,  # 当词汇表包含了该索引值，返回对应词元
            self.__unk_token  # 若词汇表不包含该索引值，则返回词元 `<UNK>`
        )

    def encode(self, tokens ) :
        """将词元列表转换为索引值列表"""
        return [self.get_index(token) for token in tokens]

    def decode(self, indices)  :
        """将索引值列表转换为词元列表"""
        return [self.get_token(index) for index in indices]

    @property
    def vocabulary(self):
        return self.__token_to_index

    @property
    def valid_token_freqs(self) :
        """获取词元的频率字典（保持语料文本输入时的原始顺序）"""
        return self.__valid_token_freqs


def remove_non_alpha_and_lower(string: str) -> str:
    """去除字符串中的非字母字符、两端的空格，并转换为小写"""
    import re
    return re.sub(r'[^a-zA-Z]+', ' ', string).strip().lower()


def get_timemachine_lines(line_processor: Callable[[str], str] = remove_non_alpha_and_lower)  :
    import os
    url = r'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    file_path = 'timemachine.txt'

    if not os.path.exists(file_path):
        import requests
        print(f'下载：{url!r}……')
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f'下载失败，错误代码：{response.status_code}')

    with open(file_path, 'r') as f:
        return [line_processor(line) for line in f]


def tokenize(string: str, token_type: Literal['word', 'char'] = 'word')  :
        if token_type== 'word':
            return string.split()
        else:
            return list(string)


def get_vocab_corpus_from_timemachine(
        token_type: Literal['word', 'char'],
        special_tokens: Iterable[ST] = (ST.UNK,),
        max_token_num: Optional[int] = None):
    tokenized_lines = [tokenize(line, token_type) for line in get_timemachine_lines()]
    flat_tokens = [token for lines in tokenized_lines for token in lines]
    vocab_instance = Vocabulary(flat_tokens, special_tokens)
    list_token_indies = [vocab_instance.get_index(token) for token in flat_tokens]

    return vocab_instance, list_token_indies[:max_token_num] if max_token_num else list_token_indies


def generate_bigram_frequencies(words ) :
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    bigram_freq = Counter(bigrams)
    return bigram_freq.most_common()


def generate_trigram_frequencies(words ):
    trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]
    trigram_freq = Counter(trigrams)
    return trigram_freq.most_common()