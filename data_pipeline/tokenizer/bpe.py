import regex as re
from typing import List, Dict, Tuple
from collections import defaultdict
from itertools import pairwise
from tqdm import tqdm

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    计算相邻元素对的频率（bigram统计）
    """
    counts = defaultdict(int)
    for pair in pairwise(ids):
        counts[pair] += 1
    return dict(counts)

def merge(ids: List[int], pair: Tuple[int, int], idx) -> List[int]:
    """
    用idx替换ids中的pair
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class SimpleTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        pbar = tqdm(range(num_merges), disable=not verbose, desc="Simple Tokenizer Training")
        for i in pbar:
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> List[int]:
        """
        将字符串编码为 int 序列
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            if not stats:
                break

            # 找到优先级最高的pair 即 先加入merges的pair
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break

            tokens = merge(tokens, pair, self.merges[pair])

        return tokens

    def decode(self, ids: List[int]) -> str:
        """解码"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")


