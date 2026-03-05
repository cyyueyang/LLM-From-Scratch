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

class RegexTokenizer:
    def __init__(self, pattern: str = None):
        self.pattern = re.compile(pattern if pattern else r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        if verbose:
            print(f"1/2 将文本转换成字节ing")

        ids = list(text.encode("utf-8"))

        if verbose:
            print(f"2/2 开始{num_merges}次合并，文本有{len(ids)}bytes")

        pbar = tqdm(range(num_merges), disable=not verbose, desc="Tokenizer Training")
        for i in pbar:
            stats = get_stats(ids)
            if not stats:
                if verbose:
                    print(f"在{i+1}次 提前合并")
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and i % 50 == 0:  # 每50次更新一次显示
                token_preview = self.vocab[new_id].decode('utf-8', errors='replace')[:20]
                pbar.set_postfix_str(f"'{token_preview}' ({stats[pair]}x)", refresh=False)

        if verbose: print("✅ 训练完成")
    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            if not stats:
                break
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text: str) -> List[int]:
        tokens = []
        for chunk in re.findall(self.pattern, text):
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            tokens.extend(chunk_ids)
        return tokens

    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_prefix: str):
        model_file = f"{file_prefix}.model"

        with open(model_file, "w", encoding="utf-8") as f:
            f.write("regex_bpe_v1\n")
            f.write(f"{self.pattern.pattern}\n")

            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1], reverse=False)
            for pair, new_ids in sorted_merges:
                f.write(f"{pair[0]} {pair[1]}\n")

            if self.special_tokens:
                special_tokens_file = f"{file_prefix}.json"
                import json
                with open(special_tokens_file, "w", encoding="utf-8") as f:
                    json.dump(self.special_tokens, f)
            print(f"合并规则保存到{model_file}")

    def load(self, model_file: str):
        assert model_file.endswith(".model")
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        idx = None
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "regex_bpe_v1"

            self.pattern = re.compile(f.readline().strip())

            for i, line in enumerate(f):
                p1, p2 = map(int, line.split())
                idx = 256 + i
                self.merges[(p1, p2)] = idx
                self.vocab[idx] = self.vocab[p1] + self.vocab[p2]

        print(f"从{model_file}加载了{len(self.merges)}规则")








