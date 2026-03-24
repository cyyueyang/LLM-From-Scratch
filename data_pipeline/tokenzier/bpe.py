import regex as re
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from itertools import pairwise

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}

    for pair in pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx) -> List[int]:
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
        self.vocab: Dict[int, bytes] = self._build_base_vocab()

    def _build_base_vocab(self):
        return {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        pbar = tqdm(range(num_merges), disable=not verbose, desc="Simple Tokenizer training")

        for i in pbar:
            stats = get_stats(ids)
            if stats is None:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)

            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = idx

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))

        while True:
            stats = get_stats(ids)
            if not stats:
                break

            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode("utf-8", errors="replace")


class RegexTokenizer:
    def __init__(self, pattern: Optional[str] = None):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab = self._build_base_vocab()
        self.pattern = re.compile(pattern if pattern else r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _build_base_vocab(self):
        return {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        num_merges = vocab_size - 256

        if verbose:
            print(f"(1/2) Pre process text to bytes")

        ids = list(text.encode("utf-8"))

        if verbose:
            print(f"(2/2) num_merges: {num_merges}, len_ids: {len(ids)}")

        pbar = tqdm(range(num_merges), disable=not verbose, desc="RegexTokenizer training")

        for i in pbar:
            stats = get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)

            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = idx

        if verbose:
            print(f"Training Finished")

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """
        对一个文本块进行编码
        """

        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            if stats is None:
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
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_prefix: str):
        model_file = f"{file_prefix}.model"

        with open(model_file, "w", encoding="utf-8") as f:
            f.write("regex_bpe_v1\n")
            f.write(f"{self.pattern.pattern}\n")

            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])

            for (p1, p2), idx in sorted_merges:
                f.write(f"{p1} {p2}\n")

        if self.special_tokens:
            special_tokens_file = f"{file_prefix}.json"

            import json
            with open(special_tokens_file, "w", encoding="utf-8") as f:
                json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

        print(f"merges is saved to {model_file}")


    def load(self, model_file: str):
        self.merges = {}
        self.vocab = self._build_base_vocab()
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()

            self.pattern = re.compile(f.readline().strip())

            for i, line in enumerate(f.readlines()):
                p1, p2 = map(int, line.strip().split())
                idx = i + 256
                self.merges[(p1, p2)] = idx
                self.vocab[idx] = self.vocab[p1] + self.vocab[p2]
        print(f"merges is loaded from {model_file}")






