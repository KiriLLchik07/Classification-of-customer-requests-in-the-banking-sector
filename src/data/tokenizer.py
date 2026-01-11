from collections import Counter
import re

class TextTokenizer:
    def __init__(
        self,
        min_freq: int = 2,
        max_vocab_size: int = 30000,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.word2idx = {}
        self.idx2word = {}

    def fit(self, texts: list[str]):
        counter = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        vocab = [word for word, freq in counter.items() if freq >= self.min_freq]

        vocab = vocab[: self.max_vocab_size - 2]

        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1,
        }

        for idx, word in enumerate(vocab, start=2):
            self.word2idx[word] = idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        return [
            self.word2idx.get(token, self.word2idx[self.unk_token])
            for token in tokens
        ]

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
