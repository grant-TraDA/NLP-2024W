from typing import Iterable

from torch import nn
from transformers import AutoTokenizer


class WordEmbeddingExtractor:

    def __init__(
        self, tokenizer: AutoTokenizer, model: nn.Module, sequence_length: int
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.sequence_length = sequence_length

    def __call__(self, input: str | Iterable[str]):
        if isinstance(input, str):
            tokens = self.tokenizer(
                input,
                return_tensors="pt",
                padding="max_length",
                max_length=self.sequence_length,
            )["input_ids"]
            return self.model.embeddings(tokens)
        assert isinstance(input, Iterable), "Improper input type"
        return [self(t) for t in input]
