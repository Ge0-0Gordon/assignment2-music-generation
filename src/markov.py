"""Tiny n-gram baseline for symbolic token smoke tests."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import exp, log
from typing import Iterable


class NGramModel:
    """A small backoff n-gram next-token model."""

    def __init__(self, order: int = 3, seed: int = 0):
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.rng = random.Random(seed)
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram: Counter[int] = Counter()

    def fit(self, sequences: Iterable[list[int]]) -> None:
        for sequence in sequences:
            for token in sequence:
                self.unigram[token] += 1
            for idx, token in enumerate(sequence):
                for context_len in range(1, self.order):
                    if idx - context_len < 0:
                        continue
                    context = tuple(sequence[idx - context_len : idx])
                    self.counts[context][token] += 1

    def _distribution(self, context: list[int]) -> Counter[int]:
        max_len = min(self.order - 1, len(context))
        for context_len in range(max_len, 0, -1):
            key = tuple(context[-context_len:])
            if key in self.counts:
                return self.counts[key]
        return self.unigram

    def sample_next(self, context: list[int]) -> int:
        distribution = self._distribution(context)
        if not distribution:
            raise ValueError("model has no token counts")
        tokens = list(distribution.keys())
        weights = list(distribution.values())
        return self.rng.choices(tokens, weights=weights, k=1)[0]

    def sample(self, length: int, prefix: list[int] | None = None) -> list[int]:
        output = list(prefix or [])
        while len(output) < length:
            output.append(self.sample_next(output))
        return output

    def perplexity(self, sequences: Iterable[list[int]]) -> float:
        total_log_prob = 0.0
        total_tokens = 0
        vocab_size = max(len(self.unigram), 1)
        for sequence in sequences:
            history: list[int] = []
            for token in sequence:
                distribution = self._distribution(history)
                total = sum(distribution.values())
                count = distribution[token]
                probability = (count + 1) / (total + vocab_size)
                total_log_prob += log(probability)
                total_tokens += 1
                history.append(token)
        if total_tokens == 0:
            return float("inf")
        return exp(-total_log_prob / total_tokens)
