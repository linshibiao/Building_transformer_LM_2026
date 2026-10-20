"""Byte-level BPE training and tokenizer (CS336 Assignment 1, 2026)."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Optional

import regex as re

# GPT-2 pre-tokenization pattern (needs the third-party `regex` module).
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _bytes_to_tuple(token: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in token)


def _count_pairs(
    word_freqs: dict[tuple[bytes, ...], int],
) -> Counter:
    pairs: Counter = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge_word(
    word: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    a, b = pair
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _split_on_specials(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens so merges cannot cross their boundaries.

    Specials themselves are discarded (they do not contribute to merge counts).
    """
    if not special_tokens:
        return [text] if text else []
    # Longer specials first so overlapping tokens match correctly under |.
    escaped = [re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)]
    pattern = "|".join(escaped)
    return [chunk for chunk in re.split(pattern, text) if chunk]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns (vocab, merges) where vocab maps id -> token bytes and merges is
    the ordered list of (token1, token2) merge pairs.
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size={vocab_size} too small for 256 bytes + "
            f"{len(special_tokens)} special tokens"
        )

    # Vocabulary: specials first (matches staff fixture style), then bytes,
    # then merges. Only set(keys)/set(values) are asserted by tests.
    vocab: dict[int, bytes] = {}
    next_id = 0
    for special in special_tokens:
        vocab[next_id] = special.encode("utf-8")
        next_id += 1
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Specials are hard segmentation boundaries and must not affect merge stats.
    word_freqs: dict[tuple[bytes, ...], int] = Counter()
    for chunk in _split_on_specials(text, special_tokens):
        for match in PRETOKEN_PATTERN.finditer(chunk):
            word_freqs[_bytes_to_tuple(match.group(0).encode("utf-8"))] += 1

    n_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []
    pair_counts = _count_pairs(word_freqs)

    for _ in range(n_merges):
        if not pair_counts:
            break
        # Highest count; ties → lexicographically greater pair.
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] < 1:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        # Rewrite words that contain the pair and refresh affected pair counts.
        affected: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freqs.items():
            if any(
                word[i] == best_pair[0] and word[i + 1] == best_pair[1]
                for i in range(len(word) - 1)
            ):
                affected[word] = freq

        for word, freq in affected.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            new_word = _merge_word(word, best_pair)
            del word_freqs[word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq
            for i in range(len(new_word) - 1):
                pair_counts[(new_word[i], new_word[i + 1])] += freq

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer with optional special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        # Append specials missing from the provided vocabulary.
        for special in self.special_tokens:
            encoded = special.encode("utf-8")
            if encoded not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = encoded

        self._bytes_to_id: dict[bytes, int] = {}
        for i, b in self.vocab.items():
            if b not in self._bytes_to_id or i < self._bytes_to_id[b]:
                self._bytes_to_id[b] = i

        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # Longest special first so overlapping specials match correctly.
        self._special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        if self._special_tokens_sorted:
            escaped = [re.escape(s) for s in self._special_tokens_sorted]
            self._special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pattern = None

        self._special_to_id: dict[str, int] = {}
        for s in self.special_tokens:
            encoded = s.encode("utf-8")
            if encoded in self._bytes_to_id:
                self._special_to_id[s] = self._bytes_to_id[encoded]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: Optional[list[str]] = None,
    ) -> Tokenizer:
        """Load a tokenizer from GPT-2-style vocab JSON and merges text files."""
        import json

        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        byte_encoder = dict(zip(bs, [chr(c) for c in cs]))
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab = {
            idx: bytes([byte_decoder[ch] for ch in token])
            for token, idx in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                t1, t2 = parts
                merges.append(
                    (
                        bytes([byte_decoder[c] for c in t1]),
                        bytes([byte_decoder[c] for c in t2]),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _merge_bytes(self, piece: bytes) -> list[bytes]:
        """Apply BPE merges to a single pre-token's UTF-8 bytes."""
        if not piece:
            return []
        symbols: list[bytes] = [bytes([b]) for b in piece]
        if len(symbols) == 1:
            return symbols

        while True:
            best = None  # (rank, index)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair)
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, i)
            if best is None:
                break
            i = best[1]
            symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
        return symbols

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in PRETOKEN_PATTERN.finditer(text):
            for token_bytes in self._merge_bytes(match.group(0).encode("utf-8")):
                ids.append(self._bytes_to_id[token_bytes])
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self._special_pattern is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        last = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > last:
                ids.extend(self._encode_ordinary(text[last : match.start()]))
            special = match.group(0)
            ids.append(self._special_to_id[special])
            last = match.end()
        if last < len(text):
            ids.extend(self._encode_ordinary(text[last:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yield token ids from a stream of text chunks without loading all text.

        Flushes on newline boundaries so GPT-2 pretokens (which do not cross
        non-whitespace line boundaries) stay intact when the iterable is a file.
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if "\n" in buffer:
                head, buffer = buffer.rsplit("\n", 1)
                yield from self.encode(head + "\n")
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
