"""
Simple word-level tokenizer for Surya inference engine.

Provides vocabulary management, encode/decode, and special token handling
suitable for simulated inference on embedded devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpecialTokens:
    """Canonical special token strings and their reserved ids.

    Attributes:
        BOS: Beginning-of-sequence token.
        EOS: End-of-sequence token.
        PAD: Padding token.
        UNK: Unknown token.
    """
    BOS: str = "<bos>"
    EOS: str = "<eos>"
    PAD: str = "<pad>"
    UNK: str = "<unk>"

    def as_dict(self) -> Dict[str, str]:
        """Return a mapping of role to token string."""
        return {
            "BOS": self.BOS,
            "EOS": self.EOS,
            "PAD": self.PAD,
            "UNK": self.UNK,
        }

    def all_tokens(self) -> List[str]:
        """Return all special token strings in a stable order."""
        return [self.PAD, self.UNK, self.BOS, self.EOS]


class Vocabulary:
    """Manages a bidirectional token-to-id mapping.

    Special tokens are always assigned the lowest ids.

    Attributes:
        special_tokens: The SpecialTokens instance.
    """

    def __init__(self, special_tokens: Optional[SpecialTokens] = None) -> None:
        self.special_tokens = special_tokens or SpecialTokens()
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        # Reserve ids for special tokens.
        for tok in self.special_tokens.all_tokens():
            self._add(tok)

    def _add(self, token: str) -> int:
        """Add a token and return its id. Idempotent."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        tid = len(self._token_to_id)
        self._token_to_id[token] = tid
        self._id_to_token[tid] = token
        return tid

    @property
    def size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._token_to_id)

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.special_tokens.PAD]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.special_tokens.UNK]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.special_tokens.BOS]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.special_tokens.EOS]

    def add_tokens(self, tokens: List[str]) -> List[int]:
        """Bulk-add tokens and return their ids."""
        return [self._add(t) for t in tokens]

    def token_to_id(self, token: str) -> int:
        """Look up the id for a token, returning unk_id if missing."""
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        """Look up the token for an id, returning UNK string if missing."""
        return self._id_to_token.get(token_id, self.special_tokens.UNK)

    def contains(self, token: str) -> bool:
        """Check whether a token is in the vocabulary."""
        return token in self._token_to_id


class SimpleTokenizer:
    """Word-level tokenizer with special token handling.

    Splits text on whitespace, maps words to ids via a Vocabulary,
    and supports optional BOS/EOS wrapping.

    Attributes:
        vocab: The underlying Vocabulary.
        add_bos: Whether to prepend BOS on encode.
        add_eos: Whether to append EOS on encode.
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.vocab = vocab or Vocabulary()
        self.add_bos = add_bos
        self.add_eos = add_eos

    def fit(self, texts: List[str]) -> None:
        """Build the vocabulary from a list of text strings.

        Args:
            texts: Training corpus; each string is whitespace-split into
                   tokens.
        """
        for text in texts:
            tokens = text.lower().split()
            self.vocab.add_tokens(tokens)

    def encode(self, text: str) -> List[int]:
        """Encode a text string into a list of token ids.

        Args:
            text: Input text to tokenize.

        Returns:
            List of integer token ids.
        """
        ids: List[int] = []
        if self.add_bos:
            ids.append(self.vocab.bos_id)

        for word in text.lower().split():
            ids.append(self.vocab.token_to_id(word))

        if self.add_eos:
            ids.append(self.vocab.eos_id)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token ids back into a text string.

        Args:
            ids: Token ids to decode.
            skip_special: If True, omit special tokens from the output.

        Returns:
            Decoded text string.
        """
        special = set(self.vocab.special_tokens.all_tokens())
        words: List[str] = []
        for tid in ids:
            token = self.vocab.id_to_token(tid)
            if skip_special and token in special:
                continue
            words.append(token)
        return " ".join(words)

    def pad_sequence(
        self, ids: List[int], target_length: int
    ) -> List[int]:
        """Pad (or truncate) a sequence to a target length.

        Args:
            ids: Token id sequence.
            target_length: Desired length.

        Returns:
            Padded or truncated sequence.
        """
        if len(ids) >= target_length:
            return ids[:target_length]
        return ids + [self.vocab.pad_id] * (target_length - len(ids))
