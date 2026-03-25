"""Tests for surya.tokenizer — vocabulary, encoding, decoding."""

import pytest

from surya.tokenizer import SimpleTokenizer, SpecialTokens, Vocabulary


# ---- SpecialTokens tests ----

def test_special_tokens_defaults():
    st = SpecialTokens()
    assert st.BOS == "<bos>"
    assert st.EOS == "<eos>"
    assert st.PAD == "<pad>"
    assert st.UNK == "<unk>"


def test_special_tokens_all():
    st = SpecialTokens()
    tokens = st.all_tokens()
    assert len(tokens) == 4
    assert "<pad>" in tokens


# ---- Vocabulary tests ----

def test_vocab_special_ids():
    vocab = Vocabulary()
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.bos_id == 2
    assert vocab.eos_id == 3


def test_vocab_add_and_lookup():
    vocab = Vocabulary()
    ids = vocab.add_tokens(["hello", "world"])
    assert vocab.token_to_id("hello") == ids[0]
    assert vocab.id_to_token(ids[1]) == "world"


def test_vocab_unknown_token():
    vocab = Vocabulary()
    assert vocab.token_to_id("nonexistent") == vocab.unk_id


def test_vocab_contains():
    vocab = Vocabulary()
    vocab.add_tokens(["alpha"])
    assert vocab.contains("alpha")
    assert not vocab.contains("beta")


def test_vocab_size():
    vocab = Vocabulary()
    assert vocab.size == 4  # just specials
    vocab.add_tokens(["a", "b", "c"])
    assert vocab.size == 7


# ---- SimpleTokenizer tests ----

def test_tokenizer_encode_decode():
    tok = SimpleTokenizer()
    tok.fit(["hello world foo bar"])
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    assert text == "hello world"


def test_tokenizer_bos_eos():
    tok = SimpleTokenizer(add_bos=True, add_eos=True)
    tok.fit(["test"])
    ids = tok.encode("test")
    assert ids[0] == tok.vocab.bos_id
    assert ids[-1] == tok.vocab.eos_id


def test_tokenizer_no_bos_eos():
    tok = SimpleTokenizer(add_bos=False, add_eos=False)
    tok.fit(["cat"])
    ids = tok.encode("cat")
    assert ids[0] != tok.vocab.bos_id


def test_tokenizer_unknown_word():
    tok = SimpleTokenizer()
    tok.fit(["apple"])
    ids = tok.encode("banana")
    # banana not in vocab -> unk
    assert tok.vocab.unk_id in ids


def test_tokenizer_pad_sequence():
    tok = SimpleTokenizer()
    tok.fit(["a b c"])
    ids = tok.encode("a b")
    padded = tok.pad_sequence(ids, 10)
    assert len(padded) == 10
    assert padded[-1] == tok.vocab.pad_id


def test_tokenizer_pad_truncate():
    tok = SimpleTokenizer()
    tok.fit(["a b c d e"])
    ids = tok.encode("a b c d e")
    truncated = tok.pad_sequence(ids, 3)
    assert len(truncated) == 3


def test_tokenizer_decode_skip_special():
    tok = SimpleTokenizer()
    tok.fit(["hi there"])
    ids = tok.encode("hi there")
    with_special = tok.decode(ids, skip_special=False)
    without_special = tok.decode(ids, skip_special=True)
    assert "<bos>" in with_special
    assert "<bos>" not in without_special
