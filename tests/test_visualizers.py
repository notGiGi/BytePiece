"""Tests for visualization helpers."""

from __future__ import annotations

from typing import List

from rich.console import Console

from bytepiece import train_bpe
from bytepiece.algorithms.bpe import BPEEncoder
from bytepiece.visualization import ChunkViewer, TokenStats, show_tokens

CORPUS: List[str] = [
    "def calculate(): return x >= 10",
    "if status >= 200: return True",
    'url = "https://api.example.com/users"',
    "for i in range(100): print(i)",
]

SAMPLE_TEXT = "def calculate(): return x >= 10"


def _build_encoder() -> BPEEncoder:
    vocab, merges, normalizer = train_bpe(texts=CORPUS, vocab_size=200)
    return BPEEncoder(vocab, merges, normalizer)


def test_show_tokens_inline(capsys) -> None:
    encoder = _build_encoder()
    show_tokens(encoder, SAMPLE_TEXT, style="inline")

    captured = capsys.readouterr()
    assert "Tokens" in captured.out


def test_chunk_viewer_all_styles() -> None:
    encoder = _build_encoder()
    tokens = encoder.encode(SAMPLE_TEXT)
    viewer = ChunkViewer(console=Console(record=True))

    for style in ("inline", "table", "colored"):
        viewer.show_tokens(SAMPLE_TEXT, tokens, style=style)

    rendered = viewer.console.export_text()
    assert "Total" in rendered or "Token Analysis" in rendered


def test_token_stats_summary() -> None:
    encoder = _build_encoder()
    token_stream: List[str] = []
    for line in CORPUS:
        token_stream.extend(encoder.encode(line))

    stats_console = Console(record=True)
    stats = TokenStats(console=stats_console)
    stats.show_stats(token_stream)

    summary = stats_console.export_text()
    assert "Total tokens" in summary
