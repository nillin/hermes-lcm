"""Token counting utilities for LCM.

Uses tiktoken when available, with a script-aware fallback when it is not.
"""

import logging
import math
import unicodedata
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_encoder = None
_encoder_checked = False


def _get_encoder():
    """Lazily load tiktoken cl100k_base encoder."""
    global _encoder, _encoder_checked
    if _encoder_checked:
        return _encoder
    _encoder_checked = True
    try:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.debug("tiktoken not available, using char-based estimates")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return _estimate_tokens_fallback(text)



def _estimate_tokens_fallback(text: str) -> int:
    """Rough token estimate without tiktoken.

    Uses a conservative, script-aware heuristic so CJK/emoji text is not
    drastically undercounted like a naive ``len(text) / 4`` estimate.
    """
    ascii_chars = 0
    non_ascii_wordish = 0
    cjk_chars = 0
    emoji_or_symbols = 0

    for ch in text:
        if ch.isspace():
            continue
        codepoint = ord(ch)
        if ch.isascii():
            ascii_chars += 1
            continue
        if _is_cjk_char(codepoint):
            cjk_chars += 1
            continue
        category = unicodedata.category(ch)
        if category.startswith("S") or category in {"So", "Sk"}:
            emoji_or_symbols += 1
            continue
        non_ascii_wordish += 1

    estimate = 0
    if ascii_chars:
        estimate += math.ceil(ascii_chars / 4)
    if non_ascii_wordish:
        estimate += math.ceil(non_ascii_wordish * 1.25)
    if cjk_chars:
        estimate += cjk_chars * 2
    if emoji_or_symbols:
        estimate += emoji_or_symbols * 2

    return max(1, estimate)



def _is_cjk_char(codepoint: int) -> bool:
    """Return True for CJK ideographs, kana, and hangul ranges."""
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x309F
        or 0x30A0 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
    )


def count_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single OpenAI-format message."""
    total = 4  # role + overhead
    content = msg.get("content") or ""
    total += count_tokens(content)
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            total += count_tokens(fn.get("name", ""))
            total += count_tokens(fn.get("arguments", ""))
        total += 3  # per-call overhead
    return total


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a message list."""
    return sum(count_message_tokens(m) for m in messages)
