"""Document chunking.

Text is split on paragraph boundaries and packed into chunks of at most
``max_chars`` characters. A single paragraph longer than ``max_chars`` is
split into sliding windows that overlap by ``overlap`` characters so that
context is not lost at window boundaries.
"""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(text, max_chars=400, overlap=80):
    """Split ``text`` into a list of chunk strings.

    Every returned chunk is at most ``max_chars`` long. Raises ValueError
    if ``overlap`` is not smaller than ``max_chars``.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    units = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for unit in units:
        if len(unit) > max_chars:
            flush()
            start = 0
            while start < len(unit):
                chunks.append(unit[start:start + max_chars])
                if start + max_chars >= len(unit):
                    break
                start += max_chars - overlap
        elif len(buffer) + len(unit) + 2 <= max_chars:
            buffer = (buffer + "\n\n" + unit) if buffer else unit
        else:
            flush()
            buffer = unit
    flush()
    return chunks
