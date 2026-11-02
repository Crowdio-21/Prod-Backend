"""
File-backed payload reference storage for large task args/results.

SQLite stores only a small reference string when payload text exceeds the
configured threshold.
"""

from __future__ import annotations

import gzip
import os
import uuid


PAYLOAD_REF_PREFIX = "payload_ref://"
MAX_INLINE_BYTES = 128 * 1024
_ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".payload_store"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _ensure_dir(category: str) -> str:
    path = os.path.join(_ROOT_DIR, _safe_name(category))
    os.makedirs(path, exist_ok=True)
    return path


def is_payload_ref(value: str | None) -> bool:
    return bool(value) and value.startswith(PAYLOAD_REF_PREFIX)


def store_text_if_large(
    text: str | None,
    category: str,
    key_hint: str,
    max_inline_bytes: int = MAX_INLINE_BYTES,
) -> str | None:
    """
    Persist text externally when it exceeds max_inline_bytes.

    Returns original text for small payloads, otherwise returns a payload_ref.
    """
    if text is None:
        return None

    if is_payload_ref(text):
        return text

    encoded = text.encode("utf-8")
    if len(encoded) <= max_inline_bytes:
        return text

    target_dir = _ensure_dir(category)
    file_name = f"{_safe_name(key_hint)}_{uuid.uuid4().hex}.json.gz"
    abs_path = os.path.join(target_dir, file_name)

    with gzip.open(abs_path, "wb") as fh:
        fh.write(encoded)

    rel_path = os.path.relpath(abs_path, _ROOT_DIR).replace("\\", "/")
    return f"{PAYLOAD_REF_PREFIX}{rel_path}"


def resolve_text_ref(value: str | None) -> str | None:
    """
    Resolve payload_ref://... back to the original text.

    If loading fails, returns the original value unchanged.
    """
    if value is None or not is_payload_ref(value):
        return value

    rel_path = value[len(PAYLOAD_REF_PREFIX) :]
    abs_path = os.path.join(_ROOT_DIR, rel_path.replace("/", os.sep))

    try:
        with gzip.open(abs_path, "rb") as fh:
            return fh.read().decode("utf-8")
    except Exception:
        return value


def payload_ref_path(value: str | None) -> str | None:
    """Absolute path of a ``payload_ref://`` file, or None if ``value`` is not a reference."""
    if not is_payload_ref(value):
        return None
    return os.path.join(_ROOT_DIR, value[len(PAYLOAD_REF_PREFIX):].replace("/", os.sep))


def payload_ref_size(value: str | None) -> int | None:
    """Uncompressed size in bytes of a referenced payload (what is sent over the wire).

    Read from the gzip trailer (ISIZE, the last four bytes), so the file is not decompressed.
    """
    path = payload_ref_path(value)
    if path is None:
        return None
    try:
        with open(path, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            return int.from_bytes(fh.read(4), "little")
    except OSError:
        return None


def payload_ref_head(value: str | None, n_bytes: int = 4096) -> str | None:
    """The first ``n_bytes`` of a referenced payload, decompressed (None if unreadable)."""
    path = payload_ref_path(value)
    if path is None:
        return None
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read(n_bytes).decode("utf-8", errors="replace")
    except Exception:
        return None
