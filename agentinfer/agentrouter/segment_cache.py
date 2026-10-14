# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Segment-level token cache for the agent front-end (P3 reference core).

This module implements the reusable core of the
"added-token 切分 + 启动 probe + 不命中即回退" design for agent requests:

* a one-shot :func:`evaluate_probe` that decides whether the renderer is
  "message-concatenable", i.e. a longer conversation's token sequence has
  the shorter conversation's token sequence as a prefix;
* a :class:`SegmentTokenCache` that, on a message-boundary prefix hit, only
  tokenizes the added-token tail (``anchor + new messages`` minus the
  anchor) and concatenates it onto the cached prefix;
* fail-open fallback: any precondition that does not hold (probe not ready,
  prefix miss, tail render error, anchor mismatch, multimodal content,
  disallowed flags, over-long conversation) returns the plain full-render
  result, so the cache can only make the tail slower, never wrong.

The module is deliberately free of global mutable state, environment
variables and magic numbers so that it can be reviewed and unit-tested in
isolation. Upstream wiring (token/block offset rewriting inside the native
Router middleware) is not part of this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Message = Mapping[str, Any]
TokenIds = tuple[int, ...]
RenderFn = Callable[[Sequence[Message]], Sequence[int]]

#: Maximum number of messages that may enter the prefix layer. Bounds the
#: per-request cache-key cost (which grows linearly with the message count)
#: and matches the P1 design value.
DEFAULT_MAX_MESSAGES = 64

#: Maximum number of cached token sequences kept in memory (LRU eviction).
DEFAULT_MAX_ENTRIES = 1024

#: Top-level request fields that change the rendering of every message and
#: therefore disable the prefix layer (the exact-match cache stays enabled).
DISALLOWED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "chat_template",
        "documents",
    }
)

#: Boolean top-level flags that make the trailing prompt position special
#: and therefore disable the prefix layer.
DISALLOWED_TOP_LEVEL_FLAGS: frozenset[str] = frozenset(
    {
        "add_generation_prompt",
        "continue_final_message",
        "return_assistant_tokens_mask",
    }
)

_KEY_SEPARATOR = b"\x1e"


class ProbeState(Enum):
    """Three-state result of the one-shot renderer probe."""

    PENDING = "pending"
    READY = "ready"
    DISABLED = "disabled"


class MismatchReason(Enum):
    """Why the probe refused to enable the prefix layer."""

    BOUNDARY_PREFIX_UNSTABLE = "boundary_prefix_unstable"
    ANCHOR_NOT_INDEPENDENT = "anchor_not_independently_renderable"
    MESSAGE_CONTEXT_MISMATCH = "message_context_mismatch"


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of :func:`evaluate_probe`."""

    state: ProbeState
    anchor_tokens: TokenIds = ()
    reason: MismatchReason | None = None


@dataclass(frozen=True)
class SegmentCacheConfig:
    """Tunables for :class:`SegmentTokenCache` (no environment variables)."""

    max_messages: int = DEFAULT_MAX_MESSAGES
    max_entries: int = DEFAULT_MAX_ENTRIES


@dataclass
class SegmentCacheStats:
    """Counters mirroring the design's observability table."""

    exact_hits: int = 0
    prefix_hits: int = 0
    misses: int = 0
    tail_errors: int = 0
    reused_tokens: int = 0
    added_tokens: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "exact_hits_total": self.exact_hits,
            "prefix_hits_total": self.prefix_hits,
            "misses_total": self.misses,
            "tail_errors_total": self.tail_errors,
            "reused_tokens_total": self.reused_tokens,
            "added_tokens_total": self.added_tokens,
        }


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    return len(prefix) <= len(full) and tuple(full[: len(prefix)]) == tuple(prefix)


def _is_multimodal(content: Any) -> bool:
    """True when a message's content carries a non-text block."""

    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, Mapping):
            return True
        if part.get("type") not in (None, "text"):
            return True
    return False


def evaluate_probe(
    render: RenderFn,
    *,
    sample_messages: Sequence[Message],
    anchor_message: Message,
) -> ProbeResult:
    """Run the three probe assertions once and return a three-state result.

    ``sample_messages`` must contain at least four messages ``[a, b, c, d]``.
    The three assertions are:

    1. ``tokens([a, b, c, d])`` starts with ``tokens([a, b])``
       -- message-boundary prefix stability.
    2. ``tokens([anchor, c, d])`` starts with ``tokens([anchor])``
       -- the anchor renders independently.
    3. ``tokens([a, b, c, d])[len(head):] == tokens([anchor, c, d])[len(anchor):]``
       -- a message renders identically inside and outside of context.

    Any failure yields :data:`ProbeState.DISABLED`; the caller then stays on
    the full-render path forever.
    """

    if len(sample_messages) < 4:
        raise ValueError("sample_messages must contain at least four messages")

    a, b, c, d = sample_messages[0], sample_messages[1], sample_messages[2], sample_messages[3]

    head = tuple(render([a, b]))
    full = tuple(render([a, b, c, d]))
    if not _is_prefix(head, full):
        return ProbeResult(ProbeState.DISABLED, reason=MismatchReason.BOUNDARY_PREFIX_UNSTABLE)

    anchor_tokens = tuple(render([anchor_message]))
    anchor_plus = tuple(render([anchor_message, c, d]))
    if not _is_prefix(anchor_tokens, anchor_plus):
        return ProbeResult(ProbeState.DISABLED, reason=MismatchReason.ANCHOR_NOT_INDEPENDENT)

    if tuple(full[len(head) :]) != tuple(anchor_plus[len(anchor_tokens) :]):
        return ProbeResult(ProbeState.DISABLED, reason=MismatchReason.MESSAGE_CONTEXT_MISMATCH)

    return ProbeResult(ProbeState.READY, anchor_tokens=anchor_tokens)


@dataclass
class SegmentTokenCache:
    """Segment-level token cache with fail-open fallback.

    ``render`` maps a sequence of messages to their token ids. The cache is
    keyed by ``hash(envelope || m1 || ... || mk)`` so it is per-message-prefix
    and per-envelope (different sessions never collide).
    """

    render: RenderFn
    anchor_message: Message | None = None
    probe: ProbeResult = field(default_factory=lambda: ProbeResult(ProbeState.PENDING))
    config: SegmentCacheConfig = field(default_factory=SegmentCacheConfig)
    stats: SegmentCacheStats = field(default_factory=SegmentCacheStats)
    _cache: OrderedDict[str, TokenIds] = field(default_factory=OrderedDict, init=False, repr=False)

    def tokens(self, envelope: Mapping[str, Any], messages: Sequence[Message]) -> list[int]:
        """Return token ids for ``messages`` under ``envelope``."""

        message_list = list(messages)
        if not message_list:
            return list(self.render(message_list))

        full_key = self._key(envelope, message_list)
        cached = self._cache.get(full_key)
        if cached is not None:
            self.stats.exact_hits += 1
            self._cache.move_to_end(full_key)
            return list(cached)

        if self._prefix_enabled(envelope, message_list):
            segmented = self._try_segment(envelope, message_list)
            if segmented is not None:
                self._store(full_key, tuple(segmented))
                return segmented

        self.stats.misses += 1
        result = list(self.render(message_list))
        self._store(full_key, tuple(result))
        return result

    def snapshot(self) -> dict[str, Any]:
        """Observability snapshot: probe gauge plus cache counters."""

        return {
            "probe_state": self.probe.state.value,
            "cache_entries": len(self._cache),
            **self.stats.snapshot(),
        }

    def _prefix_enabled(self, envelope: Mapping[str, Any], messages: Sequence[Message]) -> bool:
        if self.probe.state is not ProbeState.READY:
            return False
        if self.anchor_message is None:
            return False
        if len(messages) > self.config.max_messages:
            return False
        if any(envelope.get(name) is not None for name in DISALLOWED_TOP_LEVEL_FIELDS):
            return False
        if any(bool(envelope.get(flag)) for flag in DISALLOWED_TOP_LEVEL_FLAGS):
            return False
        if any(_is_multimodal(message.get("content")) for message in messages):
            return False
        return True

    def _try_segment(self, envelope: Mapping[str, Any], messages: Sequence[Message]) -> list[int] | None:
        anchor_tokens = self.probe.anchor_tokens
        for split in range(len(messages) - 1, 0, -1):
            prefix = self._cache.get(self._key(envelope, messages[:split]))
            if prefix is None:
                continue
            try:
                tail = tuple(self.render([self.anchor_message, *messages[split:]]))
            except Exception:  # noqa: BLE001 - fail-open on any renderer error
                self.stats.tail_errors += 1
                return None
            if tuple(tail[: len(anchor_tokens)]) != anchor_tokens:
                self.stats.tail_errors += 1
                return None
            added = tail[len(anchor_tokens) :]
            self.stats.prefix_hits += 1
            self.stats.reused_tokens += len(prefix)
            self.stats.added_tokens += len(added)
            return list(prefix) + list(added)
        return None

    def _key(self, envelope: Mapping[str, Any], messages: Sequence[Message]) -> str:
        digest = hashlib.sha256()
        digest.update(_canonical(envelope).encode("utf-8"))
        for message in messages:
            digest.update(_KEY_SEPARATOR)
            digest.update(_canonical(message).encode("utf-8"))
        return digest.hexdigest()

    def _store(self, key: str, tokens: TokenIds) -> None:
        self._cache[key] = tokens
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.max_entries:
            self._cache.popitem(last=False)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
