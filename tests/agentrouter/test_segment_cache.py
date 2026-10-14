# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Unit tests for the segment-level token cache (P3 core)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentinfer.agentrouter.segment_cache import (
    MismatchReason,
    ProbeResult,
    ProbeState,
    SegmentCacheConfig,
    SegmentTokenCache,
    evaluate_probe,
)

ANCHOR = {"role": "system", "content": "anchor"}


def msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _encode(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class ConcatRenderer:
    """Deterministic, message-concatenable renderer (simulated chat template)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        self.calls += 1
        out: list[int] = []
        for message in messages:
            out.extend(_encode(f"{message.get('role', 'user')}:{message.get('content', '')}\n"))
        return out


class HeaderRenderer(ConcatRenderer):
    """Prefix-unstable: a length header breaks message-boundary prefix stability."""

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        return [len(messages)] + super().__call__(messages)


class SoloAnchorRenderer(ConcatRenderer):
    """The anchor renders differently alone than in context (assertion 2 fails)."""

    def __init__(self, anchor: Mapping[str, Any]) -> None:
        super().__init__()
        self._anchor = anchor

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        if len(messages) == 1 and messages[0] is self._anchor:
            return _encode("ANCHOR-SOLO")
        return super().__call__(messages)


class ContextSensitiveRenderer(ConcatRenderer):
    """A message renders differently inside vs outside the four-message context."""

    def __init__(self, abcd: Sequence[Mapping[str, Any]]) -> None:
        super().__init__()
        self._abcd = list(abcd)

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        out = super().__call__(messages)
        if len(messages) == len(self._abcd) and all(m is s for m, s in zip(messages, self._abcd, strict=True)):
            out = out + _encode("EXTRA")
        return out


class ToggleAnchorRenderer(ConcatRenderer):
    """Passes the probe, then changes its anchor render to force a tail fallback."""

    def __init__(self, anchor: Mapping[str, Any]) -> None:
        super().__init__()
        self._anchor = anchor
        self.flip = False

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        if self.flip and messages and messages[0] is self._anchor:
            return _encode("FLIPPED") + super().__call__(messages)
        return super().__call__(messages)


def _samples() -> list[dict[str, Any]]:
    return [msg("user", "a"), msg("assistant", "b"), msg("user", "c"), msg("assistant", "d")]


def _ready_cache(render: ConcatRenderer, anchor: Mapping[str, Any] | None = None) -> SegmentTokenCache:
    anchor = ANCHOR if anchor is None else anchor
    probe = evaluate_probe(render, sample_messages=_samples(), anchor_message=anchor)
    assert probe.state is ProbeState.READY
    return SegmentTokenCache(render, anchor_message=anchor, probe=probe)


ENVELOPE: dict[str, Any] = {"model": "demo", "session_id": "s1"}


# --------------------------------------------------------------------------- probe


def test_probe_ready_on_concatenable_renderer() -> None:
    render = ConcatRenderer()
    probe = evaluate_probe(render, sample_messages=_samples(), anchor_message=ANCHOR)
    assert probe.state is ProbeState.READY
    assert probe.anchor_tokens == tuple(render([ANCHOR]))


def test_probe_disabled_when_boundary_prefix_unstable() -> None:
    probe = evaluate_probe(HeaderRenderer(), sample_messages=_samples(), anchor_message=ANCHOR)
    assert probe.state is ProbeState.DISABLED
    assert probe.reason is MismatchReason.BOUNDARY_PREFIX_UNSTABLE


def test_probe_disabled_when_anchor_not_independent() -> None:
    probe = evaluate_probe(SoloAnchorRenderer(ANCHOR), sample_messages=_samples(), anchor_message=ANCHOR)
    assert probe.state is ProbeState.DISABLED
    assert probe.reason is MismatchReason.ANCHOR_NOT_INDEPENDENT


def test_probe_disabled_when_message_context_mismatch() -> None:
    samples = _samples()
    probe = evaluate_probe(ContextSensitiveRenderer(samples), sample_messages=samples, anchor_message=ANCHOR)
    assert probe.state is ProbeState.DISABLED
    assert probe.reason is MismatchReason.MESSAGE_CONTEXT_MISMATCH


# --------------------------------------------------------------------------- cache


def test_exact_cache_hit() -> None:
    cache = _ready_cache(ConcatRenderer())
    messages = _samples()
    first = cache.tokens(ENVELOPE, messages)
    second = cache.tokens(ENVELOPE, messages)
    assert first == second
    assert cache.stats.exact_hits == 1


def test_prefix_reuse_matches_full_render() -> None:
    render = ConcatRenderer()
    cache = _ready_cache(render)
    first_round = _samples()
    assert cache.tokens(ENVELOPE, first_round) == list(render(first_round))

    for extra in ("e", "f", "g"):
        first_round = first_round + [msg("user", extra)]
        split_result = cache.tokens(ENVELOPE, first_round)
        assert split_result == list(render(first_round))

    assert cache.stats.prefix_hits >= 1
    assert cache.stats.reused_tokens > 0
    assert cache.stats.added_tokens > 0


def test_tail_mismatch_falls_back_to_full_render() -> None:
    render = ToggleAnchorRenderer(ANCHOR)
    probe = evaluate_probe(render, sample_messages=_samples(), anchor_message=ANCHOR)
    assert probe.state is ProbeState.READY
    cache = SegmentTokenCache(render, anchor_message=ANCHOR, probe=probe)

    first_round = _samples()
    cache.tokens(ENVELOPE, first_round)

    render.flip = True
    second_round = first_round + [msg("user", "e")]
    result = cache.tokens(ENVELOPE, second_round)
    assert result == list(render(second_round))
    assert cache.stats.tail_errors == 1
    assert cache.stats.prefix_hits == 0


def test_multimodal_content_disables_prefix_layer() -> None:
    cache = _ready_cache(ConcatRenderer())
    first_round = _samples()
    cache.tokens(ENVELOPE, first_round)
    multimodal = first_round + [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    cache.tokens(ENVELOPE, multimodal)
    cache.tokens(ENVELOPE, multimodal)
    assert cache.stats.prefix_hits == 0
    assert cache.stats.exact_hits == 1


def test_over_long_conversation_degrades_to_exact_cache() -> None:
    render = ConcatRenderer()
    anchor = ANCHOR
    probe = evaluate_probe(render, sample_messages=_samples(), anchor_message=anchor)
    cache = SegmentTokenCache(
        render,
        anchor_message=anchor,
        probe=probe,
        config=SegmentCacheConfig(max_messages=4),
    )
    first_round = _samples()
    cache.tokens(ENVELOPE, first_round)
    second_round = first_round + [msg("user", "e")]
    cache.tokens(ENVELOPE, second_round)
    cache.tokens(ENVELOPE, second_round)
    assert cache.stats.prefix_hits == 0
    assert cache.stats.exact_hits == 1


def test_disallowed_flag_disables_prefix_layer() -> None:
    cache = _ready_cache(ConcatRenderer())
    envelope = {**ENVELOPE, "add_generation_prompt": True}
    first_round = _samples()
    cache.tokens(envelope, first_round)
    cache.tokens(envelope, first_round + [msg("user", "e")])
    assert cache.stats.prefix_hits == 0


def test_different_envelope_does_not_collide() -> None:
    cache = _ready_cache(ConcatRenderer())
    messages = _samples()
    cache.tokens({**ENVELOPE, "session_id": "s1"}, messages)
    cache.tokens({**ENVELOPE, "session_id": "s2"}, messages)
    assert cache.stats.exact_hits == 0


def test_pending_probe_uses_full_render_only() -> None:
    render = ConcatRenderer()
    cache = SegmentTokenCache(render, anchor_message=ANCHOR, probe=ProbeResult(ProbeState.PENDING))
    messages = _samples()
    assert cache.tokens(ENVELOPE, messages) == list(render(messages))
    cache.tokens(ENVELOPE, messages + [msg("user", "e")])
    assert cache.stats.prefix_hits == 0


def test_empty_messages_short_circuits() -> None:
    render = ConcatRenderer()
    cache = _ready_cache(render)
    assert cache.tokens(ENVELOPE, []) == list(render([]))
    assert cache.stats.misses == 0
    assert cache.stats.exact_hits == 0


def test_snapshot_reports_probe_and_counters() -> None:
    cache = _ready_cache(ConcatRenderer())
    cache.tokens(ENVELOPE, _samples())
    snapshot = cache.snapshot()
    assert snapshot["probe_state"] == "ready"
    assert snapshot["misses_total"] == 1
    assert set(snapshot) >= {"probe_state", "cache_entries", "prefix_hits_total", "reused_tokens_total"}
