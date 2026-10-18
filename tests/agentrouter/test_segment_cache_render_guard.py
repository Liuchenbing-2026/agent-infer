# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Regressions for templates whose context changes beyond the startup probe."""

from __future__ import annotations

import pytest

from agentinfer.agentrouter.segment_cache import ProbeState, SegmentTokenCache, evaluate_probe

ANCHOR = {"role": "system", "content": "anchor"}
SAMPLES = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(4)]
FIFTH = [*SAMPLES, {"role": "user", "content": "fifth turn"}]


class ContextRenderer:
    """All modes pass the four-message probe, then change at five messages."""

    def __init__(self, mode: str = "plain") -> None:
        self.mode = mode
        self.calls: list[int] = []

    def __call__(self, messages):
        self.calls.append(len(messages))
        text = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        if len(messages) >= 5:
            if self.mode == "header":
                text = "LONG-CONTEXT-HEADER\n" + text
            elif self.mode == "footer":
                text += "LONG-CONTEXT-FOOTER\n"
            elif self.mode == "body":
                # Preserve the cached two-message head, but change the suffix.
                text = text.replace("user:2\n", "user:contextual-2\n")
        return list(text.encode("utf-8"))


def ready_cache(render):
    probe = evaluate_probe(render, sample_messages=SAMPLES, anchor_message=ANCHOR)
    assert probe.state is ProbeState.READY
    return SegmentTokenCache(render, anchor_message=ANCHOR, probe=probe)


@pytest.mark.parametrize("mode", ["header", "footer", "body"])
def test_fifth_message_must_equal_complete_render(mode):
    render = ContextRenderer(mode)
    cache = ready_cache(render)
    cache.tokens({}, SAMPLES[:2])
    assert cache.tokens({}, FIFTH) == render(FIFTH)
    assert cache.stats.prefix_hits == 0
    assert cache.snapshot()["render_mismatches_total"] == 1
    assert cache.stats.reused_tokens == 0
    assert cache.stats.added_tokens == 0


def test_fallback_is_cached_for_subsequent_exact_hit():
    render = ContextRenderer("header")
    cache = ready_cache(render)
    cache.tokens({}, SAMPLES[:2])
    expected = render(FIFTH)
    assert cache.tokens({}, FIFTH) == expected
    calls = len(render.calls)
    assert cache.tokens({}, FIFTH) == expected
    assert len(render.calls) == calls
    assert cache.stats.exact_hits == 1
    assert cache.stats.misses == 2


def test_compatible_prefix_requires_full_current_render():
    render = ContextRenderer()
    cache = ready_cache(render)
    cache.tokens({}, SAMPLES[:2])
    render.calls.clear()
    assert cache.tokens({}, FIFTH) == ContextRenderer()(FIFTH)
    assert render.calls == [5, 4]  # Complete render, then anchor + suffix.
    assert cache.stats.prefix_hits == 1
    assert cache.snapshot()["render_mismatches_total"] == 0


def test_every_growing_turn_matches_full_render():
    render = ContextRenderer("header")
    cache = ready_cache(render)
    messages = []
    for i in range(12):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": str(i)})
        assert cache.tokens({}, messages) == render(messages)
    assert cache.stats.prefix_hits > 0
    assert cache.snapshot()["render_mismatches_total"] >= 1


def test_edited_history_does_not_reuse_old_exact_sequence():
    render = ContextRenderer("body")
    cache = ready_cache(render)
    cache.tokens({}, SAMPLES[:2])
    cache.tokens({}, FIFTH)
    edited = [dict(m) for m in FIFTH]
    edited[2]["content"] = "edited 中文 🌍"
    assert cache.tokens({}, edited) == render(edited)


def test_full_render_failure_is_not_hidden_or_cached():
    render = ContextRenderer()

    def fails_on_fifth(messages):
        if len(messages) == 5:
            raise ValueError("full template failed")
        return render(messages)

    cache = ready_cache(fails_on_fifth)
    cache.tokens({}, SAMPLES[:2])
    with pytest.raises(ValueError, match="full template failed"):
        cache.tokens({}, FIFTH)
    assert cache.snapshot()["cache_entries"] == 1
    assert cache.stats.prefix_hits == 0


def test_anchor_failure_returns_already_rendered_full_result():
    render = ContextRenderer()
    fail_anchor = False

    def fails_on_anchor(messages):
        if fail_anchor and messages and messages[0] is ANCHOR:
            raise ValueError("anchor failed")
        return render(messages)

    cache = ready_cache(fails_on_anchor)
    cache.tokens({}, SAMPLES[:2])
    fail_anchor = True
    render.calls.clear()
    assert cache.tokens({}, FIFTH) == ContextRenderer()(FIFTH)
    assert render.calls == [5]
    assert cache.stats.tail_errors == 1
    assert cache.stats.prefix_hits == 0


def test_returned_list_cannot_mutate_cached_full_result():
    render = ContextRenderer("header")
    cache = ready_cache(render)
    cache.tokens({}, SAMPLES[:2])
    result = cache.tokens({}, FIFTH)
    result.clear()
    assert cache.tokens({}, FIFTH) == render(FIFTH)
