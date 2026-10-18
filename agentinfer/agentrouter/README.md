# AgentRouter integrations

This directory does **not** vendor the Router source tree. It carries:

1. **WASM affinity guest** (preferred) — `agent_hint_affinity/`
2. **Native middleware patch** (optional) — still needed for
   `agent_hint_token_offsets` only; affinity should use WASM instead

## Preferred: WASM `agent_hint_affinity`

Upstream Router ([vllm-project/router](https://github.com/vllm-project/router)
`#251`+) provides the Wasmtime OnRequest host. AgentInfer ships the affinity
**plugin** only.

| Item | Value |
| --- | --- |
| Upstream host | `--wasm-middleware` / `--wasm-middleware-sha256` / `--wasm-middleware-route` |
| Upstream WIT | `vllm:router-middleware@0.1.0` |
| Guest crate | [`agent_hint_affinity/`](agent_hint_affinity/) |
| Semantics | Copy `agent_hint.session_id` → `session_params.session_id` when missing |

### Build + run

```bash
# 1) Build the guest
./agentinfer/agentrouter/agent_hint_affinity/build.sh

# 2) Run upstream Router (must include WASM host from #251+)
DIGEST=$(sha256sum agentinfer/agentrouter/agent_hint_affinity/agent_hint_affinity.component.wasm | awk '{print $1}')

vllm-router \
  --worker-urls http://localhost:8080 \
  --policy cache_aware \
  --wasm-middleware ./agentinfer/agentrouter/agent_hint_affinity/agent_hint_affinity.component.wasm \
  --wasm-middleware-sha256 "$DIGEST" \
  --wasm-middleware-route /v1/chat/completions
```

Details: [`agent_hint_affinity/README.md`](agent_hint_affinity/README.md).

## Optional: native patch for `agent_hint_token_offsets`

Token/block offset rewriting is **not** implemented as WASM in this revision.
If you need it, apply the native patch below onto a Router checkout.

### Upstream pin (native patch base)

| Item | Value |
| --- | --- |
| Upstream project | [vllm-project/router](https://github.com/vllm-project/router) |
| Patch base revision | `d60711dc72ab8f073e33f9a3d93ee81b97274c26` |
| Patch file | `patches/0001-agent-hint-native-middleware.patch` |

The historical patch also contains native `agent_hint_affinity`. Prefer the WASM
guest above for affinity; do **not** enable native affinity together with
`--wasm-middleware`.

### What the patch adds

- `--middleware agent_hint_affinity` — legacy native affinity (prefer WASM)
- `--middleware agent_hint_token_offsets` — rewrite message token offsets into
  KV block offsets (requires tokenizer / block-size flags)

Plus supporting CLI/PyO3 wiring, protocol `AgentHint` fields, and
`tests/test_agent_hint_routing.rs`.

### Apply

```bash
git clone https://github.com/vllm-project/router.git
cd router
git checkout d60711dc72ab8f073e33f9a3d93ee81b97274c26

git apply /path/to/agent-infer/agentinfer/agentrouter/patches/0001-agent-hint-native-middleware.patch

cargo build --release
./target/release/vllm-router --help   # look for --middleware …
```

Dry-run:

```bash
git apply --check /path/to/agent-infer/agentinfer/agentrouter/patches/0001-agent-hint-native-middleware.patch
```

When combining token_offsets with WASM affinity, use a Router that has both the
WASM host (`#251`+) and the token_offsets portion of the native patch (rebase the
patch onto a `#251`+ revision if needed).

## Out of scope

- Vendoring or shipping the full Router / `agentrouter/` binary tree in this repo
- WASM implementation of `agent_hint_token_offsets`

## Segment token cache reference core

`segment_cache.py` is an isolated Python reference core, not native Router
middleware wiring. A successful four-message startup probe only establishes
sample compatibility. Before accepting a prefix hit, the core renders the
complete current request, checks that the cached token IDs remain its prefix,
and checks the reconstructed suffix against the complete result. Incompatible
headers, footers or contextual message rendering fall back to that full result;
`render_mismatches_total` counts such rejected candidates.

The current renderer callback already returns token IDs. This conservative
validation therefore performs full tokenization on every non-exact request;
prefix-hit and reused-token counters are **not** saved-tokenization measurements.
Exact hits still avoid rendering. Keep renderer configuration deterministic and
include model/template revisions and render options in the cache envelope, or
create a fresh cache when they change.

Focused checks, in the repository development environment:

```bash
python -m pytest tests/agentrouter/test_segment_cache.py
python -m pytest tests/agentrouter/test_segment_cache_render_guard.py
```

These cover the Python cache core, including context changes first appearing at
the fifth message. They do not validate native middleware, model generation,
KV reuse, NPU execution or performance.
