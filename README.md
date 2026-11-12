# llama-cpp

llama.cpp server with an embedded Web UI, packaged as a Docker Custom App for
TrueNAS SCALE (Electric Eel). Built for an **NVIDIA RTX 4060 Ti (16 GB, Ada —
sm_89)** with CPU kernels targeting an **AMD Ryzen 9 3900X (Zen 2)**.

## Why a custom build

The build host's CUDA toolkit (13.3) is newer than the TrueNAS NVIDIA driver
(570.172.08 → max CUDA 12.8) supports, so a 13.3-compiled binary fails at
runtime on the 4060 Ti. This image builds llama.cpp inside a CUDA 12.8 devel
container, pinning `CMAKE_CUDA_ARCHITECTURES=89-real` and the Zen 2-safe
`-DGGML_*` CPU flags. The cmake flag set mirrors the Gentoo ebuild
`sci-misc/llama-cpp-0_pre10636.ebuild` (commit `b10636`).

The Web UI is built from source (vite + SvelteKit) and embedded into the
`llama-server` binary, so a single process serves both the API and the chat
interface on port 8080.

## Served model

The image is model-agnostic — any GGUF model can be served by setting
`MODEL_PATH` (and optionally `MMPROJ_PATH` for vision). The shipped
`docker-compose.yml` serves Unsloth Dynamic V3.1
`Qwen3.8-27B-UD-IQ3_XXS.gguf` (10.93 GB / 10.18 GiB, ~3.2 bpw; MTP head
embedded, `nextn_predict_layers = 1`, `blk.64.nextn.*`) with the KV-stream
arena. This is the base Qwen3.8-27B model from the Unsloth release, not the
Huihui abliterated variant; swap models by changing the env vars — no rebuild
needed. The existing Q8_0 projector is reused; if it is incompatible with the
Unsloth model, use the Unsloth `mmproj-F16.gguf`.

For NVIDIA RTX 40-series (Ada Lovelace), use `mmproj-F16.gguf` (not BF16) —
Ada has native F16 tensor cores but lacks native BF16 support.

## Quick start (TrueNAS)

1. Apps → Custom Apps → **Install via YAML** → paste `docker-compose.yml`.
2. Point the `volumes` entry at your models dataset (SSD recommended).
3. Install. The Web UI comes up at `http://<nas-ip>:30084`.

## Build locally

```sh
./build.sh --local
```

`./build.sh` defaults to the latest release **including pre-releases**; pass a
tag to pin it. Both tag styles work: semver releases (`v0.9.0`) and the rolling
`bNNNN` builds (`b12110`, the same commit). Semver tags are mapped back to their
build number through the release's `nightly-tag.txt` asset, because
`build-info.sh` emits `int LLAMA_BUILD_NUMBER = N;` and the Dockerfile forwards
`--build-arg LLAMA_BUILD_NUMBER`.

Override the CUDA arch or llama.cpp tag via build args if needed:

```sh
docker buildx build --build-arg CMAKE_CUDA_ARCHITECTURES=86-real \
  --build-arg LLAMA_TAG=v0.9.0 --build-arg LLAMA_BUILD_NUMBER=12110 \
  -t llama-cpp:local .
```

### Adaptive KV streaming V2 (shared device arena) variant

```sh
./build.sh v0.9.0 --kv-stream --local
```

Applies the adaptive KV streaming **V2** patch (derived from
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming),
branch `feature/adaptive-kv-stream-v2`) on top of the standard build. V2
keeps the complete KV history in pinned host RAM and serves attention from
a bounded, device-local arena of resident pages plus one shared ring; the
same arena is phase-shared with compute workspaces (and the vision
projector). The patch file is chosen by tag: `v0.9.0` / `b12110` →
`patches/adaptive-kv-stream-v2-v0.9.0.patch`; `v0.8.0` / `b11960` →
`patches/adaptive-kv-stream-v2-v0.8.0.patch`; `b11559` →
`patches/adaptive-kv-stream-v2-b11559.patch` (kept for rollback); other tags
error out in `build.sh`, and a mismatched base would fail the
`git apply --check` guard in the Dockerfile. The binary gains `--kv-stream-arena-mib N` (shared
arena MiB; requires `-fa on` + `--fit off`, auto-derived MTP KV layer
count) and the legacy `--kv-stream-pool-mib N`, exposed as the
`KV_STREAM_ARENA_MIB` env var. If that env var is set on a
non-`--kv-stream` image, the entrypoint detects the missing flag and
continues without it instead of failing startup; it also skips the arena
in `CTX_SIZE=auto` mode, which runs `--fit on`. Note V2 renamed the old
`--kv-stream-stage-mib` flag (V1 patches, removed) to
`--kv-stream-pool-mib`.

### Patch provenance (v0.9.0 / b12110)

The v0.9.0 patch is a real 3-way port, not a re-generated copy of the fork's
branch: the fork's `feature/adaptive-kv-stream-v2` is based on upstream
`10a60cf` ≈ `b11539`, behind v0.9.0 (`cd959ba`, b12110). The v0.8.0 patch does
not apply cleanly to v0.9.0; a 3-way replay leaves 12 conflicting files and 32
conflict hunks. The v0.9.0 upstream changes that collided — `flash_attn_ext_rows`
(unified KV rows), `ggml` 0.29 FP8/NVFP4, `ggml_backend_tensor_set_n`, DFlash/
DSpark batch rows, changed MTP/speculative state — were merged as follows:

- `llama_memory_params` keeps `no_alloc`; streamed and leased caches adopt
  their externally created buffer into `ctxs_bufs.bufs`, and
  `llama_memory_bufs` gained `size()`/`front()`/`swap()` for that path.
- `n_pad` is gone from `llama_kv_cache`; the streamed cache keeps its 256-row
  alignment with `GGML_PAD(kv_size, 256)` in the stream branch only.
- `kv-cache` causal-mask selection follows v0.9.0's `SWA_ONLY`/`DENSE_ONLY`
  split; v0.9.0's `set_input_attn_mask(llm_attn_mask)` is kept, and the fork's
  `mask_is_causal()` / `set_input_kq_mask()` / mask-bounds path is added. The
  context dispatches I32 compact masks through `set_input_kq_mask()`.
- `llama-context`: `sched_need_reserve` is replaced by
  `sched_reserve_state.invalidate()`; v0.9.0's `embd_row` validation and the
  fork's `kv_shared_read` / `kv_streaming` validation are both retained.
  `llama_io_read_host` keeps the fork's `flush_tensor_reads()` but applies it
  with v0.9.0's batched `ggml_backend_tensor_set_n`.
- Three hidden-state channels coexist: v0.9.0's per-token `embd_state`,
  v0.9.0's DFlash/DSpark `embd_row`, and the fork's borrowed `embd_h` buffer.
  `llama_batch_allocr` copies `embd_h` per ubatch into `data_t::embd_h`, and
  `llm_graph_input_embd_h::set_input` prefers `embd_state` and falls back to
  `embd_h`. Wiring only one channel is silent until the first image request.
- `graph`: v0.9.0's `kv_rows` / `ggml_flash_attn_ext_rows` path is combined
  with the fork's compact-mask and external-workspace flags; `fattn-mma-f16`
  keeps the fork's `kv.template load` / staged-prefetch implementation.
- CUDA wrappers: v0.9.0's `fattn_kernel_t` carries `KV_rows_ptr`, so
  `flash_attn_ext_f16` and `flash_attn_ext_vec` were updated to match.
  Generated kv-stream kernels pass `nullptr` for rows, while resume kernels
  keep the one-pointer `ggml_kv_resume_kernel_t` signature. This was the
  main CUDA port bug found by the sm_89 compile.
- `delta-net-base` keeps the fork's staged/spill recurrent publication loop
  (v0.9.0's `set_rows` path is replaced by the spill-aware version).
- `kv-cache` rotation: v0.9.0 keeps one Hadamard rotation matrix per layer as
  constant weights, allocated on the KV buffer's type. The streaming cache's
  host KV is an `Execution_Host` wrapper whose `alloc_buffer` intentionally
  returns `nullptr`, so the 4.25 MiB rotation buffer could not be created and
  model load aborted (`failed to allocate buffer for rotation matrices`).
  Rotation matrices now go to the device that owns the storage, and the
  host-backed MTP cache uses the streaming backend's device so the draft
  graph does not fall back to CPU hadamard. (v0.8.0 fed rotation to the graph
  as an input tensor, which is why the old image never hit this.)
- `src[5]` collision: v0.9.0 stores the `kv_rows` cell list of a
  flash-attention node in `src[5]`, and the streaming patch stores its compact
  prefill mask bounds in the same slot. The managed streaming/shared-read
  attention rejects `src[5]` rows, so with `--kv-unified` the `kv_rows` probe
  found no backend and `sched_reserve` hit
  `GGML_ASSERT(*cur_backend_id != -1)`. Streamed, shared-read and
  streaming-parent contexts now disable `kv_rows` (they encode per-query
  ranges in the mask instead, as v0.8.0 did), and the execution-owner check
  only treats `src[5]` as compact bounds when `src[3]` is absent, so upstream
  non-streamed rows still work.
- `common_fit_apply()` is v0.9.0's rewrite; the fork's streamed-MTP / DFlash
  validation helpers are appended and called before it.
- `mtmd`: v0.9.0's LFM3-VL preprocessor and the fork's Gemma 4 preprocessor
  are both kept. In `clip.cpp`, the fork's `weight_source` loader replaces
  `fin`/`tensor_offset`; the two voice-tensor loops were ported to
  `weight_source->read()` (a real port bug found during the CPU compile).
- `common/speculative.cpp`: the fork's versioned MTP `get_state`/`set_state`
  is kept; v0.9.0's duplicate simple version was removed.
- Tests: v0.9.0's upstream `tests/CMakeLists.txt` is kept; fork test files are
  added but not compiled because the image builds with `LLAMA_BUILD_TESTS=OFF`.

### v0.9.0 validation (2026-11-11)

- CPU build of the ported tree: `llama-server` compiles, and the generation
  path was exercised with the previous Huihui GGUF (`-fa off`, f16 V). The
  Unsloth `Qwen3.8-27B-UD-IQ3_XXS` metadata was verified separately:
  `qwen35`, 65 blocks, `nextn_predict_layers = 1`, blk.64 nextn tensors.
  CPU generation with `-fa on` is not usable because the CPU backend does not
  implement `flash_attn_ext_rows`; that is expected, and the deployed image is
  CUDA.
- CUDA sm_89 image built from `v0.9.0` + the V2 patch and pushed:
  binary `0.9.0-dev (build 12110, commit v0.9.0+kvstream)`,
  `--kv-stream-arena-mib` and the MTP flags present in the image binary.
  Current digest: `sha256:4e72528de21877743f8846ea683d8232bf84c828801c1e8bed9542ffb563d7e4`
  (image `sha256:5b399471cec1230adb763dec014832d2384d10c31d0dde6ed150cdbac3024a94`).
  This image carries both port fixes above (rotation placement and the
  `src[5]` `kv_rows` collision).
- First 4060 Ti deploy (05:54 UTC 2026-11-12) aborted during context creation:
  `failed to allocate Execution_Host buffer of size 4456448` →
  `failed to allocate buffer for rotation matrices`. The image was rebuilt
  with the rotation placement fix described above; this digest supersedes the
  broken one.
- Second 4060 Ti deploy (14:56 UTC, rotation disabled via env) got past the
  rotation buffer and aborted in `sched_reserve` with
  `GGML_ASSERT(*cur_backend_id != -1)` — the `src[5]` `kv_rows`/compact-mask
  collision above. Both the probe and the streamed graph are now built with
  `kv_rows` disabled for streaming contexts; `LLAMA_KV_ROWS_DISABLE=1`
  exercises the same path on older images.
- Third 4060 Ti deploy (15:48 UTC 2026-11-12) is healthy on digest
  `4e72528d...`, with the production env and no `LLAMA_ATTN_ROT_DISABLE` /
  `LLAMA_KV_ROWS_DISABLE` overrides:
  - `init_rotation: CUDA0 rotation buffer size = 4.25 MiB` (target) and
    `CUDA0_KV_Host rotation buffer size = 0.27 MiB` (MTP cache)
  - `KV streaming active, disabling kv_rows`
  - arena: `shared budget 2048.00 MiB, temporary device grants 215.58 MiB,
    authoritative host 3536.00 MiB` (3328 target + 208 MTP auxiliary),
    `resident_pages=245 ring_slots=13` at boot, `293/14` at 22K tokens
  - `adaptive MTP span attention active (first TG4, reserved=22190 tokens)`,
    draft acceptance 0.83–0.89 at 22–27K (mean draft len 2.4–2.7), 0.71 at 36K
    after an image
  - prefill ~700–750 tok/s at 22K; decode 28.5 / 31.8 / 33.3 / 29.9 t/s at
    22K / 22.7K / 26.8K / 31K, 24.6 t/s at 36K. These are interactive turns,
    not `mtp-bench.sh` runs, so treat them as indicative.
  - vision: the reused Q8_0 projector works with the Unsloth base model.
    A 1472x1472 image: projector reload 2.69 s, encode 5.42 s (2 batches),
    KV suspend 0.39 ms / resume 4.3 ms; the answer was correct.
  - `KV_reload` on the prefill→decode transition at 35K: 934 MB in 70 ms
    (~13 GB/s over PCIe), `calls=32`.
- Image batches on this hybrid (SSM + attention) model log
  `find_slot: non-consecutive token position ...` from the recurrent-state
  path: M-RoPE image tokens do not use consecutive positions, and upstream's
  handler is a warning with a TODO. Output is unaffected.
- The MTP cache's rotation matrices currently land on the pinned host buffer
  (`CUDA0_KV_Host`, 0.27 MiB) because the draft context carries no streaming
  backend; it costs nothing measurable at 1 layer. Moving them to the device
  is a possible micro-optimization.

### v0.8.0 validation (previous, 2026-11-01)

- CPU build of the merged tree: `llama-server` compiles, loads the abliterated
  GGUF (`qwen35`, blk.64 MTP tensors detected) and generates through
  `/v1/chat/completions`.
- CUDA image, RTX 5090, sm_89 code (PTX JIT), production env (arena 2048,
  `CTX_SIZE=131072`, q8_0/q4_0, MTP draft-2, Q8_0 mmproj):
  - `KV stream bootstrap pool 7.31 MiB, shared budget 2048.00 MiB,
    temporary device grants 215.58 MiB, authoritative host 3536.00 MiB`
    (3328 MiB target history + 208 MiB MTP auxiliary layer)
  - `memory_phase: phase=prefill compute=149 MiB kv_pool=1780 MiB
    resident_pages=245 ring_slots=13`
  - `vision arena enabled: deferred projector weights and bounded serial phase
    grants`; an image request suspends KV, borrows the projector into the arena
    and answers correctly
  - `MTP speculative decoding active`; `mtp-bench.sh` (256 tokens, JIT'd sm_89
    kernels on sm_120, so the absolute rate is not the 4060 Ti rate):
    97 t/s decode, 632 prompt t/s, 77-85% draft acceptance.
- The 4060 Ti numbers still have to be re-measured on the NAS.

Rebuild/regenerate the patch from a merge of
`v0.9.0` + `feature/adaptive-kv-stream-v2` and diff against the tag commit
(`git diff cd959ba HEAD`), never against a fork branch tip.

### KV-stream V2 hard rules + 4060 Ti qualification

The rules below were established on b11559 and are unchanged by the v0.9.0
port: the hard-rule code paths (`llama_kv_cache` ctor guards, `common_*`
config errors, `server_vision_arena_*`) were ported, and the model geometry
that drives the numbers (`qwen35`, 64+1 blocks, 16 full-attention KV layers,
4 KV heads x 256 head_dim) is identical across the Qwen3.8-27B GGUFs.

Hard startup rules enforced by the patch — violations exit at model-load
time, before any weights load:

- `CTX_SIZE` must be a number: `auto` runs `--fit on`, and the entrypoint
  skips the arena in that mode.
- **Vision + arena is hard-qualified to `CTK=q8_0 CTV=q4_0`** — with both
  `MMPROJ_PATH` and the arena set, any other cache types are rejected
  (`vision arena is currently qualified with -fa on, Q8_0 K and Q4_0 V`).
  Text-only arena (no mmproj) is not gated.
- Attached MTP inherits the target's KV types. As of the v0.9.0 port the fork
  rejects explicit draft overrides outright when streaming is active
  (`adaptive KV speculation inherits target cache types; remove draft KV
  overrides`), so the entrypoint drops `--spec-draft-type-k/-v` whenever the
  arena is on — passing `CTKD`/`CTVD` there aborts the MTP launch and the
  probe falls back to plain decode. They still apply when
  `KV_STREAM_ARENA_MIB=0`. Draft lengths 1–3; keep `SPEC_DRAFT_N_MAX` in that
  range.
- `MMPROJ_OFFLOAD=1` — CPU projector offload is rejected on
  vision+arena builds.

Qualification run (2026-10-11, RTX 4060 Ti 16 GB — first real-hardware
SM89 result; the fork itself qualified SM120 hardware only):
Qwen3.8-27B GSQ-RCO IQ3_XXS-mtp (9.72 GiB, 3.05 bpw mixed),
`CTX_SIZE=131072`, `CTK=q8_0 CTV=q4_0 CTKD=q8_0 CTVD=q4_0`,
`TRY_MTP=1 SPEC_DRAFT_N_MAX=2`, `KV_STREAM_ARENA_MIB=2048`,
Q8_0 mmproj offloaded, `UBATCH_SIZE=512`.

- Startup: the full **3328 MiB** KV history pinned in host RAM; the GPU
  holds only the 2 GiB arena. MTP auxiliary KV layer auto-derived from
  model metadata (`nextn_predict_layers = 1`).
- Phase arena re-lays out per phase: prefill ≈ 776 MiB compute /
  1153 MiB KV pool; decode ≈ 15 MiB compute / 2129 MiB KV pool
  (761 MiB reclaimed from the prefill workspace).
- ≈75K tokens fit fully resident at 2048 MiB (293 pages × 256 tokens +
  ring slots); beyond that `streaming=1` and `KV_reload` H2D traffic
  appears — that is the feature working, not a fault.
- The Q8_0 mmproj was accepted (fork docs qualified only F16): the
  projector borrows arena regions, KV suspends for image batches and
  refills afterwards (695 MiB H2D in ~52 ms).
- Speed: prefill ~570 t/s, decode **~16.7 t/s** at 22–26K ctx with MTP
  draft-2 active.

Boot-time confirmation lines (WARN level, visible even at `-lv 3`):
`KV stream bootstrap pool ... authoritative host ...`,
`vision arena enabled`, and the `memory_phase:` lines on phase
switches.

## Publishing

```sh
# Build and push in one step
docker buildx build \
  --platform linux/amd64 \
  -t docker.io/binarybrian/llama-cpp:4060ti \
  --push .

# Or build locally first, then push separately
docker buildx build -t docker.io/binarybrian/llama-cpp:4060ti .
docker push docker.io/binarybrian/llama-cpp:4060ti

# TrueNAS picks up the new image on app restart (pull_policy: always)
# — no delete/reinstall needed, just click "Restart"
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/models/qwen38-27b-unsloth/Qwen3.8-27B-UD-IQ3_XXS.gguf` | GGUF weights (Unsloth Dynamic V3.1, MTP head embedded) |
| `MMPROJ_PATH` | `/models/mmproj-Qwen3.8-27B-Q8_0.gguf` | Vision projector (empty disables vision). Q8_0 rather than BF16: Ada has no native BF16 tensor cores; the Unsloth base model keeps the same vision tower |
| `MMPROJ_OFFLOAD` | `auto` | mmproj GPU offload (`auto` = llama.cpp decides, `on` = force GPU, `off` = keep on CPU) |
| `ALIAS` | `qwable-dau` | Model alias shown in the UI |
| `TRY_MTP` | `1` | Try MTP speculative decoding, fall back if unsupported |
| `MTP_PROBE_SECONDS` | `600` | Startup probe window before fallback |
| `SPEC_DRAFT_N_MAX` | `2` | Max speculative (draft) tokens per MTP step; lower for less per-step VRAM/latency (4 was the original tuned value) |
| `PORT` | `8080` | HTTP listen port |
| `HOST` | `0.0.0.0` | HTTP bind address |
| `CTX_SIZE` | `auto` | Context window (`auto` lets `--fit` decide, or a number) |
| `FIT_TARGET` | `256` | VRAM headroom in MiB for `--fit` (auto-fit mode only) |
| `NGL` | `all` | GPU layers to offload (`all`, `auto`, or a number) |
| `BATCH_SIZE` | unset | Logical batch size (`-b`). Unset keeps the llama-server default (2048) |
| `UBATCH_SIZE` | unset | Physical batch size (`-ub`). Unset keeps the llama-server default (512) |
| `CTK` | `auto` (→ q8_0) | KV cache type for K (auto/f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1) |
| `CTV` | `auto` (→ q8_0) | KV cache type for V (auto/f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1) |
| `CTKD` | `auto` (→ q8_0) | MTP draft KV cache type for K (auto/f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1). Ignored when the KV-stream arena is active — the draft then inherits `CTK` |
| `CTVD` | `auto` (→ q8_0) | MTP draft KV cache type for V (auto/f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1). Ignored when the KV-stream arena is active — the draft then inherits `CTV` |
| `TOOLS` | `all` | Built-in tools to enable (empty to disable; redundant when AGENT=1) |
| `AGENT` | `1` | Enable CORS proxy + all built-in tools (`--agent`). Trusted LANs only. |
| `CORS_ORIGINS` | `*` | CORS origins (`*` for all, or comma-separated URLs). Needed when AGENT=1 for LAN access. |
| `TEMP` | `1.0` | Sampling temperature (0.0 = deterministic, 1.0 = random) |
| `TOP_P` | `0.95` | Nucleus sampling probability |
| `TOP_K` | `20` | Top-k sampling (0 = disabled) |
| `MIN_P` | `0.0` | Min-p sampling (0.0 = disabled) |
| `PRESENCE_PENALTY` | `0.0` | Repeat presence penalty |
| `LOG_VERBOSITY` | `3` | Server log verbosity (`-lv`). b11179 hides library INFO lines (model arch block, `KV buffer size`, "block KV streaming enabled") at the default 3 — use `4`+ to see them in docker logs |
| `KV_STREAM_ARENA_MIB` | `0` | Adaptive KV streaming V2 shared device arena in MiB (streaming KV working set + compute workspaces). Only effective on `--kv-stream` builds (`0` = disabled); on other builds the entrypoint warns and skips the flag, and it is also skipped in `CTX_SIZE=auto` (conflicts with `--fit on`) |

Sampling defaults follow the Qwen3.8-27B recommended thinking-mode settings ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)).

## Tuning for low VRAM (16 GB 4060 Ti)

The 27B UD-IQ3_XXS (~10.2 GiB) + MTP draft + mmproj vision fits a 16 GB card
with ~3 GB free at the production arena. If the container crashes on queries
(exit code 1, no host OOM), reduce VRAM pressure via env vars — no rebuild
needed:

```yaml
environment:
  - CTX_SIZE=16384          # halve context (saves ~2 GB KV cache)
  - CTK=q8_0                # already aggressive
  - CTV=q4_0                # drop V cache further (saves ~1 GB)
  - CTKD=q4_0               # aggressive draft KV (draft is speculative, quality matters less)
  - CTVD=q4_0               # aggressive draft KV
  - TRY_MTP=0               # disable MTP draft entirely (saves ~2-3 GB)
  - MMPROJ_PATH=            # empty: disable vision (saves ~0.9 GB)
  - MMPROJ_OFFLOAD=0        # keep projector on CPU instead of GPU
  - TOOLS=                  # empty: disable built-in tools (saves RAM)
```

MTP draft KV cache defaults to f16 upstream — the `auto` sentinel sets
it to q8_0, saving ~0.75 GB vs f16. For even more VRAM, set CTKD/CTVD
to q4_0 (draft quality matters less since rejected drafts are discarded).

> **On `--kv-stream` builds this block mostly doesn't apply.** KV no
> longer consumes VRAM (the arena budget is fixed; growing `CTX_SIZE`
> grows pinned *host* RAM instead), and two knobs above become invalid:
> `CTKD`/`CTVD` are dropped (attached MTP inherits the target's types), and `MMPROJ_OFFLOAD` must stay `1` on vision+arena
> builds. If the arena itself won't admit at startup, raise
> `KV_STREAM_ARENA_MIB` or shrink `UBATCH_SIZE` — not `CTX_SIZE`.

Stop ffmpeg or other GPU processes first (`nvidia-smi` to check). Each
364 MB of foreign VRAM usage is ~3.5K tokens of context you lose.

## CPU and system RAM

The TrueNAS custom app only reserves the GPU device — there are no CPU or
memory limits by default, so the container may use all host resources.

**CPU** — the entrypoint passes no `-t/--threads`, so llama.cpp uses its
default (auto = all host cores; the startup log shows `n_threads = N`).
With `-ngl all` most math is on the GPU, but prompt processing, sampling,
and host-side work fan out over every core.

**System RAM** — no internal cap; usage is driven by your configuration.
Reference values for the production config (Qwen3.8-27B IQ3_XXS mixed
quant, `CTX_SIZE=131072`, `CTK=q8_0 CTV=q4_0`, `KV_STREAM_ARENA_MIB=2048`,
`MMPROJ_OFFLOAD=1`):

| Consumer | Host RAM |
| --- | --- |
| Pinned KV cache (full ctx) | `CTX × kv_layers × n_blocks × (block_K + block_V)`, `n_blocks = n_head_kv × head_dim / 32` (ggml "_0" quants use 32-element blocks: q8_0 = 34 B, q5_0 = 22 B, q4_0 = 18 B). Qwen3.8-27B: **16 KV layers** (full-attention, every 4th) × 4 KV heads × 256 dim = 32 blocks/tensor → 1088 B/token/layer q8_0, 704 B q5_0, 576 B q4_0 → **3328 MiB** (q8_0 K / q4_0 V — current production, matches the `KV buffer size` log line), 3584 MiB (q8_0/q5_0), 4352 MiB (q8_0 K/V) at `CTX_SIZE=131072` |
| mmproj Q8_0 (`--mmproj-offload`, arena-managed on `--kv-stream` builds) | negligible — weights are borrowed into arena (GPU) regions only during image batches |
| Model GGUF via mmap (page cache, reclaimable) | ~10.2 GB |
| Server / threadpool / misc | ~1 GB |
| **Resident total (q8_0 K / q4_0 V + MTP host history, mmproj on GPU, excl. page cache)** | **~4.5–5 GB** |

- Qwen3.8-27B is a **hybrid** model: 3 of every 4 layers are linear
  attention (gated delta net) with a fixed-size state that does *not*
  scale with ctx — with `-ngl all` that state (~0.6 GiB) lives on the
  GPU. Only the 16 full-attention layers carry a per-token KV cache,
  which is what `--kv-stream` pins in host RAM. The MTP module (blk.64)
  keeps its own 1-layer auxiliary KV cache (q8_0/q4_0, inherited from
  the target) in the draft context — 208 MiB, on the GPU.
- The KV-stream V2 arena (`KV_STREAM_ARENA_MIB`) is a bounded GPU budget
  shared between the streaming KV working set and compute workspaces; it
  adds no host RAM beyond bookkeeping. What *does* scale with ctx on the
  host is the pinned KV cache — it grows linearly with `CTX_SIZE` and with
  the K/V cache types (with streaming on, that pinned history is
  authoritative and GPU pages stream from it).
- The startup log prints the exact sizes (`llama_kv_cache: size = ... MiB
  ( N cells, N layers ...), K (...): ... MiB, V (...): ... MiB`) — but
  b11179 classifies library INFO lines as TRACE, so they are hidden at
  the default `-lv 3`. Set `LOG_VERBOSITY=4` (or more) in the app env to
  see them in docker logs.
- Pinned KV is not reclaimable, so it is what determines the host-RAM
  floor for a given context size.

**Optional: limiting resources** — the custom-app YAML accepts standard
k8s resource limits (memory limit → pod OOM-killed; CPU limit →
throttled). Add under `deploy.resources` if you ever want caps:

```yaml
    deploy:
      resources:
        limits:
          cpus: "8"
          memory: 32Gi
        reservations:
          cpus: "2"
          memory: 16Gi
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Reference: `memory: 32Gi` is comfortable for the production config above;
24Gi is tight; 16Gi will not survive 131072 ctx with q8_0/q8_0 KV.

## Benchmarking (mtp-bench.sh)

`mtp-bench.sh` measures prompt/decode speed through `/completion` and is
the standard tool for A/B-ing server configs (notably MTP on vs off):

```sh
./mtp-bench.sh [base_url] [n_predict] [runs]
# defaults: http://localhost:8484  1000  3
# e.g. against the TrueNAS app:
./mtp-bench.sh http://192.168.2.1:30084 1000 3
```

Per measured run it prints: tokens generated, decode and prompt rates
(tokens/s), draft acceptance (MTP only; `n/a` without speculative
decoding), stop reason, and context retained by the server's slot cache.
A median decode rate across runs is printed at the end.

Design notes (why the numbers are clean):

- One unmeasured warmup run absorbs one-time slot-init overhead.
- `cache_prompt: false` is sent per request — otherwise the server's slot
  cache (similarity 0.10 default) carries context between runs.
- Prompts are rotated and uniquely tagged so no two runs share content.
- Runs that stop early on EOS are flagged; the median smooths them out.

Reference result (Qwen3.8-27B IQ3_S, 4060 Ti, 131072 ctx, code prompt):
MTP on (n-max 2, ~86% acceptance) ≈ 37 t/s decode vs MTP off ≈ 22 t/s —
a ~1.7× decode win, offset by a ~1.6× prefill slowdown (the MTP module
runs on every forward pass). Keep MTP on for decode-dominated interactive
use.

## Endpoints

- `GET /` — embedded Web UI
- `GET /health` — health check
- `POST /v1/chat/completions` — OpenAI-compatible API
- `POST /completion` — native llama.cpp completion API
- `GET /metrics` — Prometheus metrics
- `GET /v1/models` — model listing