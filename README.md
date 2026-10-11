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
`MODEL_PATH` (and optionally `MMPROJ_PATH` for vision). The default config
serves `Qwen3.6-27B-Fable-Fus-711-...IQ2_M.gguf` (~11.3 GB), which fits
comfortably in 16 GB VRAM with auto-fit context allocation. Swap models by
changing the env vars — no rebuild needed.

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

`./build.sh` defaults to the latest release **including pre-releases** (the
rolling `bNNNN` series); pass a tag to pin it (e.g. `./build.sh b11559 --local`).

Override the CUDA arch or llama.cpp tag via build args if needed:

```sh
docker buildx build --build-arg CMAKE_CUDA_ARCHITECTURES=86-real \
  --build-arg LLAMA_TAG=b11559 -t llama-cpp:local .
```

### Adaptive KV streaming V2 (shared device arena) variant

```sh
./build.sh b11559 --kv-stream --local
```

Applies the adaptive KV streaming **V2** patch (derived from
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming),
branch `feature/adaptive-kv-stream-v2`) on top of the standard build. V2
keeps the complete KV history in pinned host RAM and serves attention from
a bounded, device-local arena of resident pages plus one shared ring; the
same arena is phase-shared with compute workspaces (and the vision
projector). The patch file is chosen by tag: `b11559` →
`patches/adaptive-kv-stream-v2-b11559.patch`; other tags error out in
`build.sh`, and a mismatched base would fail the `git apply --check` guard
in the Dockerfile. The binary gains `--kv-stream-arena-mib N` (shared
arena MiB; requires `-fa on` + `--fit off`, auto-derived MTP KV layer
count) and the legacy `--kv-stream-pool-mib N`, exposed as the
`KV_STREAM_ARENA_MIB` env var. If that env var is set on a
non-`--kv-stream` image, the entrypoint detects the missing flag and
continues without it instead of failing startup; it also skips the arena
in `CTX_SIZE=auto` mode, which runs `--fit on`. Note V2 renamed the old
`--kv-stream-stage-mib` flag (V1 patches, removed) to
`--kv-stream-pool-mib`.

### KV-stream V2 hard rules + 4060 Ti qualification (b11559)

Hard startup rules enforced by the patch — violations exit at model-load
time, before any weights load:

- `CTX_SIZE` must be a number: `auto` runs `--fit on`, and the entrypoint
  skips the arena in that mode.
- **Vision + arena is hard-qualified to `CTK=q8_0 CTV=q4_0`** — with both
  `MMPROJ_PATH` and the arena set, any other cache types are rejected
  (`vision arena is currently qualified with -fa on, Q8_0 K and Q4_0 V`).
  Text-only arena (no mmproj) is not gated.
- Attached MTP inherits the target's KV types: set `CTKD`/`CTVD` equal to
  `CTK`/`CTV` (the `auto` default resolves to q8_0, which mismatches a
  q4_0 V and gets rejected). Draft lengths 1–3; keep
  `SPEC_DRAFT_N_MAX` in that range.
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
| `MODEL_PATH` | `/models/qwen36-dau/...IQ2_M.gguf` | GGUF weights |
| `MMPROJ_PATH` | `/models/qwen36-dau/mmproj-BF16.gguf` | Vision projector (empty disables vision) |
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
| `CTKD` | `auto` (→ q8_0) | MTP draft KV cache type for K (auto/f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1) |
| `CTVD` | `auto` (→ q8_0) | MTP draft KV cache type for V (auto/f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1) |
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

Sampling defaults follow the Qwen3.8-27B recommended thinking-mode settings ([byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF)).

## Tuning for low VRAM (16 GB 4060 Ti)

The 27B IQ2_M (~11.3 GB) + MTP draft + mmproj vision can exceed 16 GB
during inference. If the container crashes on queries (exit code 1,
no host OOM), reduce VRAM pressure via env vars — no rebuild needed:

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
> `CTKD`/`CTVD` must match `CTK`/`CTV` (attached MTP inherits the
> target's types), and `MMPROJ_OFFLOAD` must stay `1` on vision+arena
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
| Model GGUF via mmap (page cache, reclaimable) | ~11 GB |
| Server / threadpool / misc | ~1 GB |
| **Resident total (q8_0 K / q4_0 V + MTP host history, mmproj on GPU, excl. page cache)** | **~4.5–5 GB** |

- Qwen3.8-27B is a **hybrid** model: 3 of every 4 layers are linear
  attention (gated delta net) with a fixed-size state that does *not*
  scale with ctx — with `-ngl all` that state (~0.6 GiB) lives on the
  GPU. Only the 16 full-attention layers carry a per-token KV cache,
  which is what `--kv-stream` pins in host RAM. The MTP module (blk.64)
  keeps its own 1-layer KV cache (q8_0/q8_0) in the draft context — a
  few MiB, on the GPU.
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