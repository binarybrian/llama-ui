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
rolling `bNNNN` series); pass a tag to pin it (e.g. `./build.sh b10729 --local`).

Override the CUDA arch or llama.cpp tag via build args if needed:

```sh
docker buildx build --build-arg CMAKE_CUDA_ARCHITECTURES=86-real \
  --build-arg LLAMA_TAG=b10729 -t llama-cpp:local .
```

### Adaptive KV streaming (ring buffer) variant

```sh
./build.sh --kv-stream --local
```

Applies the adaptive KV streaming patch (derived from
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming),
branch `feature/adaptive-kv-stream`) on top of the standard build. The
patch file is chosen by tag: `b11200` →
`patches/adaptive-kv-stream-b11200.patch`, `b11179` →
`patches/adaptive-kv-stream-b11179.patch`, `b11115` →
`patches/adaptive-kv-stream-b11115.patch`, `b10729` →
`patches/adaptive-kv-stream-b10729.patch`; other tags error out in
`build.sh`, and a mismatched base would fail the `git apply --check` guard
in the Dockerfile. The binary gains `--kv-stream-stage-mib N` (staging
pool in MiB, 0 = off), exposed as the `KV_STREAM_STAGE_MIB` env var. If
that env var is set on a non-`--kv-stream` image, the entrypoint detects
the missing flag and continues without it instead of failing startup.

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
| `KV_STREAM_STAGE_MIB` | `0` | Adaptive KV streaming (ring buffer) staging pool in MiB. Only effective on `--kv-stream` builds (`0` = disabled); on other builds the entrypoint warns and skips the flag |

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
Reference values for the production config (Qwen3.8-27B IQ3_S,
`CTX_SIZE=131072`, `CTK=q8_0 CTV=q8_0`, `KV_STREAM_STAGE_MIB=2048`,
`MMPROJ_OFFLOAD=0`):

| Consumer | Host RAM |
| --- | --- |
| Pinned KV cache (full ctx) | `CTX × kv_layers × n_blocks × (block_K + block_V)`, `n_blocks = n_head_kv × head_dim / 32` (ggml "_0" quants use 32-element blocks: q8_0 = 34 B, q5_0 = 22 B, q4_0 = 18 B). Qwen3.8-27B: **16 KV layers** (full-attention, every 4th) × 4 KV heads × 256 dim = 32 blocks/tensor → 1088 B/token q8_0, 704 B/token q5_0 → **3584 MiB** (q8_0/q5_0) or **4352 MiB** (q8_0 K/V) at `CTX_SIZE=131072` |
| mmproj on CPU (`--no-mmproj-offload`) | ~1–2 GB (= the mmproj file size) |
| Model GGUF via mmap (page cache, reclaimable) | ~11 GB |
| Server / threadpool / misc | ~1 GB |
| **Resident total (q8_0/q5_0, mmproj on GPU, excl. page cache)** | **~4.5–5 GB** |

- Qwen3.8-27B is a **hybrid** model: 3 of every 4 layers are linear
  attention (gated delta net) with a fixed-size state that does *not*
  scale with ctx — with `-ngl all` that state (~0.6 GiB) lives on the
  GPU. Only the 16 full-attention layers carry a per-token KV cache,
  which is what `--kv-stream` pins in host RAM. The MTP module (blk.64)
  keeps its own 1-layer KV cache (q8_0/q8_0) in the draft context — a
  few MiB, on the GPU.
- The KV pool (`KV_STREAM_STAGE_MIB`) lives entirely on the GPU; it adds
  no host RAM beyond bookkeeping. What *does* scale with ctx on the host
  is the pinned KV cache — it grows linearly with `CTX_SIZE` and with the
  K/V cache types.
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