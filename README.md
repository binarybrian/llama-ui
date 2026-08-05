# llama-ui

llama.cpp server with an embedded Web UI, packaged as a Docker Custom App for
TrueNAS SCALE (Electric Eel). Built for an **NVIDIA RTX 4060 Ti (16 GB, Ada —
sm_89)** with CPU kernels targeting an **AMD Ryzen 9 3900X (Zen 2)**.

## Why a custom build

The build host's CUDA toolkit (13.3) is newer than the TrueNAS NVIDIA driver
(570.172.08 → max CUDA 12.8) supports, so a 13.3-compiled binary fails at
runtime on the 4060 Ti. This image builds llama.cpp inside a CUDA 12.8 devel
container, pinning `CMAKE_CUDA_ARCHITECTURES=89-real` and the Zen 2-safe
`-DGGML_*` CPU flags. The cmake flag set mirrors the Gentoo ebuild
`sci-misc/llama-cpp-0_pre10235.ebuild` (commit `b10235`).

The Web UI is built from source (vite + SvelteKit) and embedded into the
`llama-server` binary, so a single process serves both the API and the chat
interface on port 8080.

## Served model

`Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-IQ2_M.gguf`
(IQ2_M, ~11.3 GB) with the matching `mmproj-BF16.gguf` for vision. MTP
speculative decoding is attempted first; the entrypoint falls back to plain
decode if the model's MTP layers fail to initialize.

## Quick start (TrueNAS)

1. Apps → Custom Apps → **Install via YAML** → paste `docker-compose.yml`.
2. Point the `volumes` entry at your models dataset (SSD recommended).
3. Install. The Web UI comes up at `http://<nas-ip>:8080`.

## Build locally

```sh
docker buildx build \
  --platform linux/amd64 \
  -t docker.io/binarybrian/llama-ui:4060ti-mtp-iq2m \
  --push .
```

Override the CUDA arch or llama.cpp tag via build args if needed:

```sh
docker buildx build --build-arg CMAKE_CUDA_ARCHITECTURES=86-real \
  --build-arg LLAMA_TAG=b10235 -t llama-ui:local .
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/models/qwen36-dau/...MTP-IQ2_M.gguf` | GGUF weights |
| `MMPROJ_PATH` | `/models/qwen36-dau/mmproj-BF16.gguf` | Vision projector (empty disables vision) |
| `ALIAS` | `qwable-dau` | Model alias shown in the UI |
| `TRY_MTP` | `1` | Try MTP speculative decoding, fall back if unsupported |
| `MTP_PROBE_SECONDS` | `45` | Startup probe window before fallback |
| `PORT` | `8080` | HTTP listen port |
| `HOST` | `0.0.0.0` | HTTP bind address |
| `CTX_SIZE` | `32768` | Context window (tokens). Lower to save VRAM. |
| `NGL` | `all` | GPU layers to offload (`all`, or a number) |
| `CACHE_TYPE_K` | `q8_0` | KV cache type for K (f32/f16/bf16/q8_0/q5_0/q4_0/iq4_nl) |
| `CACHE_TYPE_V` | `q5_0` | KV cache type for V (lower V first to save VRAM) |
| `TOOLS` | `all` | Built-in tools to enable (set empty to disable) |
| `MMPROJ_PATH` | `…/mmproj-BF16.gguf` | Vision projector (empty string disables vision) |

## Tuning for low VRAM (16 GB 4060 Ti)

The 27B IQ2_M (~11.3 GB) + 32K ctx + MTP draft + mmproj vision can exceed
16 GB during inference. If the container crashes on queries (exit code 1,
no host OOM), reduce VRAM pressure via env vars — no rebuild needed:

```yaml
environment:
  - CTX_SIZE=16384          # halve context (saves ~2 GB KV cache)
  - CACHE_TYPE_K=q8_0       # already aggressive
  - CACHE_TYPE_V=q4_0       # drop V cache further (saves ~1 GB)
  - TRY_MTP=0               # disable MTP draft (saves ~1 GB)
  - MMPROJ_PATH=            # empty: disable vision (saves ~0.9 GB)
  - TOOLS=                  # empty: disable built-in tools (saves RAM)
```

Stop ffmpeg or other GPU processes first (`nvidia-smi` to check). Each
364 MB of foreign VRAM usage is ~3.5K tokens of context you lose.

## Endpoints

- `GET /` — embedded Web UI
- `GET /health` — health check
- `POST /v1/chat/completions` — OpenAI-compatible API
- `POST /completion` — native llama.cpp completion API
- `GET /metrics` — Prometheus metrics
- `GET /v1/models` — model listing