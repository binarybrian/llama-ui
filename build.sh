#!/usr/bin/env bash
set -euo pipefail

# Build script for llama-cpp Docker image (RTX 4060 Ti / CUDA 12.8 / sm_89)
# Usage: ./build.sh [VERSION] [--local] [--kv-stream]

LOCAL_ONLY=0
KV_STREAM=0
LLAMA_TAG=""
for arg in "$@"; do
  case "$arg" in
    --local) LOCAL_ONLY=1 ;;
    --kv-stream) KV_STREAM=1 ;;
    --help|-h)
      echo "Usage: ./build.sh [VERSION] [--local] [--kv-stream]"
      echo ""
      echo "  VERSION    llama.cpp release tag: a semver release (e.g. v0.8.0) or a"
      echo "             rolling build tag (e.g. b11960). Defaults to the latest"
      echo "             release including pre-releases (the bNNNN series)."
      echo "  --local    Build locally only (no push to Docker Hub)"
      echo "  --kv-stream  Apply the adaptive KV streaming V2 (shared device arena)"
      echo "               patch from RaymondHuang210129/llama.cpp-adaptive-kv-streaming,"
      echo "               branch feature/adaptive-kv-stream-v2. The patch file is"
      echo "               chosen by tag: v0.8.0 / b11960 ->"
      echo "               adaptive-kv-stream-v2-v0.8.0.patch, b11559 ->"
      echo "               adaptive-kv-stream-v2-b11559.patch (other tags: error)."
      echo "               Adds --kv-stream-arena-mib (preferred) and legacy"
      echo "               --kv-stream-pool-mib; requires -fa on + --fit off."
      echo ""
      echo "Environment:"
      echo "  CUDA_ARCH  CUDA architectures (default: 89-real;89 for RTX 4060 Ti)"
      echo ""
      echo "Examples:"
      echo "  ./build.sh                              # Latest release, build + push"
      echo "  ./build.sh v0.8.0 --kv-stream --local   # v0.8.0 (b11960) + V2 patch, local"
      echo "  ./build.sh b11559 --kv-stream           # previous base, build + push"
      echo "  CUDA_ARCH=86-real ./build.sh            # Override CUDA arch"
      exit 0
      ;;
    *) LLAMA_TAG="$arg" ;;
  esac
done

if [[ -z "$LLAMA_TAG" ]]; then
  echo "Fetching latest llama.cpp release (incl. pre-releases)..."
  LLAMA_TAG=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" \
    | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//')
  if [[ -z "$LLAMA_TAG" ]]; then
    echo "ERROR: could not fetch latest release tag from GitHub API" >&2
    exit 1
  fi
  echo "Latest llama.cpp release: $LLAMA_TAG"
fi

# llama.cpp reports a numeric build number (build-info.sh does "int LLAMA_BUILD_NUMBER = N;"),
# so semver tags (v0.8.0) have to be mapped back to their bNNNN build. The release asset
# nightly-tag.txt carries exactly that number.
if [[ "$LLAMA_TAG" =~ ^b([0-9]+)$ ]]; then
  LLAMA_BUILD_NUMBER="${BASH_REMATCH[1]}"
else
  echo "Resolving build number for $LLAMA_TAG (nightly-tag.txt)..."
  NIGHTLY=$(curl -sfL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/nightly-tag.txt" || true)
  if [[ ! "$NIGHTLY" =~ ^b([0-9]+)$ ]]; then
    echo "ERROR: could not resolve a bNNNN build number for tag $LLAMA_TAG (got '${NIGHTLY}')" >&2
    exit 1
  fi
  LLAMA_BUILD_NUMBER="${BASH_REMATCH[1]}"
  echo "  $LLAMA_TAG == build b${LLAMA_BUILD_NUMBER}"
fi

CUDA_ARCH="${CUDA_ARCH:-89-real;89}"

KVSTREAM_ARGS=(--build-arg KV_STREAM="$KV_STREAM")
if [[ $KV_STREAM -eq 1 ]]; then
  case "$LLAMA_TAG" in
    v0.8.0|b11960) KV_STREAM_PATCH="adaptive-kv-stream-v2-v0.8.0.patch" ;;
    b11559)        KV_STREAM_PATCH="adaptive-kv-stream-v2-b11559.patch" ;;
    *)
      echo "ERROR: no kv-stream patch for $LLAMA_TAG (available bases: v0.8.0/b11960, b11559 — see patches/)" >&2
      exit 1
      ;;
  esac
  if [[ ! -f "patches/$KV_STREAM_PATCH" ]]; then
    echo "ERROR: patches/$KV_STREAM_PATCH not found" >&2
    exit 1
  fi
  KVSTREAM_ARGS+=(--build-arg KV_STREAM_PATCH="$KV_STREAM_PATCH")
  KVSTREAM_ARGS+=(--build-arg LLAMA_BUILD_SUFFIX="+kvstream")
fi

if [[ $LOCAL_ONLY -eq 1 ]]; then
  IMAGE="llama-cpp:local"
  PUSH_FLAG=""
  echo "Building locally (no push): tag=$LLAMA_TAG build=b$LLAMA_BUILD_NUMBER arch=$CUDA_ARCH kv_stream=$KV_STREAM image=$IMAGE"
else
  IMAGE="docker.io/binarybrian/llama-cpp:4060ti"
  PUSH_FLAG="--push"
  echo "Building + pushing: tag=$LLAMA_TAG build=b$LLAMA_BUILD_NUMBER arch=$CUDA_ARCH kv_stream=$KV_STREAM image=$IMAGE"
fi

docker buildx build \
  --platform linux/amd64 \
  --build-arg LLAMA_TAG="$LLAMA_TAG" \
  --build-arg LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
  --build-arg CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  "${KVSTREAM_ARGS[@]}" \
  -t "$IMAGE" \
  $PUSH_FLAG \
  .
