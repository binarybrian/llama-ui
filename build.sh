#!/usr/bin/env bash
set -euo pipefail

# Build script for llama-cpp Docker image (RTX 4060 Ti / CUDA 12.8 / sm_89)
# Usage: ./build.sh [VERSION] [--local]

LOCAL_ONLY=0
LLAMA_TAG=""
for arg in "$@"; do
  case "$arg" in
    --local) LOCAL_ONLY=1 ;;
    --help|-h)
      echo "Usage: ./build.sh [VERSION] [--local]"
      echo ""
      echo "  VERSION   llama.cpp release tag (e.g. b10290). Defaults to latest."
      echo "  --local   Build locally only (no push to Docker Hub)"
      echo ""
      echo "Environment:"
      echo "  CUDA_ARCH  CUDA architectures (default: 89-real;89 for RTX 4060 Ti)"
      echo ""
      echo "Examples:"
      echo "  ./build.sh                    # Latest release, build + push"
      echo "  ./build.sh b10290              # Specific tag, build + push"
      echo "  ./build.sh --local             # Latest release, local only"
      echo "  ./build.sh b10290 --local      # Specific tag, local only"
      echo "  CUDA_ARCH=86-real ./build.sh   # Override CUDA arch (e.g. RTX 30-series)"
      exit 0
      ;;
    *) LLAMA_TAG="$arg" ;;
  esac
done

if [[ -z "$LLAMA_TAG" ]]; then
  echo "Fetching latest llama.cpp release..."
  LLAMA_TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
    | grep '"tag_name"' | sed 's/.*"tag_name": *"//;s/".*//')
  if [[ -z "$LLAMA_TAG" ]]; then
    echo "ERROR: could not fetch latest release tag from GitHub API" >&2
    exit 1
  fi
  echo "Latest llama.cpp release: $LLAMA_TAG"
fi

CUDA_ARCH="${CUDA_ARCH:-89-real;89}"

if [[ $LOCAL_ONLY -eq 1 ]]; then
  IMAGE="llama-cpp:local"
  PUSH_FLAG=""
  echo "Building locally (no push): tag=$LLAMA_TAG arch=$CUDA_ARCH image=$IMAGE"
else
  IMAGE="docker.io/binarybrian/llama-cpp:4060ti"
  PUSH_FLAG="--push"
  echo "Building + pushing: tag=$LLAMA_TAG arch=$CUDA_ARCH image=$IMAGE"
fi

docker buildx build \
  --platform linux/amd64 \
  --build-arg LLAMA_TAG="$LLAMA_TAG" \
  --build-arg CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -t "$IMAGE" \
  $PUSH_FLAG \
  .