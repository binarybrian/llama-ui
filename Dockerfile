# syntax=docker/dockerfile:1.7

# =============================================================================
# llama.cpp server + embedded Web UI — RTX 4060 Ti (Ada, sm_89) target
# Built against CUDA 12.8 toolkit to match TrueNAS driver 570.172.08.
# CPU flags target AMD Ryzen 9 3900X (Zen 2): AVX2/BMI2/F16C/FMA3/SSE4.2 only.
# cmake flag set mirrors the Gentoo ebuild sci-misc/llama-cpp-0_pre10235.ebuild.
# =============================================================================

# ---------- Stage 1: UI build (Node) ------------------------------------------
FROM node:22-bookworm-slim AS ui-builder
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ARG LLAMA_TAG=b10235
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp.git /src/llama.cpp
WORKDIR /src/llama.cpp/tools/ui
# Configure npm: offline-friendly, no telemetry, no fund/audit noise
RUN npm config set cache /tmp/.npm \
 && npm config set audit false \
 && npm config set fund false \
 && npm config set update-notifier false \
 && npm config set progress false
# Output dir matches where CMake's ui-assets.cmake looks first (SRC_DIST_DIR)
ENV LLAMA_UI_OUT_DIR=/src/llama.cpp/tools/ui/dist
RUN --mount=type=cache,target=/tmp/.npm npm install --no-audit --no-fund --ignore-scripts \
 && npx svelte-kit sync \
 && npm run build
# Sanity: SvelteKit adapter-static produces index.html + build.json + _app/
RUN test -f dist/index.html && test -f dist/build.json && test -d dist/_app \
 && echo "UI build OK: $(ls dist | tr '\n' ' ')"


# ---------- Stage 2: llama.cpp build (CUDA 12.8 devel) ------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder
ARG LLAMA_TAG=b10235
ARG CMAKE_CUDA_ARCHITECTURES=89-real;89

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# The devel image ships a libcuda.so stub (no SONAME suffix) under
# stubs/; the ggml-cuda shared lib is linked against libcuda.so.1.
# Provide a matching symlink and make the linker find it via rpath-link
# (rpath-link keeps the path out of the runtime binary — correct for
# images where the real libcuda.so.1 comes from the host driver at run).
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1

# Fetch the exact upstream tag (matches ebuild 0_pre10235 -> b10235)
RUN git clone --depth 1 --branch "${LLAMA_TAG}" \
        https://github.com/ggml-org/llama.cpp.git /src/llama.cpp
WORKDIR /src/llama.cpp

# Bring in the prebuilt UI assets from stage 1 so CMake's ui-assets.cmake
# picks them up from SRC_DIST_DIR (priority 1) and skips both npm rebuild and
# the Hugging Face download fallback.
COPY --from=ui-builder /src/llama.cpp/tools/ui/dist ./tools/ui/dist
# Stamp the dist so ui-assets.cmake's npm_build_should_skip() returns TRUE
# (it skips when dist exists AND sources.cmake shows no newer source files).
RUN touch ./tools/ui/dist/.ui-stamp 2>/dev/null || true

# Build-info injection (mirrors ebuild's LLAMA_BUILD_NUMBER / COMMIT)
ENV LLAMA_BUILD_NUMBER=10235
ENV LLAMA_BUILD_COMMIT=b10235

# cmake configure — flag set mirrors the Gentoo ebuild src_configure():
#   - CUDA backend pinned to sm_89 (Ada / RTX 4060 Ti) via build arg
#   - CPU flags target Zen 2 (Ryzen 9 3900X): no AVX-512 family
#   - LLAMA_BUILD_UI=ON + LLAMA_USE_PREBUILT_UI=OFF (we provide our own dist)
#   - GGML_NATIVE=0 so the explicit -DGGML_AVX* flags drive the cpu kernel
#   - GGML_RPC=ON, OpenSSL ON, OpenMP ON; BLAS/Vulkan/OpenCL/SYCL OFF
#   - CUDA opts from the ebuild's `use cuda` block: FA_ALL_QUANTS, GRAPHS,
#     COMPRESSION_MODE=speed, NCCL=OFF
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llama \
        -DCMAKE_SKIP_BUILD_RPATH=ON \
        -DCMAKE_INSTALL_LIBDIR=lib/llama.cpp \
        -DCMAKE_INSTALL_RPATH=/opt/llama/lib/llama.cpp \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=ON \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_NUMBER=10235 \
        -DLLAMA_BUILD_COMMIT=b10235 \
        -DGGML_NATIVE=OFF \
        -DGGML_RPC=ON \
        -DGGML_OPENMP=ON \
        -DGGML_BLAS=OFF \
        -DGGML_OPENCL=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_SYCL=OFF \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_COMPRESSION_MODE=speed \
        -DGGML_CUDA_NCCL=OFF \
        -DGGML_SSE42=ON \
        -DGGML_AVX=ON \
        -DGGML_AVX2=ON \
        -DGGML_BMI2=ON \
        -DGGML_F16C=ON \
        -DGGML_FMA=ON \
        -DGGML_AVX512=OFF \
        -DGGML_AVX512_VBMI=OFF \
        -DGGML_AVX_VNNI=OFF \
        -DGGML_AVX512_VNNI=OFF \
        -DGGML_AVX512_BF16=OFF \
        -DGENTOO_REMOVE_CMAKE_BLAS_HACK=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

RUN cmake --build build --config Release -j"$(nproc)" \
        --target llama-server \
 && test -x build/bin/llama-server \
 && test -d build/bin

# Stage the artifacts into /opt/llama for the runtime stage to copy.
# We avoid `cmake --install` because it tries to install every target the
# project knows about (batched-bench, etc.) that we didn't build.
RUN mkdir -p /opt/llama/bin /opt/llama/lib/llama.cpp \
 && cp -L build/bin/llama-server /opt/llama/bin/ \
 && cp -L build/bin/lib*.so* /opt/llama/lib/llama.cpp/ 2>/dev/null || true \
 && test -x /opt/llama/bin/llama-server \
 && test -d /opt/llama/lib/llama.cpp

# Show what CUDA archs actually got compiled (verification artifact in build log)
RUN cuobjdump --list-elf /opt/llama/bin/llama-server 2>/dev/null | sort -u | head -20 || true


# ---------- Stage 3: runtime --------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libssl3 ca-certificates curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built server binary and its shared libraries
COPY --from=builder /opt/llama/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /opt/llama/lib/llama.cpp/ /usr/local/lib/llama.cpp/

# RPATH/libpath so the runtime linker finds the ggml/llama shared objects
ENV LD_LIBRARY_PATH=/usr/local/lib/llama.cpp

# Entrypoint: MTP-first with automatic fallback to plain decode
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /usr/local/bin/llama-server

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
  CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]