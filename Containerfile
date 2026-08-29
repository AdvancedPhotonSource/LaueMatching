# LaueMatching — CPU and CUDA images, built for Podman.
#
# APS uses Podman, not Docker, and the two cannot coexist on a host
# (https://git.aps.anl.gov/groups/bdp-public/-/wikis/Software-Containers).
# This file is plain OCI, so `docker build` would also work, but every command
# below is the Podman one and the file is named the way Podman expects.
#
#   podman build --target cpu  -t laue:cpu  .
#   podman build --target cuda -t laue:cuda .
#
# On an APS host, if a build or pull fails with "no space left on device" while
# /local has room, it is /var/tmp filling up — Podman extracts layers through
# it. Work around it for the shell and file a Vector ticket:
#
#   mkdir -p /local/$USER/tmp && export TMPDIR=/local/$USER/tmp
#
# Image storage itself needs no setup: the site-wide /etc/containers/storage.conf
# already points the graphroot at local (non-NFS) storage under /local.
#
# RUNNING. The orientation database is NOT baked in — it is 6.7 GB and would
# dominate the image. Mount it:
#
#   podman run --rm -v /local/$USER/data:/data:Z laue:cpu \
#       LaueMatchingCPU params.txt /data/100MilOrients.bin hkls.csv img.bin 8
#
# GPUs come through the NVIDIA Container Toolkit's CDI interface, which APS
# deploys as /etc/cdi/nvidia.yaml. That is `--device`, NOT docker's `--gpus`:
#
#   podman run --rm --device nvidia.com/gpu=all -p 61101:61101 \
#       -v /local/$USER/data:/data:Z laue:cuda \
#       LaueMatchingGPUStream ...
#
# The streaming daemon is the reason this image exists: it is a long-lived
# service holding the 6.7 GB database resident, not a script. For a daemon that
# should come back after a reboot, wrap it in a Quadlet .container unit rather
# than a shell loop.
#
# REGISTRY: APS GitLab (git.aps.anl.gov), which reaches the beamline private
# subnets and ALCF. Not Docker Hub.

# ---------------------------------------------------------------------------
# CPU build. No NLopt to fetch — the simplex is vendored (src/nelder_mead.c) —
# so this stage needs no network beyond the base image.
# ---------------------------------------------------------------------------
FROM docker.io/library/ubuntu:24.04 AS build-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt build.sh ./
COPY src/ src/
COPY packages/laue_index/cmake/ packages/laue_index/cmake/
# SKIP_DOWNLOAD=1: build.sh otherwise pulls the 6.7 GB orientation database,
# which must not end up in an image layer.
RUN SKIP_DOWNLOAD=1 ./build.sh && test -f bin/LaueMatchingCPU

# ---------------------------------------------------------------------------
# CPU runtime
# ---------------------------------------------------------------------------
FROM docker.io/library/ubuntu:24.04 AS cpu
LABEL org.opencontainers.image.source=https://github.com/AdvancedPhotonSource/LaueMatching
LABEL org.opencontainers.image.description="LaueMatching CPU indexer + pipeline"
LABEL org.opencontainers.image.licenses=BSD-3-Clause
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cpu /src/bin/LaueMatchingCPU /usr/local/bin/
# The Python side searches $LAUEMATCHING_BIN first, so the binary the image was
# built with is the one it runs. There is no compiler in this stage, so pip
# installs the package Python-only -- which is the documented contract
# (indexer.available() would report False) and is exactly why the env var is set.
ENV LAUEMATCHING_BIN=/usr/local/bin
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY packages/laue_index/ /opt/laue/laue_index/
RUN pip install --no-cache-dir "/opt/laue/laue_index[run]" \
    && python -c "from laue_index import indexer; assert indexer.available(), 'indexer not found'; print('indexer:', indexer.binary_path())"
WORKDIR /work
CMD ["LaueMatchingCPU"]

# ---------------------------------------------------------------------------
# CUDA build.
#
# The architecture default matters here more than anywhere: a container build
# has no GPU to ask about, and the image is meant to move between hosts. The
# build asks `nvcc --list-gpu-arch` and covers every architecture the toolkit
# supports plus PTX for the newest, so one image runs on every card at the
# facility. Building for "the local card" would be meaningless in a container.
# ---------------------------------------------------------------------------
FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04 AS build-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt build.sh ./
COPY src/ src/
COPY packages/laue_index/cmake/ packages/laue_index/cmake/
RUN SKIP_DOWNLOAD=1 ./build.sh gpu \
    && test -f bin/LaueMatchingGPU && test -f bin/LaueMatchingGPUStream

# ---------------------------------------------------------------------------
# CUDA runtime. The image carries the CUDA runtime libraries; the DRIVER comes
# from the host through CDI, which is why this must not be a -devel base.
# ---------------------------------------------------------------------------
FROM docker.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS cuda
LABEL org.opencontainers.image.source=https://github.com/AdvancedPhotonSource/LaueMatching
LABEL org.opencontainers.image.description="LaueMatching CUDA indexer + streaming daemon"
LABEL org.opencontainers.image.licenses=BSD-3-Clause
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cuda /src/bin/LaueMatchingCPU        /usr/local/bin/
COPY --from=build-cuda /src/bin/LaueMatchingGPU        /usr/local/bin/
COPY --from=build-cuda /src/bin/LaueMatchingGPUStream  /usr/local/bin/
ENV LAUEMATCHING_BIN=/usr/local/bin
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY packages/laue_index/ /opt/laue/laue_index/
RUN pip install --no-cache-dir "/opt/laue/laue_index[run]" \
    && python -c "from laue_index import indexer; assert indexer.available('GPU'), 'GPU binary not found'; print('GPU:', indexer.binary_path('GPU'))"
# The streaming daemon's default port.
EXPOSE 61101
WORKDIR /work
CMD ["LaueMatchingGPUStream"]
