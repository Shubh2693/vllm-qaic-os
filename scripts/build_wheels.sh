#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build vllm-qaic wheels for AOT and/or PYT modes.
#
# Thin wrapper around `docker buildx build --target wheel`. Each Dockerfile
# (docker/Dockerfile.aot, docker/Dockerfile.pyt) exposes a `wheel` BuildKit
# target that builds the wheel inside its own base stage and exports it via
# BuildKit's local exporter — no ephemeral container, bind mount, or
# host-uid/gid mapping needed. Wheels land on the host under --outdir
# (default: <repo_root>/dist), same layout as before.
#
# Usage:
#   ./scripts/build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
#                              [--outdir <dir>] [--device-arch v68|v81]
#                              [--base-image <image:tag>] [--nofile <n>]
#                              [--qaic-version <ver>] [--dry-run]

set -euo pipefail

DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_DEVICE_ARCH="v68"
# `uv python install` bytecode-compiles the stdlib with one worker per CPU. On
# many-core hosts that blows through Docker's default 1024-fd soft limit and
# fails with "No file descriptors available (os error 24)". Containers do not
# inherit the daemon's higher limit, so raise it per build.
DEFAULT_NOFILE="65536"

usage() {
  cat << EOM
Usage: build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
                        [--outdir <dir>] [--device-arch v68|v81]
                        [--base-image <image:tag>] [--nofile <n>]
                        [--qaic-version <ver>] [--dry-run]

aot|pyt|both   Wheel(s) to build (default: both).
--pyver        Python version to build with: 3.10, 3.11, or 3.12
               (default: ${DEFAULT_PYTHON_VERSION}).
--outdir       Wheel output directory (default: <repo_root>/dist).
--device-arch  QAIC device arch for PYT kernel builds: v68 (AI100) or v81 (AI200).
               Bypasses setup.py's live-device probe, needed when devices
               aren't accessible at build time (default: ${DEFAULT_DEVICE_ARCH}).
               Ignored for aot mode.
--base-image   Override the QAIC SDK base image (passed through as the
               BASE_IMAGE build-arg; default: each Dockerfile's own ARG
               default).
--nofile       Open-file (nofile) ulimit for the build containers
               (default: ${DEFAULT_NOFILE}). Containers do not inherit the
               daemon's limit and default to 1024, which is too low for
               'uv python install' on many-core hosts. Set to 0 to omit the
               --ulimit flag and use the daemon default.
--qaic-version QAIC SDK version recorded in the wheel's local version label
               (passed through as the VLLM_QAIC_VERSION build-arg; default:
               each Dockerfile's own ARG default). Produces a version of
               <vllm>+pyt<ver> or <vllm>+aot<ver> — e.g. 1.23.0 yields
               0.23.0+pyt1.23.0. This is metadata only: it does NOT change
               which SDK is used (that comes from --base-image).
--dry-run      Print the docker buildx commands without running them.
EOM
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DOCKER_DIR="${REPO_ROOT}/docker"

# Defaults
BUILD_TARGET="both"
PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
OUT_DIR="${REPO_ROOT}/dist"
DEVICE_ARCH="${DEFAULT_DEVICE_ARCH}"
BASE_IMAGE=""
NOFILE="${DEFAULT_NOFILE}"
QAIC_VERSION=""
DRY_RUN="OFF"

if [[ $# -gt 0 && "$1" != --* ]]; then
    BUILD_TARGET="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyver) PYTHON_VERSION="$2"; shift 2 ;;
        --outdir) OUT_DIR="$2"; shift 2 ;;
        --device-arch) DEVICE_ARCH="$2"; shift 2 ;;
        --base-image) BASE_IMAGE="$2"; shift 2 ;;
        --nofile) NOFILE="$2"; shift 2 ;;
        --qaic-version) QAIC_VERSION="$2"; shift 2 ;;
        --dry-run) DRY_RUN="ON"; shift ;;
        -h|--help) usage; exit 1 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${BUILD_TARGET}" != "aot" && "${BUILD_TARGET}" != "pyt" && "${BUILD_TARGET}" != "both" ]]; then
    echo "ERROR: unknown target '${BUILD_TARGET}'. Use aot|pyt|both" >&2
    exit 1
fi

if [[ "${PYTHON_VERSION}" != "3.10" && "${PYTHON_VERSION}" != "3.11" && "${PYTHON_VERSION}" != "3.12" ]]; then
    echo "ERROR: --pyver must be 3.10, 3.11, or 3.12" >&2
    exit 1
fi

if [[ "${DEVICE_ARCH}" != "v68" && "${DEVICE_ARCH}" != "v81" ]]; then
    echo "ERROR: --device-arch must be 'v68' or 'v81'" >&2
    exit 1
fi

if [[ ! "${NOFILE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --nofile must be a non-negative integer" >&2
    exit 1
fi

# The value lands in a PEP 440 local version label (the '+pyt<ver>' suffix),
# which permits only alphanumerics separated by periods. Reject anything else
# here rather than letting it fail deep inside the wheel build.
if [ -n "${QAIC_VERSION}" ] && [[ ! "${QAIC_VERSION}" =~ ^[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*$ ]]; then
    echo "ERROR: --qaic-version must be alphanumeric segments separated by periods" >&2
    echo "       (e.g. 1.22, 1.23.0, 1.23.0.39); got '${QAIC_VERSION}'" >&2
    exit 1
fi

PYVER_TAG="py${PYTHON_VERSION//./}"

run_echo() {
    if [[ "${DRY_RUN}" == "ON" ]]; then
        printf '[DRY-RUN] '; printf '%q ' "$@"; echo
    else
        "$@"
    fi
}

echo "================================================================"
echo "Build Configuration"
echo "----------------------------------------------------------------"
echo "PYTHON_VERSION : ${PYTHON_VERSION}"
echo "MODE           : ${BUILD_TARGET}"
echo "DEVICE_ARCH    : ${DEVICE_ARCH} (pyt only)"
echo "BASE_IMAGE     : ${BASE_IMAGE:-<Dockerfile default>}"
echo "NOFILE         : ${NOFILE}$([ "${NOFILE}" == "0" ] && echo " (--ulimit omitted)")"
echo "QAIC_VERSION   : ${QAIC_VERSION:-<Dockerfile default>}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "================================================================"

BASE_IMAGE_ARGS=()
if [ -n "${BASE_IMAGE}" ]; then
    BASE_IMAGE_ARGS=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

# Only forward VLLM_QAIC_VERSION when set, so each Dockerfile's own ARG default
# stays the single source of truth otherwise.
QAIC_VERSION_ARGS=()
if [ -n "${QAIC_VERSION}" ]; then
    QAIC_VERSION_ARGS=(--build-arg "VLLM_QAIC_VERSION=${QAIC_VERSION}")
fi

# Raise the container's open-file limit unless explicitly disabled with
# --nofile 0. Docker gives containers a 1024 soft limit regardless of the
# daemon's own (much higher) limit, and `uv python install` runs one
# bytecode-compile worker per CPU — enough to exhaust 1024 fds on a
# many-core host and fail the build with os error 24 (EMFILE).
ULIMIT_ARGS=()
if [ "${NOFILE}" != "0" ]; then
    ULIMIT_ARGS=(--ulimit "nofile=${NOFILE}:${NOFILE}")
fi

build_aot_wheel() {
    echo ""
    echo "=== Building AOT wheel (pure Python, py3-none-any — pyver-independent) ==="
    run_echo docker buildx build --target wheel -f "${DOCKER_DIR}/Dockerfile.aot" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        "${BASE_IMAGE_ARGS[@]}" \
        "${QAIC_VERSION_ARGS[@]}" \
        "${ULIMIT_ARGS[@]}" \
        --output "type=local,dest=${OUT_DIR}/aot" \
        "${REPO_ROOT}"
}

build_pyt_wheel() {
    echo ""
    echo "=== Building PYT wheel (Hexagon compiled) ==="
    run_echo docker buildx build --target wheel -f "${DOCKER_DIR}/Dockerfile.pyt" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg QAIC_DEVICE_ARCH="${DEVICE_ARCH}" \
        "${BASE_IMAGE_ARGS[@]}" \
        "${QAIC_VERSION_ARGS[@]}" \
        "${ULIMIT_ARGS[@]}" \
        --output "type=local,dest=${OUT_DIR}/pyt/${PYVER_TAG}" \
        "${REPO_ROOT}"
}

case "${BUILD_TARGET}" in
    aot)  build_aot_wheel ;;
    pyt)  build_pyt_wheel ;;
    both) build_aot_wheel; build_pyt_wheel ;;
esac

echo ""
echo "================================================================"
echo "  Wheel build results"
echo "----------------------------------------------------------------"

RUN_STATUS=0

report_wheel() {
    local label="$1"
    local pattern="$2"
    local whl
    # Report the newest match, not the alphabetically first: the output dir is
    # not cleared between runs, so stale wheels from an earlier --qaic-version
    # can otherwise be reported as this run's result.
    whl=$(ls -t ${pattern} 2>/dev/null | head -1)
    if [ -n "${whl}" ]; then
        echo "  ${label}: FOUND (${whl})"
    else
        echo "  ${label}: MISSING (expected ${pattern})"
        RUN_STATUS=1
    fi
}

if [ "${DRY_RUN}" == "ON" ]; then
    echo "  (dry-run: no wheels were actually built)"
elif [ "${BUILD_TARGET}" == "aot" ] || [ "${BUILD_TARGET}" == "both" ]; then
    report_wheel "aot" "${OUT_DIR}/aot/vllm_qaic-*aot*.whl"
fi
if [ "${DRY_RUN}" == "OFF" ] && { [ "${BUILD_TARGET}" == "pyt" ] || [ "${BUILD_TARGET}" == "both" ]; }; then
    report_wheel "pyt" "${OUT_DIR}/pyt/${PYVER_TAG}/vllm_qaic-*pyt*.whl"
fi

echo "================================================================"

exit "${RUN_STATUS}"
