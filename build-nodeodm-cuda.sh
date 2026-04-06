#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${1:-nodeodm-cuda:latest}"
# Default to RTX 3090 (8.6). For broader compatibility use "7.5;8.0;8.6;8.9;9.0"
CUDA_ARCH="${2:-8.6}"

echo "Building CUDA-enabled nodeodm image: ${IMAGE_NAME}"
echo "CUDA architectures: ${CUDA_ARCH}"
echo ""

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.nodeodm-cuda" \
    --build-arg CUDA_ARCH_BIN="${CUDA_ARCH}" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo ""
echo "Done! Image: ${IMAGE_NAME}"
echo ""
echo "To use with WebODM, edit docker-compose.nodeodm.gpu.nvidia.yml:"
echo "  image: ${IMAGE_NAME}"
echo ""
echo "Or run standalone:"
echo "  docker run --gpus all -p 3000:3000 ${IMAGE_NAME}"
