#!/usr/bin/env bash
set -euo pipefail
# Minimal helper to prepare an environment for cross-building ONNX Runtime / libs
echo "Preparing QEMU/binfmt and Docker buildx for cross-building (requires docker)."
docker run --rm --privileged tonistiigi/binfmt --install all || true
if ! docker buildx ls | grep -q mybuilder; then
  docker buildx create --use --name mybuilder || true
fi

cat <<'EOF'
Example commands:
  # Build the Pi4 runtime image (uses buildx/qemu on x86 hosts)
  docker buildx build --platform linux/arm64 -f deployment/docker/Dockerfile.pi4 -t ev-safety:pi4 --load .

  # To build ONNX Runtime natively for aarch64, prefer running inside an aarch64 container:
  git clone --recursive https://github.com/microsoft/onnxruntime.git
  docker run --rm -it --platform linux/arm64 -v $PWD/onnxruntime:/work ubuntu:22.04 /bin/bash
  # Inside container: follow ONNXRuntime build docs (cmake flags, toolchains)
EOF

echo "See ONNX Runtime docs for exact cmake/build flags required for your use case."
