#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 model.tflite"
  exit 2
fi
MODEL="$1"
if ! command -v edgetpu_compiler >/dev/null 2>&1; then
  echo "edgetpu_compiler not found. Install Coral EdgeTPU compiler on host to compile models."
  exit 2
fi
edgetpu_compiler "$MODEL"
echo "Compiled $MODEL (if supported by compiler)."
