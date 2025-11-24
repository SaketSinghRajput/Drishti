#!/usr/bin/env bash
set -euo pipefail
echo "Cross-compile helper notes for XGBoost for aarch64."

cat <<'EOF'
Example (inside an aarch64 container):
  git clone --recursive https://github.com/dmlc/xgboost.git
  mkdir build && cd build
  cmake .. -DUSE_CUDA=OFF -DUSE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  # produce wheel or libs as needed
EOF

echo "You can use docker buildx to create an aarch64 build environment if needed."
