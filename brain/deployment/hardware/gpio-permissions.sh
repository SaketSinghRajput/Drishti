#!/bin/bash
set -e
usermod -aG gpio,dialout,plugdev "$USER"
echo "Added $USER to gpio,dialout,plugdev groups"
