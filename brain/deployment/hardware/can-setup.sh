#!/bin/bash
set -euo pipefail
IFACE=${1:-can0}
BITRATE=${2:-500000}
ip link set "$IFACE" type can bitrate "$BITRATE"
ip link set "$IFACE" up
echo "Bringing up $IFACE at $BITRATE"
