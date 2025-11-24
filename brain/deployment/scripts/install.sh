#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root: sudo $0"
  exit 1
fi

# Default target installation directory
TARGET_DIR=/opt/ev-safety-guardian
# Flags
ENABLE_SERVICES=${ENABLE_SERVICES:-false}

# Simple arg parsing: support --target-dir <dir> and --enable-services
while [ "$#" -gt 0 ]; do
  case "$1" in
    --target-dir)
      shift
      if [ -z "${1:-}" ]; then
        echo "Usage: $0 [--target-dir DIR] [--enable-services]"
        exit 1
      fi
      TARGET_DIR=$1
      shift
      ;;
    --enable-services)
      ENABLE_SERVICES=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--target-dir DIR] [--enable-services]"
      exit 0
      ;;
    *)
      # allow legacy positional TARGET_DIR for backward compatibility
      if [ -d "$1" ] || [[ "$1" != --* ]]; then
        TARGET_DIR=$1
        shift
      else
        echo "Unknown option: $1"
        exit 1
      fi
      ;;
  esac
done

echo "Installing to $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# copy repository files (idempotent)
rsync -a --delete --exclude=.git --exclude=tests . "$TARGET_DIR/"

ARCH=$(uname -m || true)
echo "Host arch: $ARCH"

apt-get update

echo "Installing OS packages"
apt-get install -y python3-venv python3-dev build-essential v4l-utils can-utils curl ca-certificates || true

# Optional: install EdgeTPU runtime if requested (best done via vendor repository)
if [ "${INSTALL_EDGETPU:-false}" = "true" ]; then
  echo "EdgeTPU requested. Add Coral apt repo before running this in production. See docs/deployment.md"
fi

# Create a dedicated system user
if ! id -u evguardian >/dev/null 2>&1; then
  useradd -r -m -d /home/evguardian evguardian || true
fi

cd "$TARGET_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt || true

echo "Installing systemd unit files and config"
mkdir -p /etc/ev-safety-guardian
rsync -a config/ /etc/ev-safety-guardian/ || true
rsync -a deployment/systemd/ /etc/systemd/system/ || true

udevadm control --reload-rules || true
udevadm trigger || true

chmod +x deployment/scripts/*.sh || true

systemctl daemon-reload || true

echo
echo "Install finished."
if [ "${ENABLE_SERVICES}" = "true" ]; then
  echo "Enabling and starting services: hardware-diagnostics (+timer), ev-safety-inference, ev-safety-monitoring, ev-safety-fatigue-api, ev-safety-battery-dashboard"
  # Enable services (add hardware diagnostics)
  systemctl enable hardware-diagnostics.service || true
  systemctl enable hardware-diagnostics.timer || true
  systemctl enable ev-safety-inference || true
  systemctl enable ev-safety-monitoring || true
  systemctl enable ev-safety-fatigue-api || true
  systemctl enable ev-safety-battery-dashboard || true

  # Start hardware diagnostics first (oneshot) to perform an initial check before inference
  systemctl start hardware-diagnostics.service || true
  # Start other services after initial diagnostics; timer will handle periodic runs
  systemctl start ev-safety-inference || true
  systemctl start ev-safety-monitoring || true
  systemctl start ev-safety-fatigue-api || true
  systemctl start ev-safety-battery-dashboard || true
  # Start the timer to schedule periodic re-checks
  systemctl start hardware-diagnostics.timer || true
  # Orchestrator service (main system entrypoint)
  systemctl enable ev-safety-orchestrator.service || true
  # Start orchestrator after other services to let hardware-diagnostics run first
  systemctl start ev-safety-orchestrator.service || true
fi

echo
echo "Edit /etc/ev-safety-guardian/default_config.yaml (or your chosen config) before starting or to tune behaviour."
