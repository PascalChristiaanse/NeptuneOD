#!/usr/bin/env bash
#
# setup_aim_server.sh — one-shot setup for the AimStack Tailscale HTTPS server.
#
# Run this once (with sudo where needed) to:
#   1. Install the systemd service (starts at boot, no login required)
#   2. Configure Tailscale Serve (persists across reboots)
#   3. Start everything immediately
#
# Usage:
#   bash setup_aim_server.sh
#
# Prerequisites:
#   - Tailscale installed and logged in  (tailscale status)
#   - Conda env "NeptuneOD" with Aim installed
#   - An existing Aim repo at results/.aim/

set -euo pipefail

SERVICE_NAME="aim-server.service"
SERVICE_DEST="/etc/systemd/system/${SERVICE_NAME}"
AIM_REPO="/home/pascal/Documents/NeptuneOD/results"
AIM_PORT=43801
AIM_BINARY="/home/pascal/miniconda3/envs/NeptuneOD/bin/aim"
TAILSCALE_HOST="pc-pascal.tail290f5b.ts.net"

echo "=== AimStack server setup ==="
echo

# ------------------------------------------------------------------
# 1. Write the systemd service file
# ------------------------------------------------------------------
echo "[1/4] Writing systemd service file…"

cat > /tmp/aim-server.service << 'SERVICEEOF'
[Unit]
Description=AimStack experiment tracking UI & server (NeptuneOD)
Documentation=https://aimstack.readthedocs.io/
After=network-online.target tailscaled.service
Wants=network-online.target tailscaled.service

[Service]
Type=simple
User=pascal
Group=pascal

Environment=AIM_REPO={{AIM_REPO}}
WorkingDirectory={{AIM_REPO}}

ExecStart={{AIM_BINARY}} up \
    --host 127.0.0.1 \
    --port {{AIM_PORT}} \
    --repo {{AIM_REPO}} \
    --log-level info

KillSignal=SIGINT
TimeoutStopSec=30
Restart=on-failure
RestartSec=10
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Substitute placeholders
sed -i "s|{{AIM_REPO}}|${AIM_REPO}|g" /tmp/aim-server.service
sed -i "s|{{AIM_BINARY}}|${AIM_BINARY}|g" /tmp/aim-server.service
sed -i "s|{{AIM_PORT}}|${AIM_PORT}|g" /tmp/aim-server.service

# ------------------------------------------------------------------
# 2. Install & start the systemd service
# ------------------------------------------------------------------
echo "[2/4] Installing and starting the systemd service…"
sudo cp /tmp/aim-server.service "${SERVICE_DEST}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

# Give it a moment to start
sleep 2

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "  ✓ Service is active"
else
    echo "  ✗ Service failed to start — check: sudo systemctl status ${SERVICE_NAME}"
    exit 1
fi

# ------------------------------------------------------------------
# 3. Configure Tailscale Serve (persistent across reboots)
# ------------------------------------------------------------------
echo "[3/4] Configuring Tailscale Serve…"

# Reset any stale config first
sudo tailscale serve reset 2>/dev/null || true

# Serve Aim at the root of the tailnet HTTPS URL
sudo tailscale serve --bg http://127.0.0.1:${AIM_PORT}

echo "  ✓ Tailscale Serve configured"

# ------------------------------------------------------------------
# 4. Final verification
# ------------------------------------------------------------------
echo "[4/4] Verifying setup…"

# Local check
if curl -sf http://127.0.0.1:${AIM_PORT}/ > /dev/null 2>&1; then
    echo "  ✓ Local Aim UI is reachable at http://127.0.0.1:${AIM_PORT}/"
else
    echo "  ✗ Local check failed"
fi

echo
echo "=== Setup complete ==="
echo
echo "  Access the Aim UI from any device on your tailnet:"
echo "    https://${TAILSCALE_HOST}/"
echo
echo "  The server starts automatically at boot (no login needed)."
echo "  To check status:  sudo systemctl status ${SERVICE_NAME}"
echo "  To view logs:     sudo journalctl -u ${SERVICE_NAME} -f"
echo "  To stop:          sudo systemctl stop ${SERVICE_NAME}"
echo