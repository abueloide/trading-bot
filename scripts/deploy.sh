#!/usr/bin/env bash
#
# deploy.sh — first-time VPS provisioning for Trading System v2
#
# Target: Hetzner CX22 (4GB RAM, Ubuntu 22.04+) — adequate for everything
# except local FinBERT, which wants ~2GB. Use CX32 if you enable FinBERT,
# or stick with Alpha Vantage News Sentiment as a lighter alternative.
#
# Usage (as root or with sudo):
#   bash scripts/deploy.sh
#
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/trading-bot}"
SERVICE_USER="${SERVICE_USER:-trader}"
LOG_DIR="${LOG_DIR:-/var/log/trading-bot}"

echo "==> Updating apt + base packages"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip git build-essential libssl-dev libffi-dev

echo "==> Creating service user $SERVICE_USER"
id "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /bin/bash "$SERVICE_USER"

echo "==> Preparing $REPO_DIR"
mkdir -p "$REPO_DIR" "$LOG_DIR"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$REPO_DIR" "$LOG_DIR"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "==> Cloning repo (set REPO_URL env var to override)"
  git clone "${REPO_URL:-https://github.com/abueloide/trading-bot.git}" "$REPO_DIR"
  chown -R "$SERVICE_USER":"$SERVICE_USER" "$REPO_DIR"
fi

echo "==> Setting up Python venv"
sudo -u "$SERVICE_USER" python3 -m venv "$REPO_DIR/venv"
sudo -u "$SERVICE_USER" "$REPO_DIR/venv/bin/pip" install --upgrade pip wheel
sudo -u "$SERVICE_USER" "$REPO_DIR/venv/bin/pip" install -r "$REPO_DIR/requirements.txt"

echo "==> .env"
if [ ! -f "$REPO_DIR/.env" ]; then
  cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
  chown "$SERVICE_USER":"$SERVICE_USER" "$REPO_DIR/.env"
  chmod 600 "$REPO_DIR/.env"
  echo "    Edit $REPO_DIR/.env and add your secrets, then re-run."
fi

echo "==> Installing systemd unit"
install -m 0644 "$REPO_DIR/systemd/trading-bot.service" /etc/systemd/system/trading-bot.service
systemctl daemon-reload

echo "==> Done. Next steps:"
echo "    1) edit $REPO_DIR/.env"
echo "    2) systemctl enable --now trading-bot"
echo "    3) journalctl -u trading-bot -f   # tail logs"
