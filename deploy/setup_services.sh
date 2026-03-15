#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# USD Trader — Systemd Service Setup Script
# Sets up gunicorn + uvicorn workers for both FastHTML (UI) and FastAPI (inference)
#
# WHAT THIS DOES:
#   1. Installs gunicorn into the project venv
#   2. Creates two systemd unit files in /etc/systemd/system/
#   3. Reloads systemd daemon
#   4. Enables both services (auto-start on boot)
#   5. Starts both services
#
# WHAT YOU NEED BEFORE RUNNING:
#   - nginx reverse proxy already pointing to port 8008 (you said this is done)
#   - the .venv with pip install -e . already run
#   - sudo access
#
# HOW TO RUN:
#   chmod +x deploy/setup_services.sh
#   sudo ./deploy/setup_services.sh
#
# HOW TO MANAGE AFTER:
#   sudo systemctl status covered-call-ui
#   sudo systemctl status covered-call-inference
#   sudo systemctl restart covered-call-ui
#   sudo systemctl restart covered-call-inference
#   sudo journalctl -u covered-call-ui -f          # follow logs
#   sudo journalctl -u covered-call-inference -f
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR="/home/alberto/Documents/usd/covered_caller/covered-call-prototype"
VENV_DIR="${PROJECT_DIR}/.venv"
GUNICORN_BIN="${VENV_DIR}/bin/gunicorn"
SERVICE_USER="alberto"
SERVICE_GROUP="alberto"

# raspberry pi — keep workers low (2 per service = 4 total)
UI_WORKERS=2
INFERENCE_WORKERS=2

UI_PORT=8008
INFERENCE_PORT=8009

# ── Step 1: Install gunicorn ──────────────────────────────────────────────────
echo "[1/5] Installing gunicorn into venv..."
"${VENV_DIR}/bin/pip" install gunicorn --quiet

# ── Step 2: Create UI service unit ────────────────────────────────────────────
echo "[2/5] Creating covered-call-ui.service..."
cat > /etc/systemd/system/covered-call-ui.service << 'HEREDOC'
[Unit]
Description=USD Covered Call UI Service (FastHTML on port 8008)
After=network.target
Wants=covered-call-inference.service

[Service]
Type=simple
User=alberto
Group=alberto
WorkingDirectory=/home/alberto/Documents/usd/covered_caller/covered-call-prototype
Environment="PATH=/home/alberto/Documents/usd/covered_caller/covered-call-prototype/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=/home/alberto/Documents/usd/covered_caller/covered-call-prototype"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/alberto/Documents/usd/covered_caller/covered-call-prototype/.venv/bin/gunicorn --workers=2 --worker-class=uvicorn.workers.UvicornWorker --bind=0.0.0.0:8008 --access-logfile=- --error-logfile=- --log-level=info --timeout=60 --graceful-timeout=30 src.ui.app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
KillSignal=SIGTERM
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
HEREDOC

# ── Step 3: Create inference service unit ─────────────────────────────────────
echo "[3/5] Creating covered-call-inference.service..."
cat > /etc/systemd/system/covered-call-inference.service << 'HEREDOC'
[Unit]
Description=USD Covered Call Inference Service (FastAPI on port 8009)
After=network.target

[Service]
Type=simple
User=alberto
Group=alberto
WorkingDirectory=/home/alberto/Documents/usd/covered_caller/covered-call-prototype
Environment="PATH=/home/alberto/Documents/usd/covered_caller/covered-call-prototype/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=/home/alberto/Documents/usd/covered_caller/covered-call-prototype"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/alberto/Documents/usd/covered_caller/covered-call-prototype/.venv/bin/gunicorn --workers=2 --worker-class=uvicorn.workers.UvicornWorker --bind=127.0.0.1:8009 --access-logfile=- --error-logfile=- --log-level=info --timeout=60 --graceful-timeout=30 src.inference.app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
KillSignal=SIGTERM
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
HEREDOC

# ── Step 4: Reload systemd ────────────────────────────────────────────────────
echo "[4/5] Reloading systemd daemon..."
systemctl daemon-reload

# ── Step 5: Enable and start ──────────────────────────────────────────────────
echo "[5/5] Enabling and starting services..."
systemctl enable covered-call-ui covered-call-inference
systemctl start covered-call-inference
systemctl start covered-call-ui

echo ""
echo "Done. Both services are running:"
echo "  UI:        http://127.0.0.1:${UI_PORT}  (covered-call-ui.service)"
echo "  Inference: http://127.0.0.1:${INFERENCE_PORT}  (covered-call-inference.service)"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status covered-call-ui"
echo "  sudo systemctl status covered-call-inference"
echo "  sudo journalctl -u covered-call-ui -f"
echo "  sudo journalctl -u covered-call-inference -f"
echo "  sudo systemctl restart covered-call-ui covered-call-inference"
