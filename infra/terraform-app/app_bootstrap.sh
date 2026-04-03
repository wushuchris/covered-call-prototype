#!/bin/bash
# Bootstrap: installs Docker + Docker Compose, clones repo, builds and starts containers.
set -euo pipefail
LOG=/var/log/app_bootstrap.log
exec > >(tee -a $LOG) 2>&1

echo "=== App Bootstrap: $(date) ==="

# ── 1. Install Docker ───────────────────────────────────────────────────────
dnf update -y -q
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

# ── 2. Install Docker Compose v2 ───────────────────────────────────────────
COMPOSE_VERSION="v2.27.0"
curl -SL "https://github.com/docker/compose/releases/download/$${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
     -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# ── 3. Clone project repository ────────────────────────────────────────────
REPO_URL="${repo_url}"
PROJECT_DIR="/home/ec2-user/covered-call-prototype"

if [ ! -d "$${PROJECT_DIR}" ]; then
    sudo -u ec2-user git clone "$${REPO_URL}" "$${PROJECT_DIR}"
fi

cd "$${PROJECT_DIR}"

# ── 4. Build and start containers ──────────────────────────────────────────
cd app
sudo -u ec2-user docker-compose build --no-cache
sudo -u ec2-user docker-compose up -d

echo ""
echo "=== Bootstrap complete: $(date) ==="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "Streamlit  : http://$${PUBLIC_IP}:8501"
echo "API        : http://$${PUBLIC_IP}:8000"
echo "API Docs   : http://$${PUBLIC_IP}:8000/docs"
echo "MLflow     : http://$${PUBLIC_IP}:5000"
