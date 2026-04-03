#!/bin/bash
# Bootstrap script — runs once when the EC2 instance first starts.
# Installs the project environment and downloads data from S3.

set -euo pipefail
LOG=/var/log/ml_setup.log
exec > >(tee -a $LOG) 2>&1

echo "=== ML Training Instance Bootstrap: $(date) ==="

# ── 1. System packages ──────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git htop tmux tree

# ── 2. Clone / create project directory ────────────────────────────────────
PROJECT_DIR="/home/ubuntu/covered-call-prototype"
mkdir -p "$PROJECT_DIR/data/clean"
mkdir -p "$PROJECT_DIR/saved_models"
mkdir -p "$PROJECT_DIR/notebooks-2"
chown -R ubuntu:ubuntu /home/ubuntu/covered-call-prototype

# ── 3. Conda environment ────────────────────────────────────────────────────
# Deep Learning AMI ships with conda at /opt/conda
CONDA=/opt/conda/bin/conda
PIP=/opt/conda/envs/pytorch/bin/pip

# The DLAMI 'pytorch' env already has: torch, numpy, pandas, scikit-learn
# Just install what's missing
echo "Installing extra packages into pytorch conda env..."
sudo -u ubuntu $PIP install --quiet \
    optuna \
    pyarrow \
    shap \
    category-encoders \
    boto3 \
    s3fs \
    jupyterlab

# ── 4. Download data from S3 ────────────────────────────────────────────────
echo "Downloading data from S3..."
sudo -u ubuntu aws s3 cp \
    "s3://${s3_data_bucket}/${s3_data_key}" \
    "$PROJECT_DIR/data/clean/daily_stock_optimal_bucket_modeling_with_fred.parquet"

echo "Data download complete."

# ── 5. Start JupyterLab as a background service ────────────────────────────
JUPYTER=/opt/conda/envs/pytorch/bin/jupyter

# Set a simple password (hash of "coveredcall")
sudo -u ubuntu mkdir -p /home/ubuntu/.jupyter
sudo -u ubuntu bash -c "cat > /home/ubuntu/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$s+YhKOjPgxEOyNQRIvxRnA\$WnVxMvGvqxI9w1dD5U2Z5Q'
c.ServerApp.root_dir = '/home/ubuntu/covered-call-prototype'
c.ServerApp.allow_remote_access = True
EOF"

# Run JupyterLab in a tmux session so it persists
sudo -u ubuntu tmux new-session -d -s jupyter \
    "$JUPYTER lab --config=/home/ubuntu/.jupyter/jupyter_lab_config.py"

# ── 6. GPU check ────────────────────────────────────────────────────────────
echo "=== GPU Info ==="
nvidia-smi

echo ""
echo "=== Setup complete: $(date) ==="
echo "VSCode Remote SSH: connect to this instance, open /home/ubuntu/covered-call-prototype"
echo "JupyterLab       : http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888  (password: coveredcall)"
echo "Log file         : $LOG"
