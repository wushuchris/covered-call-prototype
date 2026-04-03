#!/bin/bash
# Bootstrap for GPU spot training instance.
# DLAMI already has conda + PyTorch + CUDA — we just clone the repo and set up.
set -euo pipefail
LOG=/var/log/train_bootstrap.log
exec > >(tee -a $LOG) 2>&1

echo "=== Train Bootstrap: $(date) ==="

# ── 1. Clone repo ────────────────────────────────────────────────────────────
REPO_URL="${repo_url}"
PROJECT_DIR="/home/ec2-user/covered-call-prototype"

if [ ! -d "$${PROJECT_DIR}" ]; then
    sudo -u ec2-user git clone "$${REPO_URL}" "$${PROJECT_DIR}"
fi

# ── 2. Install extra Python deps into base conda env ────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

pip install --quiet mlflow optuna pyarrow fastapi uvicorn

# ── 3. Start MLflow tracking server (background) ────────────────────────────
MLFLOW_DIR="$${PROJECT_DIR}/mlruns"
mkdir -p "$${MLFLOW_DIR}"

nohup mlflow server \
    --backend-store-uri sqlite:///$${MLFLOW_DIR}/mlflow.db \
    --default-artifact-root $${MLFLOW_DIR}/artifacts \
    --host 0.0.0.0 \
    --port ${mlflow_port} \
    > /var/log/mlflow.log 2>&1 &

echo "MLflow server started on port ${mlflow_port}"

# ── 4. Start Jupyter Lab (background, no token for convenience) ─────────────
nohup sudo -u ec2-user bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch
    cd $${PROJECT_DIR}
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' \
        > /var/log/jupyter.log 2>&1
" &

echo ""
echo "=== Bootstrap complete: $(date) ==="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "SSH      : ssh -i ~/.ssh/your-key.pem ec2-user@$${PUBLIC_IP}"
echo "Jupyter  : http://$${PUBLIC_IP}:8888"
echo "MLflow   : http://$${PUBLIC_IP}:${mlflow_port}"
echo ""
echo "To train:"
echo "  1. Open http://$${PUBLIC_IP}:8888"
echo "  2. Navigate to final_notebooks/09_LSTM_CNN_v2_improved.ipynb"
echo "  3. Update MLFLOW_URI to http://$${PUBLIC_IP}:${mlflow_port}"
echo "  4. Run all cells"
