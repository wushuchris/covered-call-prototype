"""
log_to_mlflow.py
────────────────
Logs the trained LSTM-CNN checkpoint to an MLflow Tracking Server.

Run once after training (or any time you want to register a new version):

    python saved_models/log_to_mlflow.py \
        --model_path saved_models/lstm_cnn_best_model.pth \
        --tracking_uri http://localhost:5000          # or http://<EC2-IP>:5000

After this script runs, the model is:
  • Logged under experiment  "covered-call-lstm-cnn"
  • Registered in the Model Registry as  "CoveredCallLSTMCNN"
  • Accessible by the run_id printed at the end
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import mlflow
import mlflow.pytorch

# ── Allow importing the model class ──────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from model.lstm_cnn import LSTMCNNClassifier


def log_model(model_path: str, tracking_uri: str, register: bool) -> None:
    ckpt = torch.load(model_path, map_location="cpu")

    best_params    = ckpt["best_params"]
    feature_cols   = ckpt["feature_cols"]
    target_classes = ckpt["target_classes"]
    seq_len        = ckpt["seq_len"]
    num_classes    = ckpt["num_classes"]
    thresholds     = ckpt.get("tuned_thresholds", [1.0] * num_classes)

    # ── Rebuild model ────────────────────────────────────────────────────────
    p = best_params
    model = LSTMCNNClassifier(
        n_features       = len(feature_cols),
        seq_len          = seq_len,
        num_classes      = num_classes,
        cnn_out_channels = p.get("cnn_out_channels", 64),
        lstm_hidden      = p.get("lstm_hidden", 128),
        lstm_layers      = p.get("lstm_layers", 2),
        attn_dim         = p.get("attn_dim", 64),
        dropout          = p.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("covered-call-lstm-cnn")

    with mlflow.start_run(run_name="lstm_cnn_best_model") as run:
        # ── Log hyperparameters ──────────────────────────────────────────────
        mlflow.log_params({
            "seq_len"           : seq_len,
            "num_classes"       : num_classes,
            "n_features"        : len(feature_cols),
            "cnn_out_channels"  : p.get("cnn_out_channels"),
            "lstm_hidden"       : p.get("lstm_hidden"),
            "lstm_layers"       : p.get("lstm_layers"),
            "attn_dim"          : p.get("attn_dim"),
            "dropout"           : p.get("dropout"),
            "batch_size"        : p.get("batch_size"),
            "learning_rate"     : p.get("lr"),
            "weight_decay"      : p.get("weight_decay"),
        })

        # ── Log per-class thresholds as metrics ──────────────────────────────
        for cls, thr in zip(target_classes, thresholds):
            mlflow.log_metric(f"threshold_{cls}", float(thr))

        # ── Log model metadata as artifacts ──────────────────────────────────
        import json, tempfile, os
        meta = {
            "feature_cols"   : feature_cols,
            "target_classes" : target_classes,
            "tuned_thresholds": thresholds,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(meta, f, indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="model_meta")
        os.unlink(tmp_path)

        # ── Log the raw .pth checkpoint ───────────────────────────────────────
        mlflow.log_artifact(model_path, artifact_path="checkpoint")

        # ── Log the PyTorch model (mlflow.pytorch flavor) ────────────────────
        # Input example: (1, seq_len, n_features) dummy tensor
        input_example = torch.zeros(1, seq_len, len(feature_cols))
        signature = mlflow.models.infer_signature(
            input_example.numpy(),
            model(input_example).detach().numpy(),
        )
        mlflow.pytorch.log_model(
            pytorch_model   = model,
            artifact_path   = "model",
            signature       = signature,
            registered_model_name = "CoveredCallLSTMCNN" if register else None,
        )

        run_id = run.info.run_id
        print(f"\n✅ Logged successfully!")
        print(f"   Tracking URI  : {tracking_uri}")
        print(f"   Experiment    : covered-call-lstm-cnn")
        print(f"   Run ID        : {run_id}")
        if register:
            print(f"   Model Registry: CoveredCallLSTMCNN (latest version)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log LSTM-CNN checkpoint to MLflow")
    parser.add_argument(
        "--model_path",
        default="saved_models/lstm_cnn_best_model.pth",
        help="Path to the .pth checkpoint (default: saved_models/lstm_cnn_best_model.pth)",
    )
    parser.add_argument(
        "--tracking_uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=True,
        help="Register model in MLflow Model Registry (default: True)",
    )
    args = parser.parse_args()
    log_model(args.model_path, args.tracking_uri, args.register)
