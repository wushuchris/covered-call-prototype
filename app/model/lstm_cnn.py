"""
LSTM-CNN model class definitions.
Shared by both the API server and the Streamlit frontend.

Architecture matches the saved checkpoint (single-branch CNN):
  - self.cnn  : single Conv1d branch with kernel_size param
  - fusion_dim: cnn_out_channels + lstm_hidden * 2  (= 128 + 256 = 384)
"""
import torch
import torch.nn as nn
import numpy as np


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, lstm_out):
        weight  = torch.softmax(self.v(torch.tanh(self.W(lstm_out))), dim=1)
        context = (weight * lstm_out).sum(dim=1)
        return context, weight.squeeze(-1)


class LSTMCNNClassifier(nn.Module):
    def __init__(self, n_features, seq_len, num_classes,
                 cnn_out_channels=128, kernel_size=7,
                 lstm_hidden=128, lstm_layers=2,
                 attn_dim=128, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_out_channels, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(cnn_out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(lstm_hidden * 2, attn_dim)
        fusion_dim = cnn_out_channels + lstm_hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, past_values):
        x_cnn           = self.cnn(past_values.transpose(1, 2)).squeeze(-1)
        lstm_out, _     = self.lstm(past_values)
        x_attn, _       = self.attention(lstm_out)
        return self.head(torch.cat([x_cnn, x_attn], dim=-1))


def load_model(model_path: str, device: torch.device):
    """Load checkpoint and return (model, checkpoint_dict)."""
    ckpt  = torch.load(model_path, map_location=device)
    p     = ckpt["best_params"]
    model = LSTMCNNClassifier(
        n_features       = len(ckpt["feature_cols"]),
        seq_len          = ckpt["seq_len"],
        num_classes      = ckpt["num_classes"],
        cnn_out_channels = p.get("cnn_out_channels", 128),
        kernel_size      = p.get("kernel_size", 7),
        lstm_hidden      = p.get("lstm_hidden", 128),
        lstm_layers      = p.get("lstm_layers", 2),
        attn_dim         = p.get("attn_dim", 128),
        dropout          = p.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict(model, X: np.ndarray, thresholds: np.ndarray,
            device: torch.device, batch_size: int = 128):
    """
    Run inference on sequences X (N, seq_len, n_features).
    Returns class indices, class names list, and probability array.
    """
    all_probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_batch = torch.tensor(
                X[start:start + batch_size], dtype=torch.float32
            ).to(device)
            logits = model(x_batch)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    probs_arr = np.vstack(all_probs)

    # Apply tuned per-class thresholds
    adjusted  = probs_arr / np.array(thresholds).reshape(1, -1)
    preds     = adjusted.argmax(axis=1)
    return preds, probs_arr
