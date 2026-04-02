"""
LSTM-CNN model class definitions.
Shared by both the API server and the Streamlit frontend.
"""
import torch
import torch.nn as nn
import numpy as np


class MultiScaleCNN(nn.Module):
    def __init__(self, n_features: int, cnn_out_channels: int, dropout: float):
        super().__init__()
        self.branches = nn.ModuleList([
            self._make_branch(n_features, cnn_out_channels, k, dropout)
            for k in [3, 5, 7]
        ])

    @staticmethod
    def _make_branch(in_ch, out_ch, kernel, dropout):
        pad = kernel // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return torch.cat([b(x).squeeze(-1) for b in self.branches], dim=-1)


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
                 cnn_out_channels=64, lstm_hidden=128,
                 lstm_layers=2, attn_dim=64, dropout=0.2):
        super().__init__()
        self.multiscale_cnn = MultiScaleCNN(n_features, cnn_out_channels, dropout)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(lstm_hidden * 2, attn_dim)
        fusion_dim = cnn_out_channels * 3 + lstm_hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, past_values):
        x_cnn           = self.multiscale_cnn(past_values.transpose(1, 2))
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
        cnn_out_channels = p.get("cnn_out_channels", 64),
        lstm_hidden      = p.get("lstm_hidden", 128),
        lstm_layers      = p.get("lstm_layers", 2),
        attn_dim         = p.get("attn_dim", 64),
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
