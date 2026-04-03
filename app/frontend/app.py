"""
Streamlit frontend for Covered Call Strategy Classifier.
Connects to the FastAPI inference server via REST API.
"""

import io
import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

CLASS_COLORS = {
    "ATM_30"      : "#2196F3",
    "ATM_60"      : "#1565C0",
    "ATM_90"      : "#0D47A1",
    "OTM5_30"     : "#4CAF50",
    "OTM5_60_90"  : "#2E7D32",
    "OTM10_30"    : "#FF9800",
    "OTM10_60_90" : "#E65100",
}

CLASS_DESCRIPTIONS = {
    "ATM_30"      : "At-the-money, 30 days to expiration",
    "ATM_60"      : "At-the-money, 60 days to expiration",
    "ATM_90"      : "At-the-money, 90 days to expiration",
    "OTM5_30"     : "5% out-of-the-money, 30 DTE",
    "OTM5_60_90"  : "5% out-of-the-money, 60–90 DTE",
    "OTM10_30"    : "10% out-of-the-money, 30 DTE",
    "OTM10_60_90" : "10% out-of-the-money, 60–90 DTE",
}

# ── Page setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Covered Call Strategy Classifier",
    page_icon="📈",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_model_info():
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


def call_predict_csv(file_bytes, filename):
    r = requests.post(
        f"{API_URL}/predict/csv",
        files={"file": (filename, file_bytes,
                        "text/csv" if filename.endswith(".csv") else "application/octet-stream")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def results_to_df(api_response):
    rows = []
    for r in api_response["results"]:
        row = {
            "index"          : r["index"],
            "predicted_class": r["predicted_class"],
            "confidence"     : r["confidence"],
        }
        row.update({f"prob_{k}": v for k, v in r["probabilities"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_url_input = st.text_input("API URL", value=API_URL)
    if api_url_input != API_URL:
        API_URL = api_url_input

    # Health check
    healthy, health_data = check_api_health()
    if healthy:
        st.success("API Online")
        st.caption(f"Device: {health_data.get('device','?')} | "
                   f"seq_len: {health_data.get('seq_len','?')} | "
                   f"features: {health_data.get('n_features','?')}")
    else:
        st.error("API Offline — start the API server first")

    # Model info
    model_info = fetch_model_info()
    if model_info:
        st.markdown("---")
        st.markdown("**Model Info**")
        st.caption(f"Classes: {len(model_info['target_classes'])}")
        st.caption(f"Sequence length: {model_info['seq_len']} days")
        st.caption(f"Features: {model_info['n_features']}")
        with st.expander("Best Hyperparameters"):
            st.json(model_info["best_params"])
        with st.expander("Class Thresholds"):
            st.json(model_info["thresholds"])

    st.markdown("---")
    st.caption("Covered Call Strategy Classifier v1.0")


# ── Main page ──────────────────────────────────────────────────────────────
st.title("📈 Covered Call Strategy Classifier")
st.markdown(
    "Upload your **test dataset** (CSV or Parquet) with stock features. "
    "The LSTM-CNN model will predict the **optimal covered call strategy bucket** "
    "for each sequence."
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Results Analysis", "ℹ️ Feature Reference"])

# ─────────────────────────────────────────────
# TAB 1 — Upload & Predict
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Upload Test Dataset")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Expected format:**
        - Columns: `symbol`, `date`, + all feature columns (the model needs 35 features)
        - Each symbol must have **at least 50 rows** (the model lookback window)
        - Rows sorted by `symbol` then `date` (the API handles this automatically)
        - File type: `.csv` or `.parquet`
        """)

    with col2:
        st.info("💡 Use the test dataset saved from notebook 09 (`X_test_seq`) "
                "or the raw `test_df` parquet slice.")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Parquet file",
        type=["csv", "parquet"],
        help="Upload the test dataset exported from notebook 09"
    )

    if uploaded_file:
        file_bytes = uploaded_file.read()

        # Preview
        try:
            if uploaded_file.name.endswith(".parquet"):
                preview_df = pd.read_parquet(io.BytesIO(file_bytes))
            else:
                preview_df = pd.read_csv(io.BytesIO(file_bytes))

            st.markdown(f"**Preview** — {len(preview_df):,} rows × {len(preview_df.columns)} columns")
            st.dataframe(preview_df.head(5), use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Rows", f"{len(preview_df):,}")
            with col_b:
                n_sym = preview_df["symbol"].nunique() if "symbol" in preview_df.columns else "?"
                st.metric("Symbols", n_sym)
            with col_c:
                st.metric("Columns", len(preview_df.columns))

        except Exception as e:
            st.warning(f"Could not preview file: {e}")

        # Predict button
        if not healthy:
            st.error("Cannot predict — API is offline.")
        else:
            if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                with st.spinner("Sending to API and running inference..."):
                    try:
                        response = call_predict_csv(file_bytes, uploaded_file.name)
                        st.session_state["results"]  = response
                        st.session_state["results_df"] = results_to_df(response)
                        st.success(
                            f"✅ {response['n_predictions']:,} predictions completed. "
                            f"Go to **Results Analysis** tab."
                        )
                    except requests.HTTPError as e:
                        st.error(f"API error: {e.response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# TAB 2 — Results Analysis
# ─────────────────────────────────────────────
with tab2:
    if "results_df" not in st.session_state:
        st.info("Run a prediction first (Tab 1).")
    else:
        df_res = st.session_state["results_df"]

        st.subheader(f"Results — {len(df_res):,} predictions")

        # ── Summary metrics ──────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", f"{len(df_res):,}")
        with col2:
            top_class = df_res["predicted_class"].value_counts().idxmax()
            st.metric("Most Predicted", top_class)
        with col3:
            avg_conf = df_res["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col4:
            high_conf = (df_res["confidence"] >= 0.6).mean()
            st.metric("High Confidence (≥60%)", f"{high_conf:.1%}")

        st.markdown("---")

        # ── Class distribution bar chart ─────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Predicted Class Distribution**")
            class_counts = df_res["predicted_class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Color"] = class_counts["Class"].map(CLASS_COLORS)

            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(class_counts["Class"], class_counts["Count"],
                           color=class_counts["Color"].fillna("#999"))
            for bar, count in zip(bars, class_counts["Count"]):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                        str(count), va="center", fontsize=9)
            ax.set_xlabel("Count")
            ax.set_title("Predicted Strategy Bucket Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.markdown("**Confidence Distribution per Class**")
            fig, ax = plt.subplots(figsize=(7, 4))
            for cls, color in CLASS_COLORS.items():
                subset = df_res[df_res["predicted_class"] == cls]["confidence"]
                if len(subset) > 0:
                    ax.hist(subset, bins=20, alpha=0.5, label=cls, color=color)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            ax.set_title("Confidence Distribution by Predicted Class")
            ax.legend(fontsize=7, loc="upper left")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        prob_cols = [c for c in df_res.columns if c.startswith("prob_")]
        clean_prob_cols = {c: c.replace("prob_", "") for c in prob_cols}

        # ── Mean probability per class ────────────────────────────────────
        st.markdown("**Mean Predicted Probability per Class (all predictions)**")
        mean_probs = df_res[prob_cols].mean().rename(clean_prob_cols)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = [CLASS_COLORS.get(c, "#999") for c in mean_probs.index]
        bars = ax.bar(mean_probs.index, mean_probs.values, color=colors)
        for bar, val in zip(bars, mean_probs.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean Probability")
        ax.set_title(f"Average Softmax Probability per Class — {len(df_res):,} predictions")
        ax.set_ylim(0, mean_probs.max() * 1.2)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Probability heatmap — all predictions ─────────────────────────
        st.markdown("**Probability Heatmap — All Predictions**")
        n_total = len(df_res)
        if n_total > 500:
            st.caption(f"Showing all {n_total:,} predictions. Use the slider to zoom in.")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                start_idx = st.slider("Start index", 0, max(0, n_total - 50), 0, step=10)
            with col_s2:
                window = st.slider("Window size", 50, min(500, n_total), 100, step=50)
            heatmap_slice = df_res[prob_cols].iloc[start_idx:start_idx + window]
            title_suffix = f"Predictions {start_idx}–{start_idx + len(heatmap_slice) - 1}"
        else:
            heatmap_slice = df_res[prob_cols]
            title_suffix = f"All {n_total} Predictions"

        heatmap_df = heatmap_slice.rename(columns=clean_prob_cols)
        fig_h = max(4, min(len(heatmap_df) * 0.06, 14))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        sns.heatmap(
            heatmap_df.T, cmap="YlOrRd", vmin=0, vmax=1,
            ax=ax, cbar_kws={"label": "Probability"},
            xticklabels=max(1, len(heatmap_df) // 50),
        )
        ax.set_xlabel("Prediction Index")
        ax.set_ylabel("Class")
        ax.set_title(f"Softmax Probabilities — {title_suffix}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Confidence over time (line chart) ────────────────────────────
        st.markdown("**Prediction Confidence Across All Sequences**")
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(df_res.index, df_res["confidence"], color="#1565C0", linewidth=0.8, alpha=0.7)
        ax.axhline(df_res["confidence"].mean(), color="red", linestyle="--",
                   linewidth=1, label=f"Mean {df_res['confidence'].mean():.2%}")
        ax.set_xlabel("Prediction Index")
        ax.set_ylabel("Confidence")
        ax.set_title("Model Confidence per Prediction")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Per-class confidence stats ────────────────────────────────────
        st.markdown("**Confidence Statistics per Predicted Class**")
        conf_stats = (
            df_res.groupby("predicted_class")["confidence"]
            .agg(Count="count", Mean="mean", Median="median",
                 Min="min", Max="max", Std="std")
            .reset_index()
            .rename(columns={"predicted_class": "Class"})
            .sort_values("Count", ascending=False)
        )
        conf_stats[["Mean", "Median", "Min", "Max", "Std"]] = \
            conf_stats[["Mean", "Median", "Min", "Max", "Std"]].applymap(lambda x: f"{x:.3f}")
        st.dataframe(conf_stats, use_container_width=True, hide_index=True)

        # ── Class descriptions ────────────────────────────────────────────
        st.markdown("**What each prediction means:**")
        desc_df = pd.DataFrame([
            {"Strategy Bucket": k, "Description": v,
             "Predicted Count": int((df_res["predicted_class"] == k).sum()),
             "% of Total": f"{(df_res['predicted_class'] == k).mean():.1%}"}
            for k, v in CLASS_DESCRIPTIONS.items()
        ])
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

        # ── Full predictions table ────────────────────────────────────────
        st.markdown("**Full Predictions Table**")
        st.dataframe(df_res, use_container_width=True, height=400)

        # ── Download ──────────────────────────────────────────────────────
        csv_bytes = df_res.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download All Predictions as CSV",
            data=csv_bytes,
            file_name="lstm_cnn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# TAB 3 — Feature Reference
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Feature Reference")
    st.markdown("The model was trained on the top 35 features selected by Random Forest importance.")

    if model_info:
        feat_df = pd.DataFrame({
            "Feature": model_info["feature_cols"],
            "Index"  : range(len(model_info["feature_cols"])),
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not load feature list — API may be offline.")

    st.markdown("---")
    st.markdown("""
    **Input format for /predict/csv endpoint:**
    ```
    symbol,  date,        feature_1, feature_2, ..., feature_35
    AAPL,    2024-01-01,  0.12,      -0.05,     ...,  1.23
    AAPL,    2024-01-02,  0.14,      -0.03,     ...,  1.21
    ...
    ```
    Each symbol needs at least **50 consecutive trading days**.
    """)
