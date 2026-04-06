# FAST API server, locally hosted on port 8009. similar logic as the FAST HTML server
# we use this space to set up the routes (app.post, gets). and trigger the corresponding backtesting, daily function
# custom logger for this service as well
# activated by the if __main.py__ clause at the bottom, we are going to be running the dev server for now which is spawned by a Popen process by a mainn.py  trigger

# we should start with an inference call method and a backtesting method
# the request is using a specific data structure (shared between both services)
# the app function is in charge of captuirng it, but unpacking is the job of the handler (backtesting/daily)
# async all the way down once again.

from fastapi import FastAPI
import uvicorn
from src.utils import create_logger, ServiceRequest
from src.inference.inference_utils import unpack_request
from src.inference.daily import run_daily_inference, run_batch_inference
from src.inference.chart import build_candlestick_data
from src.inference.model import compute_model_metrics, get_lgbm_experiment_info
from src.inference.backtesting import run_backtest_all
from src.inference.mlflow_reader import load_experiments
from src.inference.scoring_graph import invoke_scoring_graph
from src.inference.context_graph import invoke_context_graph
from src.inference.analysis_graph import invoke_analysis_graph

logger = create_logger("inference")

app = FastAPI(title="USD Trader Inference Service")


@app.post("/inference")
async def inference_endpoint(body: ServiceRequest):
    """Receive an inference request, unpack it, and run daily inference.

    The request body follows the shared ServiceRequest format
    (data + params). FastAPI validates and parses it automatically
    via the Pydantic model. Unpacking is delegated to inference_utils.

    Args:
        body: ServiceRequest parsed from JSON by FastAPI.

    Returns:
        Dict with inference results.
    """
    try:
        req = unpack_request(body)
        ticker = req.data.get("ticker", "")
        date = req.data.get("date", "")
        result = await run_daily_inference(ticker=ticker, date=date)
        return result
    except Exception as e:
        logger.error(f"Error on /inference: {e}")
        return {"error": str(e)}


@app.post("/inference_batch")
async def inference_batch_endpoint(body: ServiceRequest):
    """Receive a batch inference request for all tickers on a given date.

    Args:
        body: ServiceRequest with date in data dict.

    Returns:
        Dict with per-ticker results and summary statistics.
    """
    try:
        req = unpack_request(body)
        date = req.data.get("date", "")
        result = await run_batch_inference(date=date)
        return result
    except Exception as e:
        logger.error(f"Error on /inference_batch: {e}")
        return {"error": str(e)}


@app.get("/chart_data")
async def chart_data_endpoint(ticker: str = "", date: str = ""):
    """Return OHLC candlestick data for a ticker up to a given date.

    Lightweight endpoint — no model inference, just price data.

    Args:
        ticker: Stock symbol.
        date: End date (YYYY-MM-DD).

    Returns:
        Dict with chart_data list for ApexCharts.
    """
    try:
        data = build_candlestick_data(ticker=ticker, date=date)
        return {"ticker": ticker, "date": date, "chart_data": data}
    except Exception as e:
        logger.error(f"Error on /chart_data: {e}")
        return {"error": str(e), "chart_data": []}


@app.get("/model_metrics")
async def model_metrics_endpoint(year: str = "all", sample_type: str = "all"):
    """Compute model performance metrics for both LGBM and LSTM-CNN.

    Args:
        year: Year filter ('all' or e.g. '2020').
        sample_type: 'all', 'train', 'test', or 'validation'.

    Returns:
        Dict with lgbm and lstm sub-dicts containing metrics.
    """
    try:
        from src.inference.lstm_model import compute_model_metrics as lstm_metrics
        lgbm = compute_model_metrics(year=year, sample_type=sample_type)
        lstm = lstm_metrics(year=year, sample_type=sample_type)
        return {"lgbm": lgbm, "lstm": lstm}
    except Exception as e:
        logger.error(f"Error on /model_metrics: {e}")
        return {"error": str(e)}


@app.get("/mlflow_experiments")
async def mlflow_experiments_endpoint():
    """Return deduplicated MLflow experiment runs plus LGBM production model.

    Reads mlruns/ directory directly — no MLflow server needed.
    Appends the LGBM 3-class production model as an additional row.

    Returns:
        Dict with experiment metadata and unique runs.
    """
    try:
        result = load_experiments()
        lgbm = get_lgbm_experiment_info()
        if lgbm and result.get("experiments"):
            result["experiments"][0]["runs"].append(lgbm)
            result["experiments"][0]["unique_runs"] += 1
        return result
    except Exception as e:
        logger.error(f"Error on /mlflow_experiments: {e}")
        return {"error": str(e)}


@app.get("/scoring")
async def scoring_endpoint(ticker: str = "", date: str = ""):
    """Run scoring graph — all 6 strategies for a given date.

    Args:
        ticker: Stock symbol (for context).
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with all strategy results.
    """
    try:
        return await invoke_scoring_graph(ticker=ticker, date=date)
    except Exception as e:
        logger.error(f"Error on /scoring: {e}")
        return {"error": str(e)}


@app.get("/context")
async def context_endpoint(ticker: str = "", date: str = ""):
    """Run context graph — market regime, features, track record.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).

    Returns:
        Dict with price, feature, and track record context.
    """
    try:
        return await invoke_context_graph(ticker=ticker, date=date)
    except Exception as e:
        logger.error(f"Error on /context: {e}")
        return {"error": str(e)}


@app.get("/claude_analysis")
async def claude_analysis_endpoint(ticker: str = "", date: str = "",
                                   batch: str = ""):
    """Run full analysis pipeline — scoring + context + Claude synthesis.

    Single ticker: inference + context for that ticker, scoring for the month.
    Batch (all tickers): inference for all tickers, scoring for the month,
    context omitted (portfolio-level), Claude gets the full picture.

    Args:
        ticker: Stock symbol (ignored if batch).
        date: Date string (YYYY-MM-DD).
        batch: Non-empty string triggers batch mode.

    Returns:
        Dict with 'analysis' text, 'scoring', and 'context'.
    """
    try:
        from src.inference.model import predict_bucket as lgbm_predict
        from src.inference.lstm_model import predict_bucket as lstm_predict

        UNIVERSE = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL",
                     "META", "MSFT", "NVDA", "TSLA", "WMT"]

        # Scoring is always portfolio-level (all tickers for the month)
        scoring = await invoke_scoring_graph(ticker=ticker or "AAPL", date=date)

        if batch:
            # Batch: gather predictions for all tickers
            all_preds = []
            for t in UNIVERSE:
                lgbm = lgbm_predict(t, date)
                lstm = lstm_predict(t, date)
                all_preds.append({
                    "ticker": t,
                    "lgbm_bucket": lgbm.get("model_bucket", "?"),
                    "lgbm_confidence": lgbm.get("model_confidence", 0),
                    "lstm_prediction": lstm.get("predicted_class", "?"),
                    "lstm_confidence": lstm.get("confidence", 0),
                })

            # Compute batch analytics for Claude + human viz
            # Full bucket distribution (moneyness + maturity, e.g. ATM_LONG)
            lgbm_buckets = [p["lgbm_bucket"] for p in all_preds]
            bucket_counts = {}
            for b in lgbm_buckets:
                bucket_counts[b] = bucket_counts.get(b, 0) + 1

            lgbm_confs = [p["lgbm_confidence"] for p in all_preds]
            lstm_confs = [p["lstm_confidence"] for p in all_preds]

            # Agreement: LGBM moneyness matches LSTM moneyness prefix
            n_agree = sum(
                1 for p in all_preds
                if p["lgbm_bucket"].split("_")[0] in p.get("lstm_prediction", "")
            )

            inference = {
                "batch": True,
                "predictions": all_preds,
                "model_bucket": "BATCH",
                "model_confidence": 0,
                "lstm_prediction": "BATCH",
                "lstm_confidence": 0,
                # Batch analytics (Claude + human viz)
                "analytics": {
                    "bucket_distribution": bucket_counts,
                    "agreement_rate": n_agree / len(all_preds) if all_preds else 0,
                    "lgbm_confidence_stats": {
                        "mean": sum(lgbm_confs) / len(lgbm_confs) if lgbm_confs else 0,
                        "min": min(lgbm_confs) if lgbm_confs else 0,
                        "max": max(lgbm_confs) if lgbm_confs else 0,
                    },
                    "lstm_confidence_stats": {
                        "mean": sum(lstm_confs) / len(lstm_confs) if lstm_confs else 0,
                        "min": min(lstm_confs) if lstm_confs else 0,
                        "max": max(lstm_confs) if lstm_confs else 0,
                    },
                },
            }
            # Context for all tickers — summarized for portfolio view
            all_contexts = {}
            for t in UNIVERSE:
                try:
                    all_contexts[t] = await invoke_context_graph(ticker=t, date=date)
                except Exception:
                    all_contexts[t] = {"error": f"Context failed for {t}"}

            # Build portfolio summary from individual contexts
            valid = {t: c for t, c in all_contexts.items() if "error" not in c}
            prices = {t: c.get("price", {}) for t, c in valid.items()}
            tracks = {t: c.get("track_record", {}) for t, c in valid.items()}

            # Aggregate regime signals
            n_bullish = sum(1 for p in prices.values() if p.get("trend") == "bullish")
            n_high_vol = sum(1 for p in prices.values() if p.get("vol_regime") == "high_volatility")
            avg_vol = sum(p.get("vol_20d", 0) for p in prices.values()) / max(len(prices), 1)
            avg_acc = sum(t.get("overall_accuracy", 0) for t in tracks.values()) / max(len(tracks), 1)

            context = {
                "batch": True,
                "n_tickers": len(valid),
                "price": {
                    "trend": f"{n_bullish}/{len(valid)} bullish",
                    "vol_regime": f"{n_high_vol}/{len(valid)} high vol",
                    "vol_20d": avg_vol,
                    "current_price": "portfolio",
                    "period_return": sum(p.get("period_return", 0) for p in prices.values()) / max(len(prices), 1),
                    "drawdown_from_peak": min((p.get("drawdown_from_peak", 0) for p in prices.values()), default=0),
                },
                "features": {"iv": {}},
                "track_record": {
                    "overall_accuracy": avg_acc,
                    "recent_12m_accuracy": sum(t.get("recent_12m_accuracy", 0) for t in tracks.values()) / max(len(tracks), 1),
                },
                "per_ticker": all_contexts,
                "insights_available": True,
            }
        else:
            # Single ticker
            lgbm = lgbm_predict(ticker, date)
            lstm = lstm_predict(ticker, date)
            inference = {
                "ticker": ticker,
                "model_bucket": lgbm.get("model_bucket", "?"),
                "model_confidence": lgbm.get("model_confidence", 0),
                "lstm_prediction": lstm.get("predicted_class", "?"),
                "lstm_confidence": lstm.get("confidence", 0),
                # Probability distributions for Claude context
                "lgbm_probs": {
                    "ATM": lgbm.get("prob_ATM", 0) if "prob_ATM" in lgbm else 0,
                    "OTM5": lgbm.get("prob_OTM5", 0) if "prob_OTM5" in lgbm else 0,
                    "OTM10": lgbm.get("prob_OTM10", 0) if "prob_OTM10" in lgbm else 0,
                },
                "lstm_probs": lstm.get("probabilities", {}),
            }
            context = await invoke_context_graph(ticker=ticker, date=date)

        analysis = await invoke_analysis_graph(
            ticker=ticker if not batch else "ALL",
            date=date,
            inference=inference, scoring=scoring, context=context,
        )

        result = {
            "analysis": analysis.get("analysis", ""),
            "source": analysis.get("source", "unknown"),
            "scoring": scoring,
            "context": context,
        }
        # Pass viz data for human charts (batch only)
        if batch and "analytics" in inference:
            result["analytics"] = inference["analytics"]
            result["predictions"] = inference["predictions"]
        return result
    except Exception as e:
        logger.error(f"Error on /claude_analysis: {e}")
        return {"error": str(e)}


@app.post("/backtest")
async def backtest_endpoint(body: ServiceRequest):
    """Receive a backtest request and run the backtesting pipeline.

    Extracts preset and budget from the ServiceRequest params.
    Scoring, allocation, and per-bucket return lookup all happen here.

    Args:
        body: ServiceRequest parsed from JSON by FastAPI.

    Returns:
        Dict with backtest report (strategy + baseline metrics).
    """
    try:
        req = unpack_request(body)
        year = req.params.get("year", "all")
        budget = float(req.params.get("budget", 100_000))
        result = await run_backtest_all(budget=budget, year=year)
        return result
    except Exception as e:
        logger.error(f"Error on /backtest: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
