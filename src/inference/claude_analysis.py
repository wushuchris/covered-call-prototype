"""
Claude analysis node — synthesizes LGBM + LSTM-CNN predictions.

Separate from the LangGraph DAG. Called via its own endpoint after
the main inference returns, so the UI renders model predictions
immediately and the analysis streams in afterward.

Uses the Anthropic Python SDK (claude-haiku for cost efficiency).
API key loaded from src/.env via dotenv.
"""

import os
from pathlib import Path
# from anthropic import AsyncAnthropic
from src.utils import create_logger

logger = create_logger("claude_analysis")

# Load API key from src/.env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# _client = None


def _get_client():
    """Lazy-init the Anthropic async client.

    Loads ANTHROPIC_API_KEY from src/.env on first call.

    Returns:
        AsyncAnthropic client instance.
    """
    # global _client
    # if _client is not None:
    #     return _client
    #
    # from dotenv import load_dotenv
    # load_dotenv(_ENV_PATH)
    #
    # api_key = os.getenv("ANTHROPIC_API_KEY")
    # if not api_key:
    #     raise ValueError("ANTHROPIC_API_KEY not found in src/.env")
    #
    # _client = AsyncAnthropic(api_key=api_key)
    # return _client
    pass


def _build_prompt(lgbm: dict, lstm: dict, ticker: str, date: str) -> str:
    """Build the analysis prompt from both model predictions.

    Args:
        lgbm: LGBM 3-class result dict.
        lstm: LSTM-CNN 7-class result dict.
        ticker: Stock symbol.
        date: Inference date.

    Returns:
        Formatted prompt string.
    """
    return (
        f"You are a quantitative analyst reviewing covered call predictions for {ticker} on {date}.\n\n"
        f"LGBM 3-Class Model:\n"
        f"  Prediction: {lgbm.get('model_bucket', 'N/A')}\n"
        f"  Confidence: {lgbm.get('model_confidence', 0):.1%}\n"
        f"  Sample type: {lgbm.get('sample_type', 'N/A')}\n\n"
        f"LSTM-CNN 7-Class Model:\n"
        f"  Prediction: {lstm.get('predicted_class', 'N/A')}\n"
        f"  Confidence: {lstm.get('confidence', 0):.1%}\n"
        f"  Sample type: {lstm.get('sample_type', 'N/A')}\n\n"
        f"Both models predict which moneyness/maturity bucket yields the best covered call return.\n"
        f"LGBM uses 3 classes (ATM, OTM5, OTM10). LSTM-CNN uses 7 classes (moneyness × maturity).\n\n"
        f"In 2-3 sentences: compare the predictions, note agreement or disagreement, "
        f"and give a concise recommendation on which bucket to trade. "
        f"Flag if confidence is low on either model."
    )


async def analyze_predictions(ticker: str, date: str,
                              lgbm: dict, lstm: dict) -> dict:
    """Call Claude to synthesize both model predictions.

    Args:
        ticker: Stock symbol.
        date: Inference date.
        lgbm: LGBM result dict from the graph.
        lstm: LSTM-CNN result dict from the graph.

    Returns:
        Dict with 'analysis' text or 'error'.
    """
    try:
        prompt = _build_prompt(lgbm, lstm, ticker, date)
        logger.info(f"Claude analysis triggered for {ticker} @ {date}")

        # ── STUB: uncomment below to enable real Claude API call ──
        # client = _get_client()
        # response = await client.messages.create(
        #     model="claude-haiku-4-5-20251001",
        #     max_tokens=256,
        #     messages=[{"role": "user", "content": prompt}],
        # )
        # analysis_text = response.content[0].text

        # ── TEST STUB: remove when API is live ──
        analysis_text = (
            f"[STUB] Claude analysis for {ticker} @ {date}. "
            f"LGBM predicts {lgbm.get('model_bucket', '?')} "
            f"({lgbm.get('model_confidence', 0):.1%}), "
            f"LSTM predicts {lstm.get('predicted_class', '?')} "
            f"({lstm.get('confidence', 0):.1%}). "
            f"Real analysis will replace this when API is enabled."
        )

        logger.info(f"Claude analysis complete for {ticker}")
        return {"analysis": analysis_text}

    except Exception as e:
        logger.error(f"Claude analysis failed for {ticker}: {e}")
        return {"error": f"Claude analysis failed: {e}"}
