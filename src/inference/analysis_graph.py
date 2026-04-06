"""
LangGraph DAG for Claude analysis — Graph 4.

Consumes outputs from Graph 1 (inference), Graph 2 (scoring), and
Graph 3 (context) to produce a recommended action report.

Graph topology:
    START → build_prompt → call_claude → format_response → END

Uses the Anthropic Python SDK (Haiku for cost efficiency).
API key loaded from src/.env via dotenv.
Falls back to a structured stub if the API call fails.
"""

import os
from typing import TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.utils import create_logger

logger = create_logger("analysis_graph")

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
INSIGHTS_PATH = Path(__file__).resolve().parent / "capstone_insights.md"

_insights_cache = None


def _get_insights():
    """Load capstone insights (cached)."""
    global _insights_cache
    if _insights_cache is not None:
        return _insights_cache
    if INSIGHTS_PATH.exists():
        _insights_cache = INSIGHTS_PATH.read_text()
    else:
        _insights_cache = ""
    return _insights_cache


class AnalysisState(TypedDict):
    ticker: str
    date: str
    inference: dict     # Graph 1 output
    scoring: dict       # Graph 2 output
    context: dict       # Graph 3 output
    prompt: str
    analysis: dict


# ── Nodes ──────────────────────────────────────────────────────────────

async def build_prompt_node(state: AnalysisState) -> dict:
    """Build the analysis prompt from all three graph outputs + insights."""
    ticker = state["ticker"]
    date = state["date"]
    inf = state["inference"]
    scr = state["scoring"]
    ctx = state["context"]
    insights = _get_insights()

    # Scoring summary
    baseline_ret = scr.get("baseline", {}).get("return", 0)
    argmax_ret = scr.get("argmax", {}).get("return", 0)
    risk_adj_ret = scr.get("risk_adjusted", {}).get("return", 0)
    presets = scr.get("presets", {})

    # Context summary
    price = ctx.get("price", {})
    features = ctx.get("features", {})
    track = ctx.get("track_record", {})
    iv_data = features.get("iv", {})

    # Build inference section based on single vs batch
    is_batch = inf.get("batch", False)
    if is_batch:
        preds = inf.get("predictions", [])
        pred_lines = "\n".join(
            f"  {p['ticker']}: LGBM → {p['lgbm_bucket']} ({p['lgbm_confidence']:.1%}) | "
            f"LSTM → {p['lstm_prediction']} ({p['lstm_confidence']:.1%})"
            for p in preds
        )
        # Batch analytics for Claude
        analytics = inf.get("analytics", {})
        bucket_dist = analytics.get("bucket_distribution", {})
        agreement = analytics.get("agreement_rate", 0)
        lgbm_stats = analytics.get("lgbm_confidence_stats", {})
        lstm_stats = analytics.get("lstm_confidence_stats", {})

        analytics_section = f"""
**Batch Analytics:**
  Bucket distribution: {', '.join(f'{k}: {v}' for k, v in bucket_dist.items())}
  Model agreement rate: {agreement:.0%} (LGBM moneyness matches LSTM)
  LGBM confidence: mean {lgbm_stats.get('mean', 0):.1%}, range [{lgbm_stats.get('min', 0):.1%} – {lgbm_stats.get('max', 0):.1%}]
  LSTM confidence: mean {lstm_stats.get('mean', 0):.1%}, range [{lstm_stats.get('min', 0):.1%} – {lstm_stats.get('max', 0):.1%}]"""

        inference_section = f"""## Inference Results — ALL TICKERS @ {date}

{pred_lines}
{analytics_section}"""
    else:
        lgbm_bucket = inf.get("model_bucket", "?")
        lgbm_conf = inf.get("model_confidence", 0)
        lstm_pred = inf.get("lstm_prediction", "?")
        lstm_conf = inf.get("lstm_confidence", 0)
        lgbm_probs = inf.get("lgbm_probs", {})
        lstm_probs = inf.get("lstm_probs", {})

        prob_section = ""
        if lgbm_probs:
            prob_section += f"\n  LGBM probabilities: {', '.join(f'{k}: {v:.1%}' for k, v in lgbm_probs.items())}"
        if lstm_probs:
            top3 = sorted(lstm_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            prob_section += f"\n  LSTM top-3 probabilities: {', '.join(f'{k}: {v:.1%}' for k, v in top3)}"

        inference_section = f"""## Inference Results — {ticker} @ {date}

**LGBM 3-Class Model** (production, macro F1: 0.59):
  Prediction: {lgbm_bucket} | Confidence: {lgbm_conf:.1%}{prob_section if lgbm_probs else ''}

**LSTM-CNN 7-Class Model** (dashboard, macro F1: 0.11):
  Prediction: {lstm_pred} | Confidence: {lstm_conf:.1%}{prob_section if lstm_probs and not lgbm_probs else ''}"""

    prompt = f"""You are a quantitative analyst at Validex Growth Investors reviewing covered call predictions.

{inference_section}

## Strategy Scoring (this month, all tickers)

| Strategy | Monthly Return |
|----------|---------------|
| Baseline (OTM10) | {baseline_ret:.4%} |
| Argmax | {argmax_ret:.4%} |
| Risk-Adjusted | {risk_adj_ret:.4%} |
| Conservative | {presets.get('conservative', {}).get('return', 0):.4%} |

{"" if is_batch else f"""## Market Context — {ticker}

**Price regime:** {price.get('trend', '?')} trend, {price.get('vol_regime', '?')} volatility
  - Current: ${price.get('current_price', '?')} | 20d vol: {price.get('vol_20d', 0):.1%} | 60d vol: {price.get('vol_60d', 0):.1%}
  - 60d return: {price.get('period_return', 0):.1%} | Drawdown from peak: {price.get('drawdown_from_peak', 0):.1%}

**IV features:** IV rank: {iv_data.get('iv_rank', '?')} | IV mean: {iv_data.get('iv_mean', '?')} | Term structure: {iv_data.get('iv_term_structure', '?')}

**Model track record for {ticker}:**
  Overall accuracy: {track.get('overall_accuracy', '?')} | Recent 12m: {track.get('recent_12m_accuracy', '?')}
  Avg confidence when correct: {track.get('avg_confidence_when_correct', '?')} | when wrong: {track.get('avg_confidence_when_incorrect', '?')}
"""}
## Key Project Insights (from capstone research)

- Distribution shift is the #1 limitation: OTM10_60_90 went from 1.25% to 53.15% of test data
- LGBM is 5x more reliable than LSTM-CNN (0.59 vs 0.11 F1)
- High confidence does NOT guarantee accuracy (overfitting in all models)
- OTM10 baseline has historically been hard to beat
- Macroeconomic features don't help; technical + IV features matter most
- Model is a decision-support tool, NOT a trading signal

## Your Task

Write a report with two clearly labeled sections:

### OVERVIEW
{"Summarize the portfolio-level view for all 10 tickers" if is_batch else f"Distill the market context for {ticker}"} into 2-3 sentences a portfolio manager can scan in 10 seconds.
Cover: {"model agreement/disagreement patterns across the universe, which tickers stand out, and how strategy scoring compares to baseline this month." if is_batch else "current regime (trend + volatility), IV positioning, how the models read this environment, and whether the strategy scoring supports or contradicts the model picks this month."}
Use numbers, not vague language.

### RECOMMENDED ACTION

{"Present the portfolio recommendation using this exact structure:" if is_batch else "Present the recommendation using this exact structure:"}

{"**Confidence Level:** HIGH / MEDIUM / LOW" if not is_batch else "**Overall Confidence:** HIGH / MEDIUM / LOW"}

{"""**Recommended Positions:**

| Ticker | Bucket | LGBM Conf | LSTM Conf | Agree? | Action | Note |
|--------|--------|-----------|-----------|--------|--------|------|
| (fill for each ticker) | | | | Y/N | Trade / Default to OTM10 / Skip | (brief reason) |

**Capital Allocation:**

| Bucket | % Allocation | Tickers |
|--------|-------------|---------|
| (fill per bucket) | | |""" if is_batch else """**Trade:** (specific bucket, e.g. "Sell OTM5 short-dated calls")"""}

### RATIONALE
Write 2-3 sentences: which model(s) support the recommendation, how it compares to baseline, and the key risk.

### CRITICAL RISK
Write 2-3 sentences: the most important limitation for {"this portfolio" if is_batch else "this prediction"}, referencing specific data points from the inputs above.

YOU MUST INCLUDE ALL FOUR SECTIONS: OVERVIEW, RECOMMENDED ACTION, RATIONALE, CRITICAL RISK.
Do not skip any section. Each section must have its ### header.

FORMATTING RULES:
- OVERVIEW, RATIONALE, and CRITICAL RISK must be prose paragraphs
- RECOMMENDED ACTION must use markdown tables for ticker decisions and allocations — never bullet points or inline text for per-stock data
- Be direct, quantitative, and honest about uncertainty
- This is for institutional money managers"""

    return {"prompt": prompt}


async def call_claude_node(state: AnalysisState) -> dict:
    """Call Claude API with the analysis prompt."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY — returning stub")
            return {"analysis": _stub_analysis(state)}

        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": state["prompt"]}],
        )
        text = response.content[0].text
        logger.info(f"Claude analysis complete for {state['ticker']}")

        return {"analysis": {"analysis": text, "model": "claude-haiku-4-5", "source": "api"}}

    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return {"analysis": {"analysis": f"Claude analysis unavailable: {e}", "source": "error"}}


def _stub_analysis(state: AnalysisState) -> dict:
    """Fallback when API key is not available."""
    inf = state.get("inference", {})
    return {
        "analysis": (
            f"[STUB] Analysis for {state['ticker']} @ {state['date']}. "
            f"LGBM predicts {inf.get('model_bucket', '?')} ({inf.get('model_confidence', 0):.1%}), "
            f"LSTM predicts {inf.get('lstm_prediction', '?')} ({inf.get('lstm_confidence', 0):.1%}). "
            f"Enable ANTHROPIC_API_KEY in src/.env for real analysis."
        ),
        "source": "stub",
    }


async def format_response_node(state: AnalysisState) -> dict:
    """Pass through — analysis is already formatted."""
    return {}


# ── Graph builder + invoker ────────────────────────────────────────────

def _build_analysis_graph() -> StateGraph:
    """Construct the analysis DAG."""
    g = StateGraph(AnalysisState)
    g.add_node("build_prompt", build_prompt_node)
    g.add_node("call_claude", call_claude_node)
    g.add_node("format_response", format_response_node)

    g.add_edge(START, "build_prompt")
    g.add_edge("build_prompt", "call_claude")
    g.add_edge("call_claude", "format_response")
    g.add_edge("format_response", END)

    return g.compile()


async def invoke_analysis_graph(ticker: str, date: str,
                                inference: dict, scoring: dict,
                                context: dict) -> dict:
    """Compile and invoke the analysis graph.

    Args:
        ticker: Stock symbol.
        date: Date string (YYYY-MM-DD).
        inference: Graph 1 output (model predictions).
        scoring: Graph 2 output (strategy results).
        context: Graph 3 output (market context).

    Returns:
        Dict with 'analysis' text, 'model', and 'source'.
    """
    graph = _build_analysis_graph()
    result = await graph.ainvoke({
        "ticker": ticker,
        "date": date,
        "inference": inference,
        "scoring": scoring,
        "context": context,
        "prompt": "",
        "analysis": {},
    })
    return result["analysis"]
