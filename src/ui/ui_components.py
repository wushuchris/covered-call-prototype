# the way we will handle components is we will be setting up first smaller components as functions (e.g. a Forms with a text box and a submit button)
# these components are then spawned inside card/bigger div components. these larger components are functionality based and a single card/div covers an
# overarching function-theme (e.g. trading system backtesting results visualization, daily inference computation)
# these overarching modules are rendered by a page-renderer. having this three-tiered modularity allows for hx-swaps to take place efficiently,
# changing divs/components as required by a given trigger

# given the fact certain components will be generated differently based on user inputs (e.g. the daily inference section dashboard will render a different
# stock based on dropdown choice, but it could also render a completely different type of graph based on future functionalities). hence the reason we
# render components inside functions (we could always add a new function input, and an if else statement that renders a different component based on the conditional)
# and function include a try except finally clause that makes sure we always keep the page structure even if a specialized component failed to render correctly
# there would then exist a generic backup that takes its place (the error is logged, and we do show a time-limited toast to the user letting it know there was an
# error rendering that specific component)

from datetime import date

from fasthtml.common import *
import fasthtml.common as fh
from monsterui.all import *

TICKERS = ["AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "WMT"]

# USD brand colors as inline styles (used where MonsterUI theme doesn't reach)
_FOUNDERS = "#003b70"
_IMMACULATA = "#0074c8"
_TORERO = "#75bee9"


def _fallback(component_name: str):
    """Generic backup Div when a component fails to render.

    Shows a time-limited toast notification and a persistent fallback div
    so the page structure is preserved even on render failure.

    Args:
        component_name: Name of the component that failed.

    Returns:
        Div with an error toast and a fallback message.
    """
    return Div(
        Toast(f"Error rendering {component_name}.", cls=ToastHT.end, alert_cls=AlertT.warning, dur=3.0),
        Card(P(f"Could not load {component_name}. Please try again.", cls=TextPresets.muted_sm)),
    )


def _tip(text: str):
    """Render a small tooltip bubble (?) with hover explanation.

    Uses UIKit's uk-tooltip -no JS, just a data attribute.
    Styled as a small circular icon inline with text.

    Args:
        text: Plain-English explanation shown on hover.

    Returns:
        Span element with tooltip.
    """
    return Span(
        "?",
        uk_tooltip=f"title: {text}",
        style=f"display:inline-flex; align-items:center; justify-content:center; "
              f"width:16px; height:16px; border-radius:50%; "
              f"background:{_IMMACULATA}18; color:{_IMMACULATA}; font-size:0.65rem; "
              f"cursor:help; margin-left:0.35rem; font-weight:700; "
              f"vertical-align:middle; line-height:1;",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCHER SCREEN COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

# launcher should include a card at the center with the usd logo as a .svg, an enter button that could be either an ikon or a button,
# and a brief description, in our case we could start with a placeholder phrase. usd motto or something.

# on buton click should send to the trading screen, but we could setup up a loader intermediary (at first we manually force a 2 seconde wait)
# and then the loader component redirects to the actual screen. routes and requests are handled on the app.py
def launcher_screen():
    """Full-page launcher: centered USD logo, tagline, and enter button.

    Returns:
        Div containing the launcher layout.
    """
    try:
        return Div(
            DivCentered(
                Img(src="/static/usd_logo.png", alt="USD Logo",
                    style="max-width:220px; margin-bottom:1.5rem;"),
                H2("USD Strategy Advisor", style=f"color:{_FOUNDERS};"),
                P("in coordination with Validex Growth Investors",
                  style=f"color:{_IMMACULATA}; font-style:italic; margin-top:-0.25rem;"),
                P("Machine Learning-Driven Covered Call Optimization",
                  cls=TextPresets.muted_sm),
                Button("Enter",
                       hx_get="/trading", hx_target="#main", hx_swap="innerHTML",
                       cls="mt-4",
                       style=f"background-color:{_IMMACULATA}; color:white;"),
                cls="flex flex-col items-center",
            ),
            id="main",
            cls="flex items-center justify-center",
            style=f"min-height:100vh; background:linear-gradient(180deg, white 60%, {_TORERO}22 100%);",
        )
    except Exception:
        return _fallback("launcher")


# ═══════════════════════════════════════════════════════════════════════════════
# Trading screen components
# ═══════════════════════════════════════════════════════════════════════════════

# trading screen should be composed by a navbar (scrollspy, anchored to the top with all overarching sections mentioned),
# a first section for daily inference, and a second section for backtesting dashboards, and a footer that redirects to our "docs" (which is another screen)
# entirely. docs don't describe the code, but rather the system as a whole (data processing, feature engineering, model architectures, etc... more on docs later)

# each section should have a title/header a brief description, and a divider line
def trading_screen():
    """Assemble the full trading screen: navbar + daily inference + backtesting + footer.

    Returns:
        Div containing all trading-screen sections.
    """
    try:
        return Div(
            _navbar(),
            Div(
                _daily_inference_section(),
                DividerLine(),
                _docs_footer(),
                cls="space-y-4 px-4 py-4 w-full max-w-full",
            ),
        )
    except Exception:
        return _fallback("trading screen")


def _navbar():
    """Top navigation bar with scrollspy anchors and home icon."""
    return NavBar(
        A("Strategy Diagnostic", href="#daily-inference"),
        A("Docs", href="/docs"),
        A(UkIcon("refresh-cw", height=18, width=18),
          href="/trading",
          style=f"cursor:pointer; color:{_IMMACULATA};",
          uk_tooltip="title: Clear and reset"),
        brand=A(
            DivLAligned(
                UkIcon("brain-circuit", height=20, width=20),
                H3("USD Strategy Advisor", style=f"color:{_FOUNDERS}; margin:0;"),
                Span("| Validex", style=f"color:{_IMMACULATA}; font-size:0.85rem; margin-left:0.5rem;"),
            ),
            href="/",
        ),
        sticky=True,
        uk_scrollspy_nav=True,
        cls="p-3",
        style=f"border-bottom:2px solid {_IMMACULATA};",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# daily inference section
# ═══════════════════════════════════════════════════════════════════════════════

# here we have side bar (takes 33% of the screen on the right side) that includes:
# a date range, to allow the user to select a date for prediction (start with single day only). dates prior to march 1st 2026 should spawn a toaster
# that warns that date was included on the model training, toast should last for 3 seconds, disappear, but a warning icon should remain on the sidebar.

# a small "today's" data checkmark should allow the user to automatically select today's date for inference

# a dropdown for the given stock we are inferencing should be included, the dropdown options should be (at first) a set of 10 hard-coded stocks defined on the
#component function as a list

# the side bar should also include a compute inference button (which should trigger an inference_call @rt)
# the inference_call @rt takes the data from the side_bar as a whole, unpacking the dropdown and date values and for now printing them out only.

# the inference bar (the other 67% of the screen on the left-side)
# should be composed of a table like structure (takes about 30% of the container), that includes resulting statistics
# such as ohclv data, prediction, sharpe ratio, expected profit

# the 37% left of the inference bar should be left for a plot of the stock to be render (candle-bar graph), which librasry to use for rendering yet
# to be decided
def _daily_inference_section():
    """Daily inference section: sidebar (33% right) + results panel (67% left).

    Returns:
        Section Div.
    """
    return Section(
        H3("Strategy Diagnostic", style=f"color:{_FOUNDERS};"),
        P("Decision-support diagnostic. Evaluate model predictions, strategy scoring, and market context for covered call selection.",
          cls=TextPresets.muted_sm),
        DividerLine(),
        Grid(
            # results panel -left (spans 4 of 7 cols)
            Div(_inference_results_panel(), cls="col-span-4"),
            # sidebar -right (spans 3 of 7 cols)
            Div(_inference_sidebar(), cls="col-span-3"),
            cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
        ),
        # Claude analysis panel -full width below the grid, swapped via htmx
        _claude_analysis_panel(),
        id="daily-inference",
    )


def _inference_sidebar():
    """Sidebar with date picker, today checkbox, ticker dropdown, and compute button.

    Returns:
        Card with form controls.
    """
    return Card(
        # date input -defaults to today
        Label("Date"),
        Input(type="date", name="date", id="inference-date", value=date.today().isoformat()),
        # today checkbox. Purehtmx, no JS. hx_get swaps the date input for today's value
        Div(
            Label(
                CheckboxX(id="today-check", checked=True,
                          hx_get="/today_date",
                          hx_target="#inference-date",
                          hx_swap="outerHTML"),
                " Use today's date",
            ),
            cls="mt-2",
        ),
        # ticker dropdown
        Label("Ticker", cls="mt-3"),
        Select(
            *[Option(t, value=t) for t in TICKERS],
            name="ticker", id="inference-ticker",
            disabled=True, style="opacity:0.5;",
        ),
        # batch checkbox -default checked (all stocks), uncheck for single ticker
        Div(
            Label(
                CheckboxX(id="batch-check", name="batch", checked=True,
                          hx_get="/toggle_ticker",
                          hx_include="#batch-check",
                          hx_target="#inference-ticker",
                          hx_swap="outerHTML"),
                " All Stocks",
            ),
            cls="mt-2",
        ),
        # compute button -posts to /inference_call, swaps results panel
        Button("Run Diagnostic",
               cls="w-full mt-4",
               style=f"background-color:{_IMMACULATA}; color:white;",
               hx_post="/inference_call",
               hx_include="#inference-date, #inference-ticker, #batch-check",
               hx_target="#inference-results",
               hx_swap="innerHTML",
               hx_indicator="#inference-spinner"),
        # loading indicator shown while request is in flight
        Loading(htmx_indicator=True, id="inference-spinner"),
        header=H4("Parameters", style=f"color:{_FOUNDERS};"),
    )


def _inference_results_panel():
    """Placeholder panel where inference results get swapped in via hx-target.

    Returns:
        Div with id for hx-target.
    """
    # show the same layout as computed results, but with blank values
    blank_data = {
        "Ticker": "-", "Date": "-", "Month": "-", "Model": "-",
        "Prediction": "-", "Confidence": "-",
        "Baseline": "-", "Sample": "-",
    }
    header = ["Metric", "Value"]
    body = [{"Metric": k, "Value": v} for k, v in blank_data.items()]
    return Div(
        Card(
            Div(
                # table side -left ~40%
                Div(
                    TableFromDicts(header_data=header, body_data=body),
                    style="flex:2;",
                ),
                # chart side -right ~60%
                Div(
                    P("Chart placeholder", cls=TextPresets.muted_sm),
                    style=f"flex:3; min-height:200px; border:1px dashed {_TORERO}; border-radius:8px; "
                          "display:flex; align-items:center; justify-content:center;",
                ),
                style="display:flex; gap:1rem;",
            ),
            header=H4("Diagnostic Results", style=f"color:{_FOUNDERS};"),
        ),
        id="inference-results",
    )


def _claude_analysis_panel():
    """Placeholder panel for Claude AI analysis, swapped after inference completes.

    Returns:
        Div with id for hx-target.
    """
    return Div(
        Card(
            P("AI analysis will appear here after inference completes.",
              cls=TextPresets.muted_sm, style="text-align:center; padding:1rem 0;"),
            header=Div(
                UkIcon("brain", height=20, width=20),
                H4(" Claude Analysis", style=f"color:{_FOUNDERS}; display:inline;"),
                style="display:flex; align-items:center; gap:0.5rem;",
            ),
        ),
        id="claude-analysis",
        style="margin-top:1rem;",
    )


def claude_analysis_card(data: dict):
    """Render full analysis: scoring table, context summary, and Claude report.

    Args:
        data: Dict with 'analysis', 'scoring', 'context', or 'error'.

    Returns:
        Div with scoring card, context card, and analysis card.
    """
    try:
        if "error" in data:
            return Card(
                P(f"Analysis unavailable: {data['error']}",
                  cls=TextPresets.muted_sm, style="font-style:italic;"),
                header=Div(
                    UkIcon("brain", height=20, width=20),
                    H4(" Claude Analysis", style=f"color:{_FOUNDERS}; display:inline;"),
                    style="display:flex; align-items:center; gap:0.5rem;",
                ),
            )

        # ── Scoring table ──
        scoring = data.get("scoring", {})
        _strat_tips = {
            "baseline": "Always sell 10% OTM short-dated calls. No model involved. Purebenchmark.",
            "argmax": "Model's single highest-probability pick. Simple and transparent.",
            "risk_adjusted": "Picks the bucket that maximizes probability times expected return.",
            "conservative": "Scored strategy: spreads across 7 positions, prioritizes low-cost trades.",
        }
        scoring_rows = []
        for key, label in [("baseline", "Baseline (OTM10)"), ("argmax", "Argmax"),
                           ("risk_adjusted", "Risk-Adjusted")]:
            s = scoring.get(key, {})
            ret = s.get("return", 0)
            scoring_rows.append(Tr(Td(Span(label, _tip(_strat_tips[key]))), Td(f"{ret:.4%}")))
        for preset in ["conservative"]:
            s = scoring.get("presets", {}).get(preset, {})
            ret = s.get("return", 0)
            scoring_rows.append(Tr(Td(Span(preset.title(), _tip(_strat_tips[preset]))), Td(f"{ret:.4%}")))

        scoring_card = Card(
            Table(
                Thead(Tr(
                    Th("Strategy"),
                    Th(Span("Return", _tip("Realized covered call return for this month using actual options data."))),
                )),
                Tbody(*scoring_rows),
                cls="uk-table uk-table-small uk-table-divider",
            ),
            header=Div(
                UkIcon("bar-chart-2", height=18, width=18),
                H4(" Strategy Scoring", style=f"color:{_FOUNDERS}; display:inline;"),
                style="display:flex; align-items:center; gap:0.5rem;",
            ),
        )

        # ── Context summary ──
        ctx = data.get("context", {})
        is_batch_ctx = ctx.get("batch", False)
        price = ctx.get("price", {})
        track = ctx.get("track_record", {})
        features = ctx.get("features", {})
        iv = features.get("iv", {})

        context_items = []
        if is_batch_ctx:
            context_items.extend([
                Tr(Td(Span("Trend", _tip("Overall market direction based on moving average crossovers."))),
                   Td(str(price.get("trend", "-")))),
                Tr(Td(Span("Vol Regime", _tip("Current volatility environment across the portfolio."))),
                   Td(str(price.get("vol_regime", "-")))),
                Tr(Td(Span("Avg 20d Realized Vol", _tip("Average annualized realized volatility over the last 20 trading days across all tickers."))),
                   Td(f"{price.get('vol_20d', 0):.1%}")),
                Tr(Td(Span("60d Return (Avg)", _tip("Average price change over the last 60 trading days across all tickers."))),
                   Td(f"{price.get('period_return', 0):.1%}")),
                Tr(Td(Span("Max Drawdown", _tip("Largest peak-to-current drop across the portfolio (60d window)."))),
                   Td(f"{price.get('drawdown_from_peak', 0):.1%}")),
            ])
            if track:
                context_items.extend([
                    Tr(Td(Span("Avg Model Accuracy", _tip("How often the model's historical predictions were correct, averaged across tickers."))),
                       Td(f"{track.get('overall_accuracy', 0):.1%}")),
                    Tr(Td(Span("Avg Recent 12m", _tip("Model accuracy over the most recent 12 months only."))),
                       Td(f"{track.get('recent_12m_accuracy', 0):.1%}")),
                ])
        else:
            if price and "error" not in price:
                context_items.extend([
                    Tr(Td(Span("Trend", _tip("Price direction based on moving average crossovers. Bullish = price above key averages."))),
                       Td(f"{price.get('trend', '?').title()}")),
                    Tr(Td(Span("Vol Regime", _tip("Current volatility environment. High vol favors shorter-dated options for faster premium capture."))),
                       Td(f"{price.get('vol_regime', '?').replace('_', ' ').title()}")),
                    Tr(Td(Span("20d Realized Vol", _tip("Annualized realized volatility computed from the last 20 trading days."))),
                       Td(f"{price.get('vol_20d', 0):.1%}")),
                    Tr(Td(Span("60d Return", _tip("Total price change over the last 60 trading days."))),
                       Td(f"{price.get('period_return', 0):.1%}")),
                ])
            if iv:
                context_items.append(
                    Tr(Td(Span("IV Rank", _tip("Current implied volatility as a percentile of its 12-month range. Above 0.5 = elevated vol."))),
                       Td(f"{iv.get('iv_rank', '?')}")))
            if track and "error" not in track:
                context_items.extend([
                    Tr(Td(Span("Ticker Accuracy", _tip("How often the model correctly predicted this ticker's best bucket historically."))),
                       Td(f"{track.get('overall_accuracy', 0):.1%}")),
                    Tr(Td(Span("Recent 12m", _tip("Model accuracy for this ticker over the most recent 12 months."))),
                       Td(f"{track.get('recent_12m_accuracy', 0):.1%}")),
                ])

        ctx_label = "Portfolio Context" if is_batch_ctx else "Market Context"
        context_card = Card(
            Table(
                Thead(Tr(Th("Metric"), Th("Value"))),
                Tbody(*context_items),
                cls="uk-table uk-table-small uk-table-divider",
            ) if context_items else P("Context unavailable.", cls=TextPresets.muted_sm),
            header=Div(
                UkIcon("activity", height=18, width=18),
                H4(f" {ctx_label}", style=f"color:{_FOUNDERS}; display:inline;"),
                style="display:flex; align-items:center; gap:0.5rem;",
            ),
        )

        # ── Analysis text ──
        analysis_text = data.get("analysis", "No analysis returned.")
        source = data.get("source", "unknown")
        source_badge = Span(
            f"  [{source}]",
            style=f"font-size:0.7rem; color:{_TORERO}; font-weight:400;",
        ) if source != "api" else ""

        analysis_card = Card(
            render_md(analysis_text),
            header=Div(
                UkIcon("brain", height=20, width=20),
                H4(" Claude Analysis", style=f"color:{_FOUNDERS}; display:inline;"),
                source_badge,
                style="display:flex; align-items:center; gap:0.5rem;",
            ),
        )

        # ── Batch viz cards (only in batch mode) ──
        viz_row = ""
        analytics = data.get("analytics", {})
        predictions = data.get("predictions", [])
        if analytics and predictions:
            # Confidence heatmap: tickers × models
            heatmap_data = []
            for p in predictions:
                heatmap_data.append({
                    "x": p.get("ticker", "?"),
                    "y": round(p.get("lgbm_confidence", 0) * 100, 1),
                })
            heatmap_lstm = []
            for p in predictions:
                heatmap_lstm.append({
                    "x": p.get("ticker", "?"),
                    "y": round(p.get("lstm_confidence", 0) * 100, 1),
                })

            heatmap_card = Card(
                ApexChart(opts={
                    "chart": {"type": "heatmap", "height": 180, "toolbar": {"show": False}},
                    "series": [
                        {"name": "LGBM", "data": heatmap_data},
                        {"name": "LSTM", "data": heatmap_lstm},
                    ],
                    "dataLabels": {"enabled": True, "style": {"fontSize": "11px"}},
                    "colors": [_IMMACULATA],
                    "plotOptions": {"heatmap": {"radius": 4, "colorScale": {
                        "ranges": [
                            {"from": 0, "to": 40, "color": _TORERO, "name": "Low"},
                            {"from": 40, "to": 70, "color": _IMMACULATA, "name": "Med"},
                            {"from": 70, "to": 100, "color": _FOUNDERS, "name": "High"},
                        ],
                    }}},
                    "xaxis": {"labels": {"style": {"fontSize": "10px"}}},
                }),
                header=Div(
                    UkIcon("grid", height=18, width=18),
                    H4(" Confidence Map", style=f"color:{_FOUNDERS}; display:inline;"),
                    style="display:flex; align-items:center; gap:0.5rem;",
                ),
            )

            # Prediction distribution: bucket counts
            bucket_dist = analytics.get("bucket_distribution", {})
            dist_categories = list(bucket_dist.keys())
            dist_values = list(bucket_dist.values())

            dist_card = Card(
                ApexChart(opts={
                    "chart": {"type": "bar", "height": 180, "toolbar": {"show": False}},
                    "series": [{"name": "Tickers", "data": dist_values}],
                    "xaxis": {"categories": dist_categories},
                    "colors": [_IMMACULATA],
                    "plotOptions": {"bar": {"borderRadius": 4, "columnWidth": "60%"}},
                    "dataLabels": {"enabled": True},
                }),
                header=Div(
                    UkIcon("bar-chart", height=18, width=18),
                    H4(" LSTM-CNN Predicted Classes", style=f"color:{_FOUNDERS}; display:inline;"),
                    style="display:flex; align-items:center; gap:0.5rem;",
                ),
            )

            viz_row = Div(
                Div(heatmap_card, style="flex:1;"),
                Div(dist_card, style="flex:1;"),
                style="display:flex; gap:1rem;",
            )

        # Layout: context + scoring (row 1), viz charts (row 2, batch only), analysis (row 3)
        return Div(
            Div(
                Div(context_card, style="flex:2;"),
                Div(scoring_card, style="flex:1;"),
                style="display:flex; gap:1rem;",
            ),
            viz_row,
            analysis_card,
            style="display:flex; flex-direction:column; gap:1rem;",
        )
    except Exception:
        return _fallback("Claude analysis")


def inference_results_card(data: dict):
    """Render dual-model inference results: LGBM + LSTM-CNN table + chart.

    Args:
        data: Combined dict from graph aggregate node with lgbm/lstm sub-dicts.

    Returns:
        Card Div with results table and candlestick chart.
    """
    try:
        # Table: ticker, date, LSTM first (default model), then LGBM
        table_rows = [
            Tr(Td("Ticker"), Td(data.get("ticker", "-"))),
            Tr(Td("Date"), Td(data.get("date", "-"))),
            # LSTM-CNN (primary)
            Tr(Td(Span(Strong("LSTM-CNN 7-Class"), style=f"color:{_IMMACULATA};")), Td("")),
            Tr(Td(Span("Bucket", _tip(
                "The model's recommended strike and expiry combination. "
                "7 classes: ATM/OTM5/OTM10 crossed with 30-day or 60-90 day expiry."))),
               Td(data.get("lstm_prediction", "-"))),
            Tr(Td(Span("Confidence", _tip(
                "How strongly the model favors this pick over the alternatives. "
                "Higher is more decisive, but not necessarily more accurate."))),
               Td(f"{data.get('lstm_confidence', 0):.1%}")),
            # LGBM (secondary)
            Tr(Td(Span(Strong("LGBM 3-Class"), style=f"color:{_IMMACULATA};")), Td("")),
            Tr(Td(Span("Bucket", _tip(
                "Moneyness bucket (ATM/OTM5/OTM10) plus maturity (SHORT or LONG) "
                "determined by an IV-rank rule."))),
               Td(data.get("model_bucket", "-"))),
            Tr(Td(Span("Confidence", _tip(
                "LGBM prediction probability for its top pick."))),
               Td(f"{data.get('model_confidence', 0):.1%}")),
        ]

        # OHLC chart
        chart_data = data.get("chart_data", [])
        if chart_data:
            chart_el = ApexChart(opts={
                "chart": {"type": "candlestick", "height": 320},
                "series": [{"name": "Price", "data": chart_data}],
                "xaxis": {"type": "datetime"},
                "yaxis": {"tooltip": {"enabled": True}},
                "plotOptions": {"candlestick": {
                    "colors": {"upward": _IMMACULATA, "downward": _FOUNDERS},
                }},
            })
        else:
            chart_el = P("No chart data available.", cls=TextPresets.muted_sm)

        # Warnings
        snap_warning = None
        if data.get("snapped"):
            snap_warning = Div(
                P(f"No data for requested date. Showingnearest available: {data.get('month', '?')}",
                  style="margin:0;"),
                cls="uk-alert uk-alert-warning",
                style="padding:0.75rem 1rem; margin-bottom:0.5rem; border-radius:6px; "
                      f"background:{_TORERO}22; border-left:4px solid {_TORERO};",
            )

        live_warning = None
        if data.get("is_live"):
            live_warning = Div(
                P(Strong("Experimental -Live Pipeline"), style="margin:0 0 0.25rem 0;"),
                P("Features computed from real-time market data (yfinance), not the historical "
                  "training dataset. Predictions may differ from backtested performance due to "
                  "data source differences.",
                  style="margin:0;", cls=TextPresets.muted_sm),
                cls="uk-alert",
                style="padding:0.75rem 1rem; margin-bottom:0.5rem; border-radius:6px; "
                      f"background:{_IMMACULATA}12; border-left:4px solid {_IMMACULATA};",
            )

        disclaimer = P(
            "Confidence reflects how strongly each model favors its pick -"
            "it does not guarantee the outcome. Past model accuracy varies by market conditions.",
            cls=TextPresets.muted_sm,
            style="font-style:italic; margin-top:0.75rem; padding-top:0.5rem; "
                  f"border-top:1px solid {_TORERO}30;",
        )

        # htmx out-of-band swap: replace the Claude analysis panel with a loading
        # state that auto-fires the API call. This Div targets #claude-analysis
        # via hx-swap-oob, so it replaces the placeholder in one shot.
        ticker = data.get("ticker", "")
        date_val = data.get("date", "")
        analysis_oob = Div(
            Card(
                Div(
                    Loading(),
                    P("Analyzing predictions...", cls=TextPresets.muted_sm,
                      style="text-align:center; margin-top:0.5rem;"),
                    style="padding:1rem 0;",
                ),
                header=Div(
                    UkIcon("brain", height=20, width=20),
                    H4(" Claude Analysis", style=f"color:{_FOUNDERS}; display:inline;"),
                    style="display:flex; align-items:center; gap:0.5rem;",
                ),
            ),
            hx_get=f"/claude_analysis_call?ticker={ticker}&date={date_val}",
            hx_trigger="load",
            hx_swap="innerHTML",
            id="claude-analysis",
            hx_swap_oob="true",
            style="margin-top:1rem;",
        )

        return Div(
            live_warning if live_warning else "",
            snap_warning if snap_warning else "",
            Card(
                Div(
                    Div(
                        Table(
                            Thead(Tr(Th("Metric"), Th("Value"))),
                            Tbody(*table_rows),
                            cls="uk-table uk-table-small uk-table-divider",
                        ),
                        disclaimer,
                        style="flex:2;",
                    ),
                    Div(chart_el, style="flex:3; min-height:200px;"),
                    style="display:flex; gap:1rem;",
                ),
                header=H4(f"Diagnostic -{data.get('ticker', '?')} @ {data.get('date', '?')}",
                           style=f"color:{_FOUNDERS};"),
            ),
            # OOB swap replaces the Claude analysis panel with loading → auto-fetch
            analysis_oob,
        )
    except Exception:
        return _fallback("inference results")


def _batch_ticker_chart_modal(ticker: str, date: str):
    """Render a per-ticker candlestick chart inside a modal via lazy loading.

    The chart is NOT pre-loaded; itfetches via htmx when the modal opens.
    ApexCharts can't render in hidden containers (0x0 dimensions), so we
    use hx-trigger='intersect once' to load after the modal is visible.

    Args:
        ticker: Stock symbol.
        date: Date string for the chart endpoint.

    Returns:
        Div with trigger icon + modal with lazy-loaded chart.
    """
    modal_id = f"batch-chart-{ticker.lower()}"

    return Div(
        A(UkIcon("search", height=18, width=18),
          href=f"#{modal_id}", uk_toggle="",
          style=f"cursor:pointer; color:{_IMMACULATA};"),
        Modal(
            Div(
                Loading(),
                hx_get=f"/ticker_chart?ticker={ticker}&date={date}",
                hx_trigger="intersect once",
                hx_swap="innerHTML",
                style="min-height:350px;",
            ),
            header=f"{ticker}: Price Chart",
            id=modal_id,
            dialog_cls="uk-modal-dialog-large",
        ),
    )


def batch_results_card(data: dict):
    """Render batch inference results: summary stats + expandable modal with per-ticker rows.

    The summary card shows aggregate statistics. A search icon opens a modal
    with all tickers, each row having its own chart expand icon.

    Args:
        data: Dict with 'results' (list of per-ticker dicts) and 'summary' stats.

    Returns:
        Div with summary card and nested modals.
    """
    try:
        summary = data.get("summary", {})
        results = data.get("results", [])

        # ── Summary stats card ──
        stats_display = {
            "Date": summary.get("date", "-"),
            "Models": summary.get("model", "-"),
            "Tickers": str(summary.get("n_tickers", 0)),
        }
        stats_header = ["Metric", "Value"]
        stats_body = [{"Metric": k, "Value": v} for k, v in stats_display.items()]

        # ── Per-ticker modal table ──
        detail_modal_id = "batch-detail-modal"
        detail_rows = []
        for r in results:
            if "error" in r:
                continue
            detail_rows.append({
                "Ticker": r.get("ticker", "?"),
                "Prediction": r.get("model_bucket", "-"),
                "Confidence": f"{r.get('model_confidence', 0):.1%}",
                "Correct": "Y" if r.get("model_correct") else "N",
                "Sample": r.get("sample_type", "-"),
            })

        detail_header = ["Ticker", "LSTM-CNN", "LSTM Conf", "LGBM", "LGBM Conf", "Chart"]

        # Build table rows manually to embed the chart icon in each row
        batch_date = summary.get("date", "")
        table_rows = []
        for r in results:
            if "error" in r:
                continue
            ticker = r.get("ticker", "?")
            table_rows.append(
                Tr(
                    Td(Strong(ticker)),
                    Td(r.get("lstm_prediction", "-")),
                    Td(f"{r.get('lstm_confidence', 0):.1%}"),
                    Td(r.get("model_bucket", "-")),
                    Td(f"{r.get('model_confidence', 0):.1%}"),
                    Td(_batch_ticker_chart_modal(ticker, batch_date)),
                )
            )

        detail_table = Table(
            Thead(Tr(*[Th(h) for h in detail_header])),
            Tbody(*table_rows),
            cls="uk-table uk-table-small uk-table-divider",
        )

        # Snap warning
        snapped_tickers = [r.get("ticker") for r in results
                           if "error" not in r and r.get("snapped")]
        batch_snap_warning = None
        if snapped_tickers:
            batch_snap_warning = Div(
                P(f"Date snapped for: {', '.join(snapped_tickers)} - showing nearest available month.",
                  style="margin:0;"),
                cls="uk-alert uk-alert-warning",
                style="padding:0.75rem 1rem; margin-bottom:0.5rem; border-radius:6px; "
                      f"background:{_TORERO}22; border-left:4px solid {_TORERO};",
            )

        # Live warning
        live_tickers = [r.get("ticker") for r in results
                        if "error" not in r and r.get("is_live")]
        batch_live_warning = None
        if live_tickers:
            batch_live_warning = Div(
                P(Strong("Experimental - Live Pipeline"), style="margin:0 0 0.25rem 0;"),
                P("Features computed from real-time market data (yfinance), not the historical "
                  "training dataset. Predictions may differ from backtested performance.",
                  style="margin:0;", cls=TextPresets.muted_sm),
                cls="uk-alert",
                style="padding:0.75rem 1rem; margin-bottom:0.5rem; border-radius:6px; "
                      f"background:{_IMMACULATA}12; border-left:4px solid {_IMMACULATA};",
            )

        return Div(
            batch_live_warning if batch_live_warning else "",
            batch_snap_warning if batch_snap_warning else "",
            Card(
                Div(
                    # Summary table
                    Div(
                        TableFromDicts(header_data=stats_header, body_data=stats_body),
                        style="flex:1;",
                    ),
                    # Expand icon for detail modal
                    Div(
                        A(
                            Div(
                                UkIcon("list", height=32, width=32),
                                P("View All Results", cls=(TextT.sm,), style="margin-top:0.25rem;"),
                                style="display:flex; flex-direction:column; align-items:center;",
                            ),
                            href=f"#{detail_modal_id}", uk_toggle="",
                            style=f"cursor:pointer; color:{_IMMACULATA}; text-decoration:none;",
                        ),
                        style="flex:1; display:flex; align-items:center; justify-content:center;",
                    ),
                    style="display:flex; gap:1rem;",
                ),
                header=H4(f"Batch Diagnostic -{summary.get('date', '?')} ({summary.get('n_tickers', 0)} tickers)",
                          style=f"color:{_FOUNDERS};"),
            ),
            # Detail modal with per-ticker rows + chart icons
            Modal(
                detail_table,
                header=f"All Results -{summary.get('date', '?')}",
                id=detail_modal_id,
                dialog_cls="uk-modal-dialog-large",
            ),
            # OOB trigger for Claude analysis (batch mode)
            Div(
                Card(
                    Div(
                        Loading(),
                        P("Analyzing all tickers...", cls=TextPresets.muted_sm,
                          style="text-align:center; margin-top:0.5rem;"),
                        style="padding:1rem 0;",
                    ),
                    header=Div(
                        UkIcon("brain", height=20, width=20),
                        H4(" Claude Analysis", style=f"color:{_FOUNDERS}; display:inline;"),
                        style="display:flex; align-items:center; gap:0.5rem;",
                    ),
                ),
                hx_get=f"/claude_analysis_call?date={batch_date}&batch=true",
                hx_trigger="load",
                hx_swap="innerHTML",
                id="claude-analysis",
                hx_swap_oob="true",
                style="margin-top:1rem;",
            ),
        )

    except Exception:
        return _fallback("batch inference results")


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard sub-components (sidebar + results panels)
# Used by the dashboard doc sections (dash-strategy, dash-model, dash-mlflow)
# ═══════════════════════════════════════════════════════════════════════════════


def _backtest_sidebar():
    """Sidebar with year dropdown, budget input, and run button.

    Returns:
        Card with backtest controls.
    """
    year_options = [fh.Option("All Years", value="all", selected=True)] + [
        fh.Option(str(y), value=str(y)) for y in range(2008, 2026)
    ]
    return Card(
        Label("Time Window"),
        fh.Select(*year_options, name="year", id="backtest-year", cls="uk-select"),
        # View mode toggle
        Label("View", cls="mt-3"),
        Div(
            Label(
                Input(type="radio", name="mode", value="absolute", checked=True,
                      cls="uk-radio", style="margin-right:0.3rem;"),
                " Absolute",
                style="margin-right:1rem;",
            ),
            Label(
                Input(type="radio", name="mode", value="delta",
                      cls="uk-radio", style="margin-right:0.3rem;"),
                " vs Baseline",
            ),
            style="display:flex; margin-top:0.25rem;",
        ),
        Button("Run Backtest",
               cls="w-full mt-4",
               style=f"background-color:{_IMMACULATA}; color:white;",
               hx_post="/backtest_call",
               hx_include="#backtest-year, [name='mode']:checked",
               hx_target="#backtest-results",
               hx_swap="innerHTML",
               hx_indicator="#backtest-spinner"),
        Loading(htmx_indicator=True, id="backtest-spinner"),
        header=H4("Parameters", style=f"color:{_FOUNDERS};"),
    )


def _backtest_results_panel():
    """Placeholder panel where backtest results get swapped in.

    Returns:
        Div with id for hx-target.
    """
    return Div(
        Card(
            P("Select a preset and click Run Backtest.", cls=TextPresets.muted_sm),
            header=H4("Backtest Results", style=f"color:{_FOUNDERS};"),
        ),
        id="backtest-results",
    )


def backtest_results_card(data: dict, mode: str = "absolute"):
    """Render dual-model backtest results.

    Args:
        data: Combined backtest report from run_backtest_all().
        mode: 'absolute' shows raw metrics, 'delta' shows difference vs baseline.

    Returns:
        Div with two strategy comparison tables.
    """
    try:
        year = data.get("year", "all")
        lgbm = data.get("lgbm", {})
        lstm = data.get("lstm", {})
        baseline_m = data.get("baseline", {}).get("metrics", {})
        lgbm_r = data.get("lgbm_range", {})
        lstm_r = data.get("lstm_range", {})
        is_delta = mode == "delta"

        def fmt_pct(v): return f"{v:.1%}"
        def fmt_ratio(v): return f"{v:.2f}"
        def fmt_delta_pct(v): return f"{v:+.1%}"
        def fmt_delta_ratio(v): return f"{v:+.2f}"

        metric_defs = [
            ("Ann. Return", "annualized_return", fmt_pct, fmt_delta_pct,
             "Total return converted to a yearly rate."),
            ("Sharpe", "sharpe_ratio", fmt_ratio, fmt_delta_ratio,
             "Return per unit of risk. Above 1 is good, above 2 is very good."),
            ("Max Drawdown", "max_drawdown", fmt_pct, fmt_delta_pct,
             "Biggest peak-to-trough drop before recovery."),
            ("Hit Rate", "hit_rate", fmt_pct, fmt_delta_pct,
             "Percentage of months with a positive return."),
            ("Avg P / Avg L", "avg_p_l", fmt_ratio, fmt_delta_ratio,
             "Average win divided by average loss."),
        ]

        def _strat_table(model_data: dict, model_label: str):
            am = model_data.get("argmax", {}).get("metrics", {})
            ra = model_data.get("risk_adjusted", {}).get("metrics", {})
            cm = model_data.get("conservative", {}).get("metrics", {})

            rows = []
            for label, key, fmt_abs, fmt_d, tip in metric_defs:
                bv = baseline_m.get(key, 0)
                if is_delta:
                    rows.append(Tr(
                        Td(Span(label, _tip(tip))),
                        Td("-"),
                        Td(fmt_d(am.get(key, 0) - bv)),
                        Td(fmt_d(ra.get(key, 0) - bv)),
                        Td(fmt_d(cm.get(key, 0) - bv)),
                    ))
                else:
                    rows.append(Tr(
                        Td(Span(label, _tip(tip))),
                        Td(fmt_abs(bv)),
                        Td(fmt_abs(am.get(key, 0))),
                        Td(fmt_abs(ra.get(key, 0))),
                        Td(fmt_abs(cm.get(key, 0))),
                    ))

            baseline_header = "Baseline" if not is_delta else "Baseline (ref)"
            return Div(
                H4(model_label, style=f"color:{_IMMACULATA}; font-size:1rem; margin-bottom:0.5rem;"),
                Table(
                    Thead(Tr(
                        Th("Metric"),
                        Th(Span(baseline_header, _tip("OTM10 short-dated on all tickers. No model."))),
                        Th(Span("Argmax", _tip("Model's top pick per ticker, equal weight."))),
                        Th(Span("Risk-Adj", _tip("P(bucket) x E[return], expanding averages."))),
                        Th(Span("Conservative", _tip("Scored: 7 positions, low-cost priority."))),
                    )),
                    Tbody(*rows),
                    cls="uk-table uk-table-small uk-table-divider",
                ),
            )

        period_label = f"Year: {year}" if year != "all" else "All Years"
        mode_label = "vs Baseline" if is_delta else "Absolute"

        lgbm_period = f"{lgbm_r.get('start', '?')}–{lgbm_r.get('end', '?')} ({lgbm_r.get('n_months', '?')} mo)"
        lstm_period = f"{lstm_r.get('start', '?')}–{lstm_r.get('end', '?')} ({lstm_r.get('n_months', '?')} mo)"

        return Div(
            Card(
                P(f"{period_label} | {mode_label}",
                  cls=TextPresets.muted_sm),
                DividerLine(),
                _strat_table({"argmax": lstm.get("argmax", {}),
                              "risk_adjusted": lstm.get("risk_adjusted", {}),
                              "conservative": lstm.get("conservative", {})},
                             f"LSTM-CNN 7-Class -{lstm_period}"),
                DividerLine(),
                _strat_table(lgbm, f"LGBM 3-Class -{lgbm_period}"),
                header=H4("Strategy Comparison", style=f"color:{_FOUNDERS};"),
            ),
        )

    except Exception:
        return _fallback("backtest results")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Performance tab components
# ═══════════════════════════════════════════════════════════════════════════════

def _model_sidebar():
    """Sidebar with sample type and year filters for model performance.

    Returns:
        Card with filter controls.
    """
    sample_options = [
        fh.Option("All", value="all"),
        fh.Option("Train Dataset", value="train"),
        fh.Option("Test Dataset", value="test", selected=True),
    ]
    year_options = [fh.Option("All Years", value="all", selected=True)] + [
        fh.Option(str(y), value=str(y)) for y in range(2008, 2026)
    ]
    return Card(
        Label("Sample Type"),
        fh.Select(*sample_options, name="sample_type", id="model-sample-type", cls="uk-select",
                  hx_get="/toggle_model_year",
                  hx_include="#model-sample-type",
                  hx_target="#model-year",
                  hx_swap="outerHTML"),
        Label("Year", cls="mt-3"),
        fh.Select(*year_options, name="year", id="model-year", cls="uk-select",
                  hx_get="/toggle_model_sample",
                  hx_include="#model-year",
                  hx_target="#model-sample-type",
                  hx_swap="outerHTML"),
        Button("Load Metrics",
               cls="w-full mt-4",
               style=f"background-color:{_IMMACULATA}; color:white;",
               hx_post="/model_metrics_call",
               hx_include="#model-sample-type, #model-year",
               hx_target="#model-results",
               hx_swap="innerHTML",
               hx_indicator="#model-spinner"),
        Loading(htmx_indicator=True, id="model-spinner"),
        header=H4("Filters", style=f"color:{_FOUNDERS};"),
    )


def _model_performance_panel():
    """Placeholder panel for model performance metrics.

    Lazy-loaded on first view or when Load Metrics is clicked.

    Returns:
        Div with id for hx-target.
    """
    return Div(
        Card(
            Div(
                Loading(),
                hx_get="/model_metrics_call?sample_type=test&year=all",
                hx_trigger="intersect once",
                hx_swap="innerHTML",
                hx_target="#model-results",
            ),
            P("Loading model performance metrics...", cls=TextPresets.muted_sm),
            header=H4("Model Performance", style=f"color:{_FOUNDERS};"),
        ),
        id="model-results",
    )


def _model_summary_block(d: dict, classes: list):
    """Render a single model's metrics: summary + per-class + confidence.

    Args:
        d: Metrics dict from compute_model_metrics (LGBM or LSTM).
        classes: List of class names for the per-class table.

    Returns:
        Div with model metrics.
    """
    if "error" in d:
        return P(f"Error: {d['error']}", cls="uk-text-danger")

    conf = d.get("confidence", {})

    # Summary row
    summary = Div(
        *[Div(
            P(Strong(label), _tip(tip), cls=TextPresets.muted_sm, style="margin-bottom:0.25rem;"),
            P(val, style="font-size:1.3rem; font-weight:600; margin:0;"),
            style="text-align:center; padding:0.5rem;",
        ) for label, val, tip in [
            ("Accuracy", f"{d.get('accuracy', 0):.1%}",
             "How often the model picks the correct bucket overall."),
            ("Macro F1", f"{d.get('macro_f1', 0):.3f}",
             "Average F1 across all classes, weighted equally. Balances how precise and how complete the model is. Random guessing would score ~0.14 for 7 classes or ~0.33 for 3 classes."),
            ("Samples", f"{d.get('n_samples', 0):,}",
             "Number of predictions evaluated in this view."),
        ]],
        style="display:grid; grid-template-columns:repeat(3, 1fr); gap:0.5rem;",
    )

    # Per-class table
    pc_header = Tr(
        Th("Class"),
        Th(Span("Prec", _tip("Of all times the model predicted this class, how often was it actually correct."))),
        Th(Span("Recall", _tip("Of all real instances of this class, how many did the model find."))),
        Th(Span("F1", _tip("Balance between precision and recall. 1.0 is perfect."))),
        Th(Span("Support", _tip("How many times this class actually appeared in the data."))),
    )
    pc_rows = []
    for cls in classes:
        m = d.get("per_class", {}).get(cls, {})
        pc_rows.append(Tr(
            Td(Strong(cls)), Td(f"{m.get('precision', 0):.3f}"),
            Td(f"{m.get('recall', 0):.3f}"), Td(f"{m.get('f1', 0):.3f}"),
            Td(str(m.get("support", 0))),
        ))

    # Confidence
    conf_line = P(
        Span("Confidence", _tip(
            "Average model certainty when it gets the answer right vs wrong. "
            "If these numbers are close, the model can't tell when it's guessing.")),
        f", correct: {conf.get('avg_when_correct', 0):.1%}, "
        f"incorrect: {conf.get('avg_when_incorrect', 0):.1%}, "
        f"overall: {conf.get('overall_avg', 0):.1%}",
        cls=TextPresets.muted_sm, style="margin-top:0.5rem;",
    )

    return Div(
        summary, DividerLine(),
        Table(Thead(pc_header), Tbody(*pc_rows),
              cls="uk-table uk-table-small uk-table-divider"),
        conf_line,
    )


def model_performance_card(data: dict):
    """Render dual-model performance metrics side by side.

    Args:
        data: Dict with 'lgbm' and 'lstm' sub-dicts from /model_metrics.

    Returns:
        Div with both models' metrics and per-year breakdown.
    """
    try:
        lgbm = data.get("lgbm", {})
        lstm = data.get("lstm", {})

        year_filter = lgbm.get("year_filter", "all")
        sample_filter = lgbm.get("sample_filter", "all")
        filter_label = f"Year: {year_filter}" if year_filter != "all" else "All Years"
        sample_labels = {"all": "All Data", "train": "Train", "test": "Test", "validation": "Validation"}
        sample_label = sample_labels.get(sample_filter, sample_filter)

        # Training scope disclaimer
        disclaimer = Div(
            P("These models were trained on different data, targets, and time periods -"
              "direct comparison requires context.",
              style="font-weight:600; margin-bottom:0.25rem;"),
            P("LSTM-CNN: 35 daily features (price + fundamentals + FRED macro), "
              "50-day sliding windows, 7-class target (moneyness x maturity). "
              "Train: pre-2022 | Validation: 2022-2023 | Test: 2024.",
              cls=TextPresets.muted_sm),
            P("LGBM: 34 monthly features (technical + IV), single-row lookup, "
              "3-class target (moneyness only, maturity via IV-rank rule). "
              "Train: pre-2025 (walk-forward annual) | Test: 2025.",
              cls=TextPresets.muted_sm),
            style=f"background:{_TORERO}10; border-left:3px solid {_TORERO}; "
                  "padding:0.75rem 1rem; border-radius:4px; margin-bottom:0.75rem;",
        )

        # Two-column layout: LSTM (primary) | LGBM
        model_columns = Div(
            Div(
                H4("LSTM-CNN 7-Class", style=f"color:{_IMMACULATA}; font-size:1rem;"),
                _model_summary_block(lstm, [
                    "ATM_30", "ATM_60", "ATM_90", "OTM10_30",
                    "OTM10_60_90", "OTM5_30", "OTM5_60_90",
                ]),
                style=f"flex:1; padding-right:1rem; border-right:1px solid {_TORERO}40;",
            ),
            Div(
                H4("LGBM 3-Class", style=f"color:{_IMMACULATA}; font-size:1rem;"),
                _model_summary_block(lgbm, ["ATM", "OTM5", "OTM10"]),
                style="flex:1; padding-left:1rem;",
            ),
            style="display:flex; gap:0; margin-top:0.5rem;",
        )

        # Per-year breakdown (LGBM -has longer history)
        per_year = lgbm.get("per_year", [])
        year_header = ["Year", "Accuracy", "Macro F1", "Samples"]
        year_rows = [
            {"Year": str(y["year"]), "Accuracy": f"{y['accuracy']:.1%}",
             "Macro F1": f"{y['macro_f1']:.3f}", "Samples": str(y["n_samples"])}
            for y in per_year
        ]

        # Per-year breakdowns
        year_header = ["Year", "Accuracy", "Macro F1", "Samples"]

        lstm_per_year = lstm.get("per_year", [])
        lstm_year_rows = [
            {"Year": str(y["year"]), "Accuracy": f"{y['accuracy']:.1%}",
             "Macro F1": f"{y['macro_f1']:.3f}", "Samples": str(y["n_samples"])}
            for y in lstm_per_year
        ]

        lgbm_per_year = lgbm.get("per_year", [])
        lgbm_year_rows = [
            {"Year": str(y["year"]), "Accuracy": f"{y['accuracy']:.1%}",
             "Macro F1": f"{y['macro_f1']:.3f}", "Samples": str(y["n_samples"])}
            for y in lgbm_per_year
        ]

        show_yearly = year_filter == "all"

        return Div(
            Card(
                P(f"{filter_label} | {sample_label}", cls=TextPresets.muted_sm),
                DividerLine(),
                disclaimer,
                model_columns,
                header=H4("Model Performance", style=f"color:{_FOUNDERS};"),
            ),
            Card(
                TableFromDicts(header_data=year_header, body_data=lstm_year_rows),
                header=H4("LSTM-CNN Performance by Year", style=f"color:{_FOUNDERS};"),
                cls="mt-2",
            ) if show_yearly and lstm_year_rows else "",
            Card(
                TableFromDicts(header_data=year_header, body_data=lgbm_year_rows),
                header=H4("LGBM Performance by Year", style=f"color:{_FOUNDERS};"),
                cls="mt-2",
            ) if show_yearly and lgbm_year_rows else "",
        )

    except Exception:
        return _fallback("model performance")


def mlflow_experiments_card(data: dict):
    """Render MLflow experiment tracking: deduplicated runs table with artifact modals.

    Note: MLflow tracking covers the team's deep learning and XGBoost experiments
    (7-class task). The production LGBM 3-class model was trained separately via
    walk-forward validation in the experiment notebooks.

    Args:
        data: Dict from load_experiments() with experiment runs.

    Returns:
        Div with experiment table and confusion matrix / ROC curve modals.
    """
    try:
        experiments = data.get("experiments", [])
        if not experiments:
            return Card(
                P("No MLflow experiments found.", cls=TextPresets.muted_sm),
                header=H4("MLflow Experiments", style=f"color:{_FOUNDERS};"),
            )

        all_runs = []
        for exp in experiments:
            all_runs.extend(exp["runs"])

        if not all_runs:
            return Card(
                P("No experiment runs found.", cls=TextPresets.muted_sm),
                header=H4("MLflow Experiments", style=f"color:{_FOUNDERS};"),
            )

        # Build table rows + modals
        header = Tr(
            Th("Model"),
            Th("Variant"),
            Th("Classes"),
            Th("Features"),
            Th("Val F1"),
            Th("Test F1"),
            Th("Test Acc"),
            Th("Test Bal. Acc"),
            Th("Params"),
            Th("Plots"),
        )

        rows = []
        modals = []
        for run in all_runs:
            m = run.get("metrics", {})
            run_id = run["run_id"][:8]

            # Hyperparameters modal
            params = run.get("params", {})
            params_cell = "-"
            if params:
                params_modal_id = f"params-{run_id}"
                param_items = [
                    Tr(Td(k, cls=TextPresets.muted_sm), Td(v))
                    for k, v in sorted(params.items())
                ]
                params_cell = A(
                    UkIcon("settings", height=18, width=18),
                    href=f"#{params_modal_id}", uk_toggle="",
                    style=f"color:{_IMMACULATA}; cursor:pointer;",
                    uk_tooltip="title: Hyperparameters",
                )
                modals.append(
                    Modal(
                        Table(
                            Thead(Tr(Th("Parameter"), Th("Value"))),
                            Tbody(*param_items),
                            cls="uk-table uk-table-small uk-table-divider",
                        ),
                        header=f"{run['run_name']}: Hyperparameters",
                        id=params_modal_id,
                    )
                )

            # Confusion matrix modal
            cm_path = run.get("artifacts", {}).get("confusion_matrix")
            roc_path = run.get("artifacts", {}).get("roc_curves")

            plot_links = []
            if cm_path:
                cm_modal_id = f"cm-{run_id}"
                plot_links.append(
                    A(UkIcon("grid", height=18, width=18),
                      href=f"#{cm_modal_id}", uk_toggle="",
                      style=f"color:{_IMMACULATA}; cursor:pointer;",
                      uk_tooltip="title: Confusion Matrix")
                )
                modals.append(
                    Modal(
                        Img(src=f"/mlruns/{cm_path}", alt="Confusion Matrix",
                            style="width:100%; border-radius:6px;"),
                        header=f"{run['run_name']}: Confusion Matrix",
                        id=cm_modal_id,
                        dialog_cls="uk-modal-dialog-large",
                    )
                )

            if roc_path:
                roc_modal_id = f"roc-{run_id}"
                plot_links.append(
                    A(UkIcon("trending-up", height=18, width=18),
                      href=f"#{roc_modal_id}", uk_toggle="",
                      style=f"color:{_IMMACULATA}; cursor:pointer; margin-left:0.5rem;",
                      uk_tooltip="title: ROC Curves")
                )
                modals.append(
                    Modal(
                        Img(src=f"/mlruns/{roc_path}", alt="ROC Curves",
                            style="width:100%; border-radius:6px;"),
                        header=f"{run['run_name']}: ROC Curves",
                        id=roc_modal_id,
                        dialog_cls="uk-modal-dialog-large",
                    )
                )

            rows.append(Tr(
                Td(Strong(run["run_name"])),
                Td(run.get("variant", "-")),
                Td(run.get("n_classes", "-")),
                Td(run.get("n_features", "-")),
                Td(f"{m.get('val_macro_f1', 0):.3f}" if m.get("val_macro_f1") else "-"),
                Td(f"{m.get('test_macro_f1', 0):.3f}" if m.get("test_macro_f1") else "-"),
                Td(f"{m.get('test_accuracy', 0):.1%}" if m.get("test_accuracy") else "-"),
                Td(f"{m.get('test_balanced_accuracy', 0):.1%}" if m.get("test_balanced_accuracy") else "-"),
                Td(params_cell),
                Td(Span(*plot_links) if plot_links else "-"),
            ))

        exp_name = experiments[0].get("experiment_name", "Unknown")
        total = experiments[0].get("total_runs", 0)
        unique = experiments[0].get("unique_runs", 0)

        return Div(
            Card(
                P(f"Experiment: {exp_name} | {total} total runs, {unique} unique",
                  cls=TextPresets.muted_sm),
                P("Deep learning and XGBoost runs tracked via MLflow (7-class task). "
                  "LGBM 3-class production model tracked separately via walk-forward validation, included for comparison.",
                  cls=TextPresets.muted_sm, style="margin-top:0.25rem; font-style:italic;"),
                DividerLine(),
                Table(Thead(header), Tbody(*rows),
                      cls="uk-table uk-table-small uk-table-divider uk-table-hover"),
                header=H4("MLflow Experiments", style=f"color:{_FOUNDERS};"),
            ),
            # All modals (params + plots) rendered at end of component
            *modals,
        )

    except Exception:
        return _fallback("MLflow experiments")


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard doc sections (interactive, rendered inside /docs)
# ═══════════════════════════════════════════════════════════════════════════════

def _dash_strategy_section():
    """Strategy backtesting dashboard as a docs section."""
    return _doc_section("dash-strategy", "Strategy Backtest",
        P("Run historical backtests comparing model-guided strategies against baselines.",
          cls=TextPresets.muted_sm),
        DividerLine(),
        Grid(
            Div(_backtest_results_panel(), cls="col-span-5"),
            Div(_backtest_sidebar(), cls="col-span-2"),
            cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
        ),
    )


def _dash_model_section():
    """Model performance dashboard as a docs section."""
    return _doc_section("dash-model", "Model Performance",
        P("Evaluate prediction accuracy, per-class metrics, and confidence analysis.",
          cls=TextPresets.muted_sm),
        DividerLine(),
        Grid(
            Div(_model_performance_panel(), cls="col-span-4"),
            Div(_model_sidebar(), cls="col-span-3"),
            cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
        ),
    )


def _dash_mlflow_section():
    """MLflow experiment tracking as a docs section."""
    return _doc_section("dash-mlflow", "MLflow Tracking",
        P("Experiment runs across all model architectures and training variants.",
          cls=TextPresets.muted_sm),
        DividerLine(),
        Div(
            Loading(htmx_indicator=True, id="mlflow-spinner"),
            id="mlflow-results",
            hx_get="/mlflow_call",
            hx_trigger="intersect once",
            hx_swap="innerHTML",
            hx_indicator="#mlflow-spinner",
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# docs footer
# ═══════════════════════════════════════════════════════════════════════════════

# docs footer should be minimalistic and send to the documentation screen
def _docs_footer():
    """Minimalistic footer linking to the documentation screen.

    Returns:
        Footer Div.
    """
    return Footer(
        DivCentered(
            A("Documentation & Dashboards ->", href="/docs",
              style=f"color:{_IMMACULATA};"),
            cls="py-4",
        ),
        id="docs-footer",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# full page renderer (trading screen), to be triggered by an @rt call
# ═══════════════════════════════════════════════════════════════════════════════

# documentations screen should follow the same format as the nymo whitepaper (https://nymo.finance/whitepaper/). NOT BY ANY MEANS THE SAME AESTHETIC!
# full .htmx, as (everything on this ui), divided into sections, with a side-bar on the left that allows for navigation between the different sections,
# and section description displayed on the remaining right side

_FIG = "/figures"


def _doc_fig(filename: str, caption: str = "", max_width: str = "560px"):
    """Render a figure as an inline image with caption.

    Args:
        filename: Image filename in reports/figures/.
        caption: Caption text shown below the image.
        max_width: CSS max-width constraint.

    Returns:
        Div with constrained image and optional caption.
    """
    return Div(
        Img(src=f"{_FIG}/{filename}", alt=caption,
            style=f"max-width:{max_width}; width:100%; border-radius:8px; "
                  f"border:1px solid {_TORERO}40;"),
        P(caption, cls=TextPresets.muted_sm,
          style="text-align:center; margin-top:0.5rem; font-style:italic;") if caption else "",
        style="margin:1rem 0; text-align:center;",
    )


def _doc_row(text_el, fig_el):
    """Two-column row: text left, inline figure right."""
    return Div(
        Div(text_el, style="flex:1; min-width:0;"),
        Div(fig_el, style="flex:1; min-width:0; display:flex; align-items:center; justify-content:center;"),
        style=f"display:flex; gap:1.5rem; padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
    )


def _doc_section(sid: str, title: str, *content):
    """Standard docs section wrapper."""
    return Section(
        H3(title, style=f"color:{_FOUNDERS};"),
        *content,
        DividerLine(),
        id=sid,
        cls="mb-6 mt-2",
    )


def _doc_columns(left, right):
    """Two-column layout with a vertical divider. Used for side-by-side comparisons."""
    return Div(
        Div(left, style=f"flex:1; padding-right:1.5rem; border-right:1px solid {_TORERO};"),
        Div(right, style="flex:1; padding-left:1.5rem;"),
        style="display:flex; gap:0; margin-top:0.75rem;",
    )


def _docs_overview():
    """Section 1: Project overview."""
    return _doc_section("doc-overview", "Overview",
        P("This system is a decision-support tool for covered call strategy selection, notan automated "
          "trading signal. It evaluates which strike and maturity combination is most likely to produce the "
          "best risk-adjusted return for a given stock and month, then presents that analysis alongside "
          "strategy scoring, market context, and model limitations so the portfolio manager can make an "
          "informed decision."),
        P("The problem is framed as multi-class classification over predefined strategy buckets. Two parallel "
          "modeling pipelines were developed and are deployed together: a LightGBM tree-based pipeline with "
          "walk-forward validation on a 3-class moneyness target (production, macro F1: 0.47), and an "
          "LSTM-CNN deep learning pipeline on the full 7-class moneyness-maturity space (experimental, "
          "macro F1: 0.11). Both models run in parallel via a LangGraph DAG, and their outputs feed into "
          "a Claude AI analysis node that synthesizes a recommended action report.", cls="mt-2"),
        P("Key finding from our research: model performance is fundamentally constrained by distribution "
          "shift across market regimes, not by model architecture. All models overfit significantly, and "
          "the OTM10 baseline strategy remains hard to beat. This system is most valuable as a diagnostic "
          "tool in regimes where the models show strong agreement and high confidence.", cls="mt-2"),
        Card(
            Div(
                Div(P(Strong("Universe"), cls=TextPresets.muted_sm),
                    P("AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA, WMT")),
                Div(P(Strong("Decision Frequency"), cls=TextPresets.muted_sm),
                    P("Monthly (last trading day of each month)")),
                Div(P(Strong("Moneyness Buckets"), cls=TextPresets.muted_sm),
                    P("ATM (delta 0.45-0.60), OTM5 (0.30-0.45), OTM10 (0.15-0.30)")),
                Div(P(Strong("Maturity Buckets"), cls=TextPresets.muted_sm),
                    P("SHORT (DTE 7-45), LONG (DTE 46-120)")),
                Div(P(Strong("Data Period"), cls=TextPresets.muted_sm),
                    P("2015-2025: daily prices, options chains, quarterly fundamentals")),
                Div(P(Strong("Data Sources"), cls=TextPresets.muted_sm),
                    P("Alpha Vantage API (equities, options, fundamentals) + FRED (macro indicators)")),
                Div(P(Strong("LightGBM Pipeline"), cls=TextPresets.muted_sm),
                    P("3-class moneyness + IV-rank maturity rule. Production model (0.47 walk-forward F1)")),
                Div(P(Strong("Deep Learning Pipeline"), cls=TextPresets.muted_sm),
                    P("7-class LSTM-CNN + PatchTST. Experimental (0.11 best test F1)")),
                style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;",
            ),
            header=H4("System Summary", style=f"color:{_IMMACULATA};"),
            cls="mt-2",
        ),
    )


def _docs_data_pipeline():
    """Section 2: Data sources and cleaning."""
    return _doc_section("doc-data-pipeline", "Data Pipeline",
        P("The modeling dataset integrates multiple sources of financial and market data obtained through the "
          "Alpha Vantage API, spanning 2015 through 2025 for ten publicly traded companies. Raw data is cached "
          "in an S3 mirror for reproducibility."),
        Card(
            Div(
                Div(P(Strong("Daily Prices")), P("52,486 observations -OHLCV, adjusted close, dividends, stock splits", cls=TextPresets.muted_sm)),
                Div(P(Strong("Income Statements")), P("781 quarterly reports -revenue, margins, net income", cls=TextPresets.muted_sm)),
                Div(P(Strong("Balance Sheets")), P("773 quarterly reports -assets, liabilities, equity", cls=TextPresets.muted_sm)),
                Div(P(Strong("Cash Flow")), P("778 quarterly reports -operating cash flow, capex", cls=TextPresets.muted_sm)),
                Div(P(Strong("Options Chains")), P("1.12M cleaned contracts (560K calls) -monthly board snapshots", cls=TextPresets.muted_sm)),
                Div(P(Strong("Company Overview")), P("10 companies -sector, industry, shares outstanding, equity beta", cls=TextPresets.muted_sm)),
                style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;",
            ),
            header=H4("Data Sources", style=f"color:{_IMMACULATA};"),
            cls="mt-2",
        ),
        H4("Cleaning & Preprocessing", style=f"color:{_IMMACULATA}; margin-top:1.5rem;"),
        P("Options data represents monthly snapshots of the full options board: alllisted contracts at all strikes "
          "and expirations, not executed trades. For AAPL this means ~900 call contracts per snapshot; for smaller "
          "names like AVGO, ~200. The cleaned options dataset contains 1.12M contracts, of which 560K are calls "
          "within the relevant strike and maturity ranges (delta 0.10-0.70, DTE 7-150 days)."),
        P("Missing data in financial statements ranged from 27% to 56% depending on the variable, but most "
          "corresponded to fields not applicable for certain companies rather than true gaps. Debt-to-equity was "
          "the only engineered feature with missing values (~1% of observations), imputed via median. Financial "
          "ratios were clipped to economically reasonable ranges: P/E to [-100, 500], FCF yield to [-1, 1]."
          "to remove distortions from near-zero denominators.", cls="mt-2"),
        P("Quarterly fundamentals were joined to daily data via as-of merge, matching on the most recent fiscal "
          "reporting date to prevent lookahead bias. Observations prior to January 2016 were removed to ensure "
          "sufficient history for rolling window initialization (200-day moving averages). The final modeling "
          "dataset contains 1,391 monthly decision points across 10 tickers. The LGBM pipeline uses 34 features "
          "(27 base + 8 IV-derived, after pruning 5 leaky features); the DL pipeline uses the top 35 features "
          "selected by Random Forest importance.", cls="mt-2"),
    )


def _docs_eda():
    """Section 3: Exploratory analysis."""
    return _doc_section("doc-exploratory-analysis", "Exploratory Analysis",
        P("Exploratory data analysis revealed substantial diversity in price behavior, volatility dynamics, "
          "and options characteristics across the ticker universe. These patterns directly inform feature "
          "selection and model design."),
        # Feature correlations
        _doc_row(
            Div(
                H4("Feature Correlations", style=f"color:{_IMMACULATA};"),
                P("Several highly correlated feature pairs were identified: operating margin and net margin "
                  "correlate at ~0.95, 21-day momentum and price/SMA50 ratio at ~0.88, and short-term "
                  "volatility measures (10d and 21d) at ~0.82."),
                P("Volatility features cluster together while fundamentals form a separate block, confirming "
                  "the feature set captures distinct informational dimensions. These correlations informed "
                  "regularization and feature selection to avoid multicollinearity.", cls="mt-2"),
            ),
            _doc_fig("feature_correlations.png", "Feature correlation matrix"),
        ),
        # Bucket boundaries
        _doc_row(
            Div(
                H4("Bucket Boundaries", style=f"color:{_IMMACULATA};"),
                P("Option contracts are segmented into strategy buckets using delta for moneyness "
                  "(ATM: 0.45-0.60, OTM5: 0.30-0.45, OTM10: 0.15-0.30) and DTE for maturity "
                  "(30d: 7-45, 60d: 46-75, 90d: 76-120)."),
                P("Most contracts concentrate in shorter maturities and slightly out-of-the-money strikes, "
                  "which aligns with common covered call implementations that maximize premium income "
                  "while limiting assignment risk.", cls="mt-2"),
            ),
            _doc_fig("bucket_analysis.png", "Delta and DTE distributions with bucket boundaries"),
        ),
        # Label distribution
        _doc_row(
            Div(
                H4("Optimal Bucket Distribution", style=f"color:{_IMMACULATA};"),
                P("The best moneyness bucket (highest realized covered call return) varies by year, ticker, "
                  "and market conditions. ATM dominates across all years and most tickers."),
                P("Class imbalance is significant: ATM accounts for ~45% of labels, requiring "
                  "inverse-frequency class weights during training. WMT and MSFT skew heavily toward ATM; "
                  "NVDA and TSLA show more OTM diversity.", cls="mt-2"),
            ),
            _doc_fig("label_distribution.png", "Best moneyness by year and by ticker"),
        ),
        # Vol regime
        _doc_row(
            Div(
                H4("Volatility Regimes", style=f"color:{_IMMACULATA};"),
                P("The optimal bucket shifts between volatility regimes. ATM dominates in both low and high "
                  "vol environments, but long-dated buckets shrink in high vol as the market prices in "
                  "uncertainty more aggressively."),
                P("This pattern motivates the IV-rank maturity rule used in production: when implied "
                  "volatility rank exceeds 0.5, sell shorter-dated options to capture elevated premium "
                  "quickly; otherwise, sell longer-dated options to collect more time value.", cls="mt-2"),
            ),
            _doc_fig("label_by_vol_regime.png", "Best bucket by volatility regime"),
        ),
        # Temporal coverage
        _doc_row(
            Div(
                H4("Temporal Coverage", style=f"color:{_IMMACULATA};"),
                P("Label distribution evolves over time as more tickers enter the options market. Coverage "
                  "peaks in 2022-2023 with the full universe available. ATM buckets (DTE30/DTE60) dominate "
                  "across all years."),
                P("The dataset captures multiple major market regimes: the post-2016 bull market, the 2020 "
                  "pandemic crash and recovery, the 2022 rate-hike selloff, and the 2023-2024 AI rally. "
                  "This diversity is essential for evaluating model robustness.", cls="mt-2"),
            ),
            _doc_fig("label_temporal.png", "Best bucket distribution over time"),
        ),
    )


def _docs_features():
    """Section 4: Feature engineering -two-column comparison of both pipelines."""
    return _doc_section("doc-feature-engineering", "Feature Engineering",
        P("Both pipelines share the same raw data sources but engineer different feature sets and targets. "
          "All features are computed at monthly decision points from daily price data and quarterly "
          "fundamentals, with merge_asof joins to prevent lookahead bias."),
        _doc_columns(
            # Left: LGBM pipeline
            Div(
                H4("LightGBM Pipeline", style=f"color:{_IMMACULATA};"),
                P(Strong("27 base + 8 IV = 34 features (production)"), cls="mt-1"),
                P(Strong("Technical (18)"), cls="mt-2"),
                P("Volatility (5d/21d/63d), momentum (5d/21d/63d), price-to-SMA (21/50/200), "
                  "SMA crossovers (21>50, 50>200), drawdowns (63d/252d), volume ratio, RSI-14, "
                  "Bollinger width, high-vol regime flag", cls=TextPresets.muted_sm),
                P(Strong("Fundamental (9)"), cls="mt-2"),
                P("Gross/operating/net margin, revenue growth YoY, earnings growth YoY, "
                  "debt-to-equity, cash ratio, ROE, ROA", cls=TextPresets.muted_sm),
                P(Strong("Implied Volatility (8, added in walk-forward)"), cls="mt-2"),
                P("IV mean/median/skew, short/long-term IV, IV term structure, "
                  "IV rank (percentile), IV month-over-month change", cls=TextPresets.muted_sm),
                P(Strong("Target: 3-class moneyness"), cls="mt-3"),
                P("ATM / OTM5 / OTM10 -maturity selected post-hoc by IV-rank rule", cls=TextPresets.muted_sm),
            ),
            # Right: DL pipeline
            Div(
                H4("Deep Learning Pipeline", style=f"color:{_IMMACULATA};"),
                P(Strong("Top 35 features (RF importance-selected)"), cls="mt-1"),
                P(Strong("Price & Returns"), cls="mt-2"),
                P("Open, High, Low, Close, Volume, daily/weekly/monthly returns", cls=TextPresets.muted_sm),
                P(Strong("Technical"), cls="mt-2"),
                P("Rolling volatility (5/20/60d), momentum, RSI, MACD, Bollinger Bands", cls=TextPresets.muted_sm),
                P(Strong("Valuation & Profitability"), cls="mt-2"),
                P("P/E, P/S, EV/EBITDA, Price/Book, margins, ROA, ROE, leverage ratios", cls=TextPresets.muted_sm),
                P(Strong("Input"), cls="mt-2"),
                P("50-day sliding window sequences (35 features x 50 timesteps)", cls=TextPresets.muted_sm),
                P(Strong("Target: 7-class buckets"), cls="mt-3"),
                P("ATM_30, ATM_60, ATM_90, OTM5_30, OTM5_60_90, OTM10_30, OTM10_60_90", cls=TextPresets.muted_sm),
            ),
        ),
        # Feature importance figure
        _doc_row(
            Div(
                H4("Feature Importance Evolution", style=f"color:{_IMMACULATA};"),
                P("Feature importance rankings shifted dramatically across model stages. In baselines, "
                  "raw price-level features (adjusted_close, volume) leaked information and were removed. "
                  "Fundamentals then dominated: debt_to_equity, cash_ratio, gross_margin."),
                P("After adding IV features in the walk-forward stage, implied volatility measures "
                  "(iv_change, iv_rank, iv_mean, iv_std) consistently occupied the top 4 positions -"
                  "confirming that volatility is the central signal for covered call optimization.", cls="mt-2"),
            ),
            _doc_fig("feature_importance_comparison.png", "Feature importance across model stages"),
        ),
        # Label construction
        Div(
            H4("Label Construction", style=f"color:{_IMMACULATA};"),
            P("Labels are constructed by computing realized covered call payoffs for every contract in each "
              "bucket. For each (ticker, month), the bucket with the highest realized return becomes the "
              "ground truth label. Payoff = premium received + capped stock P&L (capped at strike if assigned)."),
            P("3-class distribution: ATM 62%, OTM5 17%, OTM10 21% (3.6x imbalance ratio). "
              "Class weights computed via inverse frequency: ATM 0.54, OTM5 1.94, OTM10 1.59.", cls="mt-2"),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
    )


def _docs_lgbm_pipeline():
    """Section 5: Tree-based pipeline -3-class moneyness, RF → XGB → LGBM → walk-forward."""
    return _doc_section("doc-lgbm-pipeline", "Tree-Based Pipeline (3-Class)",
        P("This pipeline reformulates the problem as a simpler 3-class moneyness prediction "
          "(ATM / OTM5 / OTM10), decoupling maturity selection into a post-hoc IV-rank rule. "
          "Three tree-based models were evaluated (Random Forest, XGBoost, and LightGBM), each"
          "progressively refined through Optuna tuning and walk-forward validation. "
          "LightGBM with walk-forward annual retraining emerged as the production model."),
        # Baselines
        _doc_row(
            Div(
                H4("Random Forest & XGBoost Baselines", style=f"color:{_IMMACULATA};"),
                P("Both models trained on 27 features with 80/20 time-based split "
                  "(21,395 train / 7,878 test rows)."),
                P("Random Forest: 48.5% accuracy / 0.333 macro F1. Heavily biased toward ATM -"
                  "misclassifies 1,778 OTM10 samples as ATM. Only 45 correct OTM5 predictions "
                  "out of 2,266 actual OTM5 instances.", cls="mt-2"),
                P("XGBoost: 48.0% accuracy / 0.359 macro F1. Slightly better OTM5 recall (198 correct) "
                  "but still dominated by ATM predictions. Top features: debt_to_equity, cash_ratio, "
                  "gross_margin -fundamentals dominate after pruning leaky price-level features.", cls="mt-2"),
            ),
            _doc_fig("baseline_confusion.png", "RF and XGB confusion matrices, both default to ATM"),
        ),
        # Optuna-tuned
        _doc_row(
            Div(
                H4("Optuna Hyperparameter Tuning", style=f"color:{_IMMACULATA};"),
                P("All three models tuned via Optuna (20 trials each) with TimeSeriesSplit "
                  "cross-validation and balanced class weights to address ATM dominance."),
                P("RF: 50.0% / 0.338 F1. XGB: 50.4% / 0.342. LGBM: 48.4% / 0.349. "
                  "Marginal improvement: allnear random-level F1 (0.333). The limitation "
                  "at this stage was the feature set, not the model architecture.", cls="mt-2"),
                P("LGBM achieved highest macro F1 despite lowest accuracy, with betterminority-class "
                  "recall. Key params: n_estimators=300, max_depth=8, lr=0.05, num_leaves=50. "
                  "Top features shifted to volatility: vol_63d, vol_21d, bb_width.", cls="mt-2"),
            ),
            _doc_fig("improved_confusion_matrices.png", "Tuned RF, XGB, and LGBM. LGBM best minority-class balance"),
        ),
        # Walk-forward (production)
        _doc_row(
            Div(
                H4("Walk-Forward Validation (Production)", style=f"color:{_IMMACULATA};"),
                P("The breakthrough came from walk-forward annual retraining: for each year N, "
                  "train on all data from years 1 to N-1, predict year N. This expanding-window "
                  "approach eliminates future leakage and simulates deployment conditions."),
                P("Eight IV features were added at this stage (IV mean, median, skew, short/long-term IV, "
                  "term structure, IV rank, IV change), computed from the full options board. These "
                  "immediately dominated feature importance, pushing total features from 27 to 34.", cls="mt-2"),
                P("Overall macro F1 = 0.468, asignificant jump from the tuned model's 0.349. "
                  "Strong ATM recall (13,472 correct), OTM10 reasonably separated (1,850 of 4,557). "
                  "OTM5 remains the hardest class.", cls="mt-2"),
            ),
            _doc_fig("walkforward_confusion_matrix.png", "Walk-forward confusion matrix. IV features drive F1 from 0.35 to 0.47"),
        ),
        _doc_row(
            Div(
                H4("Yearly Performance", style=f"color:{_IMMACULATA};"),
                P("Walk-forward F1 varies by year, peakingat 0.60 in 2019, sustained above 0.50 "
                  "from 2015-2021, dips to ~0.32 in 2023 (regime-shift year with rapid rate hikes)."),
                P("The model beats the random baseline (0.333) in most years. IV features (iv_change, "
                  "iv_rank, iv_mean, iv_std) consistently occupy the top 4 importance positions across "
                  "all walk-forward folds.", cls="mt-2"),
            ),
            _doc_fig("walkforward_yearly_f1.png", "Macro F1 by year, consistently above random baseline"),
        ),
        # LSTM 3-class
        _doc_row(
            Div(
                H4("LSTM on 3-Class (Explored)", style=f"color:{_IMMACULATA};"),
                P("A bidirectional LSTM with temporal attention was evaluated on the same 3-class target "
                  "using 60-day lookback sequences and walk-forward annual retraining."),
                P("Overall macro F1 = 0.411. Competitive in some years (2017, 2020) but less stable -"
                  "degrades sharply in 2022-2025 where LGBM maintains consistency. LGBM selected for "
                  "production: better stability, sub-millisecond inference, simpler deployment.", cls="mt-2"),
            ),
            _doc_fig("lstm_vs_lgbm_walkforward.png", "LSTM vs LGBM walk-forward F1 by year"),
        ),
        # Maturity rule
        Div(
            H4("Maturity Selection Rule", style=f"color:{_IMMACULATA};"),
            P("Rather than predicting maturity jointly (requiring 6+ classes and exacerbating data "
              "scarcity), the production system uses a deterministic IV-rank rule:"),
            P(Strong("IV rank > 0.5 → sell SHORT"), " (capture elevated premium quickly). ",
              Strong("IV rank ≤ 0.5 → sell LONG"), " (collect more time value in calm markets).", cls="mt-1"),
            P("This decoupled approach avoids the curse of dimensionality that limited the 7-class "
              "models while still adapting maturity to current market conditions.", cls="mt-2"),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
    )


def _docs_dl_pipeline():
    """Section 6: Deep learning pipeline -XGBoost baseline, LSTM-CNN, PatchTST on 7-class."""
    return _doc_section("doc-dl-pipeline", "Deep Learning Pipeline (7-Class)",
        P("This pipeline tackles the full 7-class moneyness-maturity target, predicting both strike "
          "and expiry jointly. The 7 classes are: ATM_30, ATM_60, ATM_90, OTM5_30, OTM5_60_90, "
          "OTM10_30, OTM10_60_90. Models process 50-day sliding window sequences of the top 35 "
          "features (selected by Random Forest importance). Train/val/test: pre-2022 / 2022-2023 / 2024+."),
        P("Severe class imbalance is the defining challenge: ATM_30 represents 35.4% of samples "
          "while OTM10_60_90 and OTM5_60_90 each represent ~1%. Partial resampling and class-weighted "
          "loss functions were applied, but the imbalance fundamentally limits 7-class performance.", cls="mt-2"),
        # XGBoost baseline
        _doc_row(
            Div(
                H4("XGBoost Baseline (7-Class)", style=f"color:{_IMMACULATA};"),
                P("XGBoost with Optuna tuning, partial class resampling, and multi-class softmax. "
                  "225 estimators, max_depth=3, lr=0.049, early stopping at 50 rounds."),
                P("Train accuracy: 82.5% / Test accuracy: 33.3% / Test macro F1: 0.142. "
                  "Massive overfitting gap. Predictions concentrate on dominant buckets (ATM_30, ATM_60); "
                  "minority classes (OTM10_60_90, OTM5_30) receive near-zero F1 on the test set.", cls="mt-2"),
                P("This established the fundamental difficulty of 7-class granularity and motivated "
                  "exploration of sequence-based deep learning architectures.", cls="mt-2"),
            ),
            _doc_fig("model_comparison.png", "7-class model comparison (XGBoost baseline)"),
        ),
        # LSTM-CNN architecture
        _doc_row(
            Div(
                H4("LSTM-CNN + Bahdanau Attention", style=f"color:{_IMMACULATA};"),
                P("Hybrid architecture with two parallel branches:"),
                P(Strong("CNN Branch"), ": Two-layer 1D convolution (35→128, kernel=7) with BatchNorm "
                  "and AdaptiveAvgPool. Captures short-range patterns: momentum shifts, vol spikes.", cls="mt-1"),
                P(Strong("BiLSTM Branch"), ": Bidirectional LSTM (2 layers, hidden=128) with Bahdanau "
                  "temporal attention. Captures long-range dependencies: trend reversals, regime shifts.", cls="mt-1"),
                P(Strong("Fusion"), ": CNN (128d) + LSTM attention (256d) concatenated → LayerNorm → "
                  "FC(384→192→7). Combined local and global temporal context.", cls="mt-1"),
                P("Trained with AdamW, class-weighted cross-entropy loss, early stopping on "
                  "validation macro F1. Three training variants with different regularization.", cls="mt-2"),
            ),
            _doc_fig("lstm_training_curves.png", "LSTM-CNN training curves showing early divergence of train/val loss"),
        ),
        # LSTM-CNN results
        _doc_row(
            Div(
                H4("LSTM-CNN Results", style=f"color:{_IMMACULATA};"),
                P("20 experiment runs tracked in MLflow across three variants:"),
                P(Strong("Regularised"), " (dropout=0.5, weight_decay=0.01): 38.1% accuracy / 0.110 F1 -"
                  "highest test F1 of any deep learning model. Heavy regularization improved "
                  "generalization.", cls="mt-1"),
                P(Strong("Best"), " (dropout=0.155): 24.5% accuracy / 0.091 F1. Lower dropout "
                  "led to overfitting. Training F1 reached 0.819 while validation peaked at 0.206.", cls="mt-1"),
                P("Four of seven classes received F1 scores of zero across most models, indicating "
                  "a fundamental failure to distinguish minority strategy buckets. The OTM10_60_90 class "
                  "shifted from 1.25% of training data to 53.15% of the 2024 test set.", cls="mt-2"),
            ),
            _doc_fig("comparison_confusion_matrices.png", "LSTM-CNN confusion matrices across training variants"),
        ),
        # PatchTST
        _doc_row(
            Div(
                H4("PatchTST Transformer", style=f"color:{_IMMACULATA};"),
                P("Patch-based time series transformer (HuggingFace) using 100-day sliding windows. "
                  "4 encoder layers, embedding dim=64, 2 attention heads. Walk-forward annual retraining."),
                P("Base: 14.4% accuracy / 0.086 F1. Pretrained + FRED macro variant: no improvement. "
                  "Transformers require substantially more data than the ~1,300 available monthly "
                  "decision points -self-attention overfits to spurious temporal correlations.", cls="mt-2"),
            ),
            _doc_fig("patchtst_walkforward_confusion_matrix.png", "PatchTST walk-forward confusion matrix"),
        ),
        _doc_row(
            Div(
                H4("PatchTST Yearly Stability", style=f"color:{_IMMACULATA};"),
                P("F1 by year shows high variance, oscillating between near-random and moderately "
                  "above baseline. The model cannot maintain stability across different market regimes."),
                P("Key finding from the report: increasing model complexity did not improve performance. "
                  "The primary limitation is data volume and regime sensitivity, not model capacity.", cls="mt-2"),
            ),
            _doc_fig("patchtst_walkforward_yearly_f1.png", "PatchTST F1 by year, high variance and unstable"),
        ),
        # Deployment
        Div(
            H4("Docker Deployment", style=f"color:{_IMMACULATA};"),
            P("The LSTM-CNN model is independently deployed via Docker Compose on AWS EC2 with three "
              "containerized services: Streamlit frontend (:8501) for batch CSV prediction, FastAPI "
              "inference server (:8000) with /predict endpoints, and MLflow tracking server (:5000) "
              "for model registry and experiment management."),
            P("Model loading follows a priority chain: MLflow Registry "
              "(models:/CoveredCallLSTMCNN/Champion) → local fallback (lstm_cnn_best_model.pth). "
              "Infrastructure provisioned via Terraform on t3.medium.", cls="mt-2"),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
    )


def _docs_strategy():
    """Section 7: Post-inference strategy -scoring engine, allocation, backtesting."""
    return _doc_section("doc-strategy", "Strategy & Post-Inference",
        P("The model predicts a moneyness bucket, but a prediction alone is not actionable. The deployed "
          "system adds a scoring and allocation layer between model output and the portfolio manager's "
          "decision. This layer, developed post-capstone, ranks the 10-ticker universe each month, "
          "evaluates each position against multiple scoring criteria, and presents the results alongside "
          "Claude AI analysis as a decision-support diagnostic."),
        # Scoring engine
        Div(
            H4("Composable Scoring Engine", style=f"color:{_IMMACULATA};"),
            P("Each ticker receives a composite score from three weighted components:"),
            P(Strong("1. Model Confidence"), ": the LGBM prediction probability for the chosen bucket. "
              "Higher confidence means the model sees a clearer signal for this ticker-month.", cls="mt-1"),
            P(Strong("2. Transaction Cost Score"), ": computed from the bid-ask spread of matching "
              "options contracts. Tighter spreads = lower cost = higher score. A turnover penalty "
              "doubles the cost if the bucket changed from the prior month, discouraging churn.", cls="mt-1"),
            P(Strong("3. Delta-Hedged Return Score"), ": monthly approximation inspired by "
              "Bali et al. (2021). Isolates the volatility premium by removing directional exposure: "
              "DH_gain = option_pnl - delta * stock_move - financing_cost. Higher values indicate "
              "more pure vol premium available.", cls="mt-1"),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
        # Presets
        Div(
            H4("Strategy Presets", style=f"color:{_IMMACULATA};"),
            P("Three model-guided presets control the weight given to each scoring component, "
              "the number of positions taken, and how capital is sized. Two model-only strategies "
              "(Argmax and Risk-Adjusted) and one no-model baseline complete the comparison."),
            _doc_columns(
                Div(
                    P(Strong("Conservative (Scored)"), cls="mt-1"),
                    P("Confidence 30% / TC 50% / Delta-Hedge 20%", cls=TextPresets.muted_sm),
                    P("7 positions, equal weight", cls=TextPresets.muted_sm),
                    P("Spread capital wide, prioritize low-cost trades. Run with both LGBM and LSTM predictions.", cls="mt-1"),
                    # Balanced and Aggressive removed -Conservative is the production preset
                ),
                Div(
                    P(Strong("Argmax (No Scoring)"), cls="mt-1"),
                    P("Model's top prediction per ticker, all 10 tickers traded, equal weight.", cls=TextPresets.muted_sm),
                    P("Pure model signal with no filtering.", cls="mt-1"),
                    P(Strong("Risk-Adjusted (No Scoring)"), cls="mt-3"),
                    P("P(bucket) x E[return | bucket] using expanding historical averages.", cls=TextPresets.muted_sm),
                    P("Probability-weighted expected return.", cls="mt-1"),
                    P(Strong("Baseline (No Model)"), cls="mt-3"),
                    P("Always sell 10% OTM short-dated on all tickers, equal weight.", cls=TextPresets.muted_sm),
                    P("No model, no scoring. Purebenchmark.", cls="mt-1"),
                ),
            ),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
        # Backtesting results -these figures cover Argmax, Risk-Adjusted, Baseline, and Oracle only.
        # The 3 scoring presets (Conservative, Balanced, Aggressive) are computed live in the
        # Strategy tab of the trading dashboard and are not included in these static figures.
        _doc_row(
            Div(
                H4("Annual Returns", style=f"color:{_IMMACULATA};"),
                P("Annual returns for the non-scored strategies: Argmax (model top pick), "
                  "Risk-Adjusted (probability-weighted expected return), and OTM10 Baseline "
                  "(no model). Oracle (perfect foresight) included as an upper bound."),
                P("The scored presets (Conservative, Balanced, Aggressive) are not shown here -"
                  "their results are computed live in the Strategy tab of the trading dashboard.", cls="mt-2"),
            ),
            _doc_fig("annual_returns_comparison.png", "Annual returns: Argmax, Risk-Adjusted, Baseline, Oracle"),
        ),
        _doc_row(
            Div(
                H4("Equity Curves", style=f"color:{_IMMACULATA};"),
                P("Cumulative portfolio growth for Argmax, Risk-Adjusted, and static baselines. "
                  "Model-Argmax and Risk-Adjusted deliver similar trajectories, diverging most "
                  "in volatile periods (2020, 2022) where bucket selection matters most."),
                P("Risk-Adjusted achieves the best Sharpe ratio among these strategies. "
                  "The three scoring presets are available in the live dashboard.", cls="mt-2"),
            ),
            _doc_fig("equity_curves.png", "Equity curves: Argmax, Risk-Adjusted, Baseline"),
        ),
    )


def _docs_results():
    """Section 8: Side-by-side model results comparison of both pipelines."""
    return _doc_section("doc-results", "Results & Limitations",
        P("Both pipelines were evaluated on held-out test data with honest temporal splits. "
          "Direct F1 comparison across pipelines is misleading (3-class vs 7-class), but "
          "within each pipeline the progression from baseline to final model is clear. "
          "These results inform how much trust to place in each model's diagnostic output."),
        # Side-by-side metrics
        _doc_columns(
            # Left: Tree-based pipeline results
            Div(
                H4("Tree-Based (3-Class)", style=f"color:{_IMMACULATA};"),
                P(Strong("Production: LightGBM Walk-Forward")),
                Div(
                    P(Strong("Macro F1"), cls=TextPresets.muted_sm), P("0.468"),
                    P(Strong("Test Accuracy"), cls=TextPresets.muted_sm), P("63.0% (2025 held-out)"),
                    P(Strong("Top-2 Accuracy"), cls=TextPresets.muted_sm), P("85.0%"),
                    P(Strong("Validation"), cls=TextPresets.muted_sm), P("Walk-forward annual"),
                    cls="mt-2",
                ),
                P(Strong("Progression:"), cls="mt-3"),
                P("RF Baseline → 0.333 F1", cls=TextPresets.muted_sm),
                P("XGB Baseline → 0.359 F1", cls=TextPresets.muted_sm),
                P("LGBM Tuned → 0.349 F1", cls=TextPresets.muted_sm),
                P("LGBM + IV + Walk-Forward → 0.468 F1", cls=TextPresets.muted_sm),
                P("IV features + walk-forward provided the largest jump "
                  "(+0.12 F1). Simplifying to 3 classes made the problem tractable.", cls="mt-2"),
            ),
            # Right: DL pipeline results
            Div(
                H4("Deep Learning (7-Class)", style=f"color:{_IMMACULATA};"),
                P(Strong("Best: LSTM-CNN Regularised")),
                Div(
                    P(Strong("Macro F1"), cls=TextPresets.muted_sm), P("0.110"),
                    P(Strong("Test Accuracy"), cls=TextPresets.muted_sm), P("38.1%"),
                    P(Strong("MLflow Runs"), cls=TextPresets.muted_sm), P("20 tracked experiments"),
                    P(Strong("Validation"), cls=TextPresets.muted_sm), P("Time-based 80/20"),
                    cls="mt-2",
                ),
                P(Strong("Progression:"), cls="mt-3"),
                P("XGB 7-Class → 0.142 F1", cls=TextPresets.muted_sm),
                P("LSTM-CNN Best → 0.091 F1", cls=TextPresets.muted_sm),
                P("LSTM-CNN Regularised → 0.110 F1", cls=TextPresets.muted_sm),
                P("PatchTST → 0.086 F1", cls=TextPresets.muted_sm),
                P("7-class joint prediction is fundamentally harder. "
                  "~1,300 monthly points limits what sequence models can learn.", cls="mt-2"),
            ),
        ),
        # Key findings
        Div(
            H4("Key Findings", style=f"color:{_IMMACULATA};"),
            P("All models outperform random baseline, confirming meaningful predictive signal in the "
              "feature set. The best 7-class model (LSTM-CNN regularised) achieved 34.0% test accuracy "
              "- a 2.4x improvement over random selection (14.3%)."),
            P("Performance is fundamentally constrained by distribution shift: the OTM10_60_90 class "
              "increased from 1.25% of training data to 53.15% of the 2024 test set. Increasing model "
              "complexity did not help. The limitation is data volume and regime sensitivity, "
              "not architecture.", cls="mt-2"),
            P("Transformer-based models underperformed simpler tree-based methods, confirming that "
              "data-hungry architectures are not appropriate for this dataset size. Macroeconomic "
              "features (FRED) did not contribute meaningfully; stock-level technical and fundamental "
              "features -especially implied volatility measures -proved most informative.", cls="mt-2"),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
        # Decision-support implications
        Div(
            H4("Implications for Decision Support", style=f"color:{_IMMACULATA};"),
            P("These results define the boundaries of what this system can and cannot do. The LGBM model "
              "provides a meaningful signal (F1: 0.47) that is most reliable when: (1) model confidence is "
              "high and both models agree, (2) current market conditions resemble the training data, and "
              "(3) the prediction is for a frequently observed bucket (ATM, OTM10)."),
            P("The system should NOT be used for: (1) automated execution without human review, "
              "(2) high-confidence trading in regimes the model hasn't seen (post-2025 data), or "
              "(3) minority-class predictions (OTM5) where model recall is weakest. The OTM10 baseline "
              "strategy remains the safest default when model signals are ambiguous.", cls="mt-2"),
            style=f"padding:1rem 0; border-bottom:1px solid {_TORERO}40;",
        ),
    )


def _docs_navbar(sections):
    """Top navigation bar for docs with PDF download and Trader link."""
    return NavBar(
        A("Advisor", href="/trading"),
        A(DivLAligned(UkIcon("download", height=16, width=16), Span("PDF Report", style="margin-left:0.3rem;")),
          href="/static/AAI-590-G6-Capstone-Report.pdf", download=True,
          style=f"color:{_IMMACULATA}; text-decoration:none;"),
        brand=A(
            DivLAligned(
                UkIcon("book-open", height=20, width=20),
                H3("USD Strategy Advisor", style=f"color:{_FOUNDERS}; margin:0;"),
                Span("| Docs", style=f"color:{_IMMACULATA}; font-size:0.85rem; margin-left:0.5rem;"),
            ),
            href="/",
        ),
        sticky=True,
        cls="p-3",
        style=f"border-bottom:2px solid {_IMMACULATA};",
    )


# Report sections (static project documentation)
_REPORT_SECTIONS = [
    ("doc-overview", "Overview"),
    ("doc-data-pipeline", "Data Pipeline"),
    ("doc-exploratory-analysis", "Exploratory Analysis"),
    ("doc-feature-engineering", "Features"),
    ("doc-lgbm-pipeline", "Tree-Based"),
    ("doc-dl-pipeline", "Deep Learning"),
    ("doc-results", "Results"),
    ("doc-strategy", "Strategy"),
]

# Dashboard sections (interactive evaluation tools)
_DASHBOARD_SECTIONS = [
    ("dash-strategy", "Strategy Backtest"),
    ("dash-model", "Model Performance"),
    ("dash-mlflow", "MLflow Tracking"),
]

# Combined for routing -sidebar renders the divider separately
DOC_SECTIONS = _REPORT_SECTIONS + _DASHBOARD_SECTIONS

_DOC_RENDERERS = {
    "doc-overview": _docs_overview,
    "doc-data-pipeline": _docs_data_pipeline,
    "doc-exploratory-analysis": _docs_eda,
    "doc-feature-engineering": _docs_features,
    "doc-lgbm-pipeline": _docs_lgbm_pipeline,
    "doc-dl-pipeline": _docs_dl_pipeline,
    "doc-results": _docs_results,
    "doc-strategy": _docs_strategy,
    "dash-strategy": lambda: _dash_strategy_section(),
    "dash-model": lambda: _dash_model_section(),
    "dash-mlflow": lambda: _dash_mlflow_section(),
}


def render_doc_section(section_id: str):
    """Render a single docs section by ID. Called by the /docs/section route.

    Args:
        section_id: One of the DOC_SECTIONS keys.

    Returns:
        Section Div, or fallback if unknown.
    """
    renderer = _DOC_RENDERERS.get(section_id)
    if renderer:
        try:
            return renderer()
        except Exception:
            return _fallback(section_id)
    return _fallback("unknown section")


def _docs_sidebar(sections, active_id: str = "doc-overview"):
    """Sticky left sidebar with htmx-powered section links.

    Report sections and dashboard sections are separated by a labeled divider.

    Args:
        sections: List of (section_id, display_name) tuples.
        active_id: Currently active section ID.
    """
    def _link(sid, name):
        is_active = sid == active_id
        style = (
            f"color:white; text-decoration:none; display:block; padding:0.5rem 0.75rem; "
            f"border-radius:4px; font-size:0.9rem; background:{_IMMACULATA};"
            if is_active else
            f"color:{_FOUNDERS}; text-decoration:none; display:block; padding:0.5rem 0.75rem; "
            f"border-radius:4px; font-size:0.9rem;"
        )
        return Li(A(name,
                    hx_get=f"/docs/section?id={sid}",
                    hx_target="#docs-content",
                    hx_swap="innerHTML",
                    hx_push_url=f"/docs?section={sid}",
                    style=style))

    report_ids = {s[0] for s in _REPORT_SECTIONS}
    report_items = [_link(s, n) for s, n in sections if s in report_ids]
    dash_items = [_link(s, n) for s, n in sections if s not in report_ids]

    nav_items = report_items
    if dash_items:
        nav_items.append(Li(
            Div(
                Hr(style=f"border-color:{_TORERO}40; margin:0.5rem 0;"),
                P("Dashboards", style=f"color:{_IMMACULATA}; font-size:0.7rem; "
                  "font-weight:700; text-transform:uppercase; letter-spacing:0.05em; "
                  "margin:0.25rem 0 0.25rem 0.75rem;"),
            ),
        ))
        nav_items.extend(dash_items)

    return Div(
        Ul(
            *nav_items,
            cls="uk-nav uk-nav-default",
            style="list-style:none; padding:0;",
        ),
        style=f"position:sticky; top:80px; padding-top:1rem; "
              f"border-right:1px solid {_TORERO}40; padding-right:1rem; min-height:300px;",
    )


def docs_section_response(section_id: str):
    """Return section content + updated sidebar (oob swap) for htmx.

    The sidebar is re-rendered with the new active state and swapped
    out-of-band so the highlight follows the user's clicks.

    Args:
        section_id: The section to render.

    Returns:
        Tuple of (section content Div, sidebar oob Div).
    """
    content = render_doc_section(section_id)
    sidebar = Div(
        _docs_sidebar(DOC_SECTIONS, active_id=section_id),
        style="flex:0 0 200px; min-width:180px;",
        id="docs-sidebar",
        hx_swap_oob="true",
    )
    return content, sidebar


def docs_screen(section: str = "doc-overview"):
    """Documentation screen: top navbar + sticky sidebar + single section content.

    Only one section is rendered at a time. Sidebar links use htmx to swap
    the content panel without full page reload.

    Args:
        section: Initial section to display.

    Returns:
        Div with docs layout.
    """
    try:
        initial_content = render_doc_section(section)
        return Div(
            _docs_navbar(DOC_SECTIONS),
            Div(
                # Left sidebar (sticky, swapped oob on section change)
                Div(
                    _docs_sidebar(DOC_SECTIONS, active_id=section),
                    style="flex:0 0 200px; min-width:180px;",
                    id="docs-sidebar",
                ),
                # Right content (single section, swapped via htmx)
                Div(
                    initial_content,
                    style="flex:1; min-width:0;",
                    id="docs-content",
                ),
                style="display:flex; gap:1.5rem;",
                cls="px-4 py-4",
            ),
        )
    except Exception:
        return _fallback("documentation")
