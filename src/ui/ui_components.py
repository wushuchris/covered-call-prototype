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
                H2("USD Trader", style=f"color:{_FOUNDERS};"),
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
                _backtesting_section(),
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
        A("Historical Inference", href="#daily-inference"),
        A("Backtesting", href="#backtesting"),
        A("Docs", href="/docs"),
        brand=A(
            DivLAligned(
                UkIcon("brain-circuit", height=20, width=20),
                H3("USD Trader", style=f"color:{_FOUNDERS}; margin:0;"),
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
        H3("Historical Inference", style=f"color:{_FOUNDERS};"),
        P("Run a historical model prediction for a given ticker and date.", cls=TextPresets.muted_sm),
        DividerLine(),
        Grid(
            # results panel — left (spans 4 of 7 cols)
            Div(_inference_results_panel(), cls="col-span-4"),
            # sidebar — right (spans 3 of 7 cols)
            Div(_inference_sidebar(), cls="col-span-3"),
            cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
        ),
        id="daily-inference",
    )


def _inference_sidebar():
    """Sidebar with date picker, today checkbox, ticker dropdown, and compute button.

    Returns:
        Card with form controls.
    """
    return Card(
        # date input
        Label("Date"),
        Input(type="date", name="date", id="inference-date"),
        # today checkbox — pure htmx, no JS. hx_get swaps the date input for today's value
        Div(
            Label(
                CheckboxX(id="today-check",
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
        ),
        # compute button — posts to /inference_call, swaps results panel
        Button("Compute Inference",
               cls="w-full mt-4",
               style=f"background-color:{_IMMACULATA}; color:white;",
               hx_post="/inference_call",
               hx_include="#inference-date, #inference-ticker",
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
        "Ticker": "—", "Date": "—", "Month": "—", "Model": "—",
        "Prediction": "—", "Confidence": "—",
        "Baseline": "—", "Sample": "—",
    }
    header = ["Metric", "Value"]
    body = [{"Metric": k, "Value": v} for k, v in blank_data.items()]
    return Div(
        Card(
            Div(
                # table side — left ~40%
                Div(
                    TableFromDicts(header_data=header, body_data=body),
                    style="flex:2;",
                ),
                # chart side — right ~60%
                Div(
                    P("Chart placeholder", cls=TextPresets.muted_sm),
                    style=f"flex:3; min-height:200px; border:1px dashed {_TORERO}; border-radius:8px; "
                          "display:flex; align-items:center; justify-content:center;",
                ),
                style="display:flex; gap:1rem;",
            ),
            header=H4("Inference Results", style=f"color:{_FOUNDERS};"),
        ),
        id="inference-results",
    )


def inference_results_card(data: dict):
    """Render inference results as a stats table + candlestick chart.

    The chart data comes as OHLC dicts from the inference service and
    is rendered via MonsterUI's ApexChart component (candlestick type).

    Args:
        data: Dict with model prediction fields and chart_data.

    Returns:
        Card Div with results table and candlestick chart.
    """
    try:
        # Daily inference display — only model output, no ground truth
        display = {
            "Ticker": data.get("ticker", "—"),
            "Date": data.get("date", "—"),
            "Month": data.get("month", "—"),
            "Model": "LGBM 3-Class Moneyness",
            "Prediction": data.get("model_bucket", "—"),
            "Confidence": f"{data.get('model_confidence', 0):.1%}",
            "Baseline": data.get("baseline", "—"),
            "Sample": data.get("sample_type", "—"),
        }
        header = ["Metric", "Value"]
        body = [{"Metric": k, "Value": v} for k, v in display.items()]

        # OHLC data from inference service → ApexCharts candlestick
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

        # Warn if month was snapped to nearest available
        snap_warning = None
        if data.get("snapped"):
            snap_warning = Div(
                Toast(
                    f"No data for requested month — showing nearest available: {data.get('month', '?')}",
                    cls=ToastHT.end, alert_cls=AlertT.warning, dur=5.0,
                ),
                style="position:relative; z-index:9999;",
            )

        return Div(
            snap_warning if snap_warning else "",
            Card(
                Div(
                    # table side — left ~40%
                    Div(
                        TableFromDicts(header_data=header, body_data=body),
                        style="flex:2;",
                    ),
                    # chart side — right ~60%, rendered via ApexChart component
                    Div(
                        chart_el,
                        style="flex:3; min-height:200px;",
                    ),
                    style="display:flex; gap:1rem;",
                ),
                header=H4(f"Inference — {data.get('ticker', '?')} @ {data.get('date', '?')}",
                           style=f"color:{_FOUNDERS};"),
            ),
        )
    except Exception:
        return _fallback("inference results")


# ═══════════════════════════════════════════════════════════════════════════════
# backtesting dashboards
# ═══════════════════════════════════════════════════════════════════════════════

# the backtesting dashboards section should be based upon the MonsterUI dashboard example (https://monsterui.answer.ai/dashboard/)
# and allow the user to understand (four different sections):

# overall system performance (in terms of profit and risk), per stock performance (dropdown), baseline statistics
# (what calls the system made the most, etc), baseline statistics in terms of the EDA

def _backtesting_section():
    """Backtesting dashboards placeholder.

    Four subsections: overall performance, per-stock, baseline stats, EDA stats.

    Returns:
        Section Div.
    """
    return Section(
        H3("Backtesting Dashboards", style=f"color:{_FOUNDERS};"),
        P("Historical strategy performance and analytics.", cls=TextPresets.muted_sm),
        DividerLine(),
        Grid(
            _dashboard_card("Overall Performance",
                            "System-wide profit & risk metrics."),
            _dashboard_card("Per-Stock Performance",
                            "Select a ticker to view individual results."),
            _dashboard_card("Baseline Statistics",
                            "Most frequent calls, assignment rates."),
            _dashboard_card("EDA Overview",
                            "Feature distributions and data quality."),
            cols_min=1, cols_md=2, cols_max=2,
        ),
        id="backtesting",
    )


def _dashboard_card(title: str, description: str):
    """Individual dashboard placeholder card.

    Args:
        title: Card heading.
        description: Short description.

    Returns:
        Card with styled header.
    """
    return Card(
        P(description, cls=TextPresets.muted_sm),
        P("Coming soon.", cls=(TextT.sm, TextT.italic)),
        header=H4(title, style=f"color:{_IMMACULATA};"),
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
            A("Documentation ->", href="/docs",
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


def _doc_fig(filename: str, caption: str = ""):
    """Render a figure as a clickable modal trigger (Lucide icon + caption).

    Clicking opens a full-size modal with the image.
    """
    modal_id = f"fig-{filename.replace('.', '-').replace('_', '-')}"
    return Div(
        # Trigger — icon + caption text
        A(
            Div(
                UkIcon("search", height=28, width=28),
                P(caption, cls=(TextT.sm,), style="margin-top:0.25rem;") if caption else "",
                style="display:flex; flex-direction:column; align-items:center; text-align:center;",
            ),
            href=f"#{modal_id}",
            uk_toggle="",
            style=f"cursor:pointer; color:{_IMMACULATA}; text-decoration:none;",
        ),
        # Modal — full-size image
        Modal(
            Img(src=f"{_FIG}/{filename}", alt=caption, style="width:100%; border-radius:6px;"),
            header=caption if caption else filename,
            id=modal_id,
            dialog_cls="uk-modal-dialog-large",
        ),
    )


def _doc_row(text_el, fig_el):
    """Two-column row: text left, modal icon right. Horizontal divider below."""
    return Div(
        Div(
            Div(text_el, style="flex:1;"),
            Div(fig_el, style="flex:1; display:flex; align-items:center; justify-content:center;"),
            style="display:flex; gap:1.5rem;",
        ),
        style=f"border-bottom:1px solid {_TORERO}; padding-bottom:0.75rem; margin-top:0.75rem;",
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


def _docs_overview():
    """Section 1: Project overview."""
    return _doc_section("doc-overview", "Overview",
        P("This system uses machine learning to optimize covered call option selling across a universe of 10 large-cap "
          "US equities. Given a stock and a monthly decision date, the model predicts which moneyness bucket "
          "(ATM, OTM5%, or OTM10%) is most likely to yield the highest realized covered call return."),
        P("A volatility-based rule then selects maturity: sell short-dated when IV is elevated (capture premium quickly), "
          "sell long-dated when IV is low (collect more time value). The combination produces a 6-bucket action space.", cls="mt-2"),
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
                Div(P(Strong("Production Model"), cls=TextPresets.muted_sm),
                    P("LightGBM 3-class moneyness + IV-rank maturity rule")),
                Div(P(Strong("Data Period"), cls=TextPresets.muted_sm),
                    P("2008-2025 (~1,391 monthly decision points)")),
                style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;",
            ),
            header=H4("System Summary", style=f"color:{_IMMACULATA};"),
            cls="mt-2",
        ),
    )


def _docs_data_pipeline():
    """Section 2: Data sources and cleaning."""
    return _doc_section("doc-data-pipeline", "Data Pipeline",
        P("Raw data is pulled from Alpha Vantage APIs and cached in an S3 mirror. The pipeline processes six data sources "
          "through standardization, deduplication, type conversion, and quality validation."),
        P("Options data represents monthly snapshots of the full options board — all listed contracts at all strikes "
          "and expirations, not executed trades. For AAPL this means ~900 call contracts per snapshot; for smaller names "
          "like AVGO, ~200. Cleaning reduces options from 3.2M to 1.1M contracts by filtering to reasonable delta "
          "(0.10-0.70) and DTE (7-150 days) ranges.", cls="mt-2"),
        Card(
            Div(
                Div(P(Strong("Daily Prices")), P("52,486 rows — OHLCV + dividends + splits (2000-2025)", cls=TextPresets.muted_sm)),
                Div(P(Strong("Income Statements")), P("781 quarterly reports — revenue, margins, net income", cls=TextPresets.muted_sm)),
                Div(P(Strong("Balance Sheets")), P("773 quarterly reports — assets, liabilities, equity", cls=TextPresets.muted_sm)),
                Div(P(Strong("Cash Flow")), P("778 quarterly reports — operating cash flow, capex", cls=TextPresets.muted_sm)),
                Div(P(Strong("Options")), P("1.1M contracts after cleaning (from 3.2M raw) — full monthly snapshots", cls=TextPresets.muted_sm)),
                Div(P(Strong("Overview")), P("10 companies — sector, industry, beta, dividend yield", cls=TextPresets.muted_sm)),
                style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;",
            ),
            header=H4("Data Sources", style=f"color:{_IMMACULATA};"),
            cls="mt-2",
        ),
    )


def _docs_eda():
    """Section 3: Exploratory analysis."""
    return _doc_section("doc-exploratory-analysis", "Exploratory Analysis",
        _doc_row(
            Div(
                P("Comprehensive analysis of price behavior, volatility regimes, fundamental trends, "
                  "and options data characteristics across the 10-ticker universe."),
                P("Highest volatility: TSLA (54%), NVDA (52%), AMZN (42%). Best Sharpe ratios: "
                  "AVGO (1.13), TSLA (0.92), GOOGL (0.90). Highest correlated pair: GOOGL-GOOG (0.994).", cls="mt-2"),
                P("Options data has average implied volatility of 49.49%. META has the shortest "
                  "history (since Aug 2021); AAPL, AMZN, GOOGL, MSFT, NVDA, WMT have data since Feb 2008.", cls="mt-2"),
            ),
            _doc_fig("feature_correlations.png", "Feature correlation matrix — volatility features cluster, fundamentals form a separate block."),
        ),
        _doc_row(
            Div(
                P("Bucket boundary analysis uses delta for moneyness and DTE for maturity. "
                  "The 9-bucket structure (3 moneyness x 3 maturity) captures 87,194 contracts — 15.5% of all calls."),
                P("Sufficient coverage confirmed for all 10 tickers, though some long-dated ATM buckets "
                  "have lower liquidity (e.g. AVGO ATM_DTE90: 55 contracts, META ATM_DTE90: 41).", cls="mt-2"),
            ),
            _doc_fig("bucket_analysis.png", "Delta/DTE distributions with bucket boundaries and per-ticker coverage."),
        ),
        _doc_row(
            P("Target label distribution shows ATM_DTE60 as the most frequent best bucket (319 occurrences) "
              "and OTM10_DTE90 as the least (65). This class imbalance is handled via inverse-frequency "
              "class weights during training."),
            _doc_fig("label_distribution.png", "Label distribution — ATM_DTE60 most frequent, OTM10_DTE90 least."),
        ),
        _doc_row(
            P("Best bucket shifts between volatility regimes — ATM dominates in both low and high vol, "
              "but long-dated buckets shrink in high vol as the market prices in uncertainty. "
              "This motivates the IV-rank maturity rule used in production."),
            _doc_fig("label_by_vol_regime.png", "Best bucket by vol regime — long-dated shrinks in high vol."),
        ),
        _doc_row(
            P("Best bucket distribution shifts over time as more tickers enter the options market post-2010. "
              "ATM buckets (DTE30/DTE60) dominate across all years. Coverage peaks in 2022-2023 with the "
              "full 10-ticker universe."),
            _doc_fig("label_temporal.png", "Best bucket over time — coverage grows post-2010."),
        ),
    )


def _docs_features():
    """Section 4: Feature engineering + label construction."""
    return _doc_section("doc-feature-engineering", "Feature Engineering",
        _doc_row(
            Div(
                P("Features are computed at monthly decision points (last trading day of each month). "
                  "Technical indicators are computed from daily prices then sampled at decision dates. "
                  "Quarterly fundamentals are forward-filled via merge_asof to avoid lookahead bias."),
                P(Strong("Technical (15)"), cls="mt-2"),
                P("Volatility (10d/21d/63d), momentum (5d/21d/63d), price-to-SMA ratios, "
                  "SMA crossovers, drawdowns (63d/252d), volume ratio, vol regime", cls=TextPresets.muted_sm),
                P(Strong("Fundamental (10)"), cls="mt-2"),
                P("Gross/operating/net margin, revenue/earnings growth YoY, debt-to-equity, "
                  "cash ratio, ROE, ROA, free cash flow", cls=TextPresets.muted_sm),
                P(Strong("Valuation (5)"), cls="mt-2"),
                P("P/E, P/S, EV/EBITDA, FCF yield, market cap proxy", cls=TextPresets.muted_sm),
                P(Strong("Implied Volatility (8)"), cls="mt-2"),
                P("IV mean/median/skew, short/long-term IV, IV term structure, "
                  "IV rank (12-month percentile), IV month-over-month change", cls=TextPresets.muted_sm),
            ),
            _doc_fig("improved_feature_importance.png", "Top 20 features — IV change and momentum dominate. Orange = IV features."),
        ),
        _doc_row(
            Div(
                H4("Label Construction", style=f"color:{_IMMACULATA};"),
                P("Labels are constructed by computing realized covered call payoffs for every contract "
                  "in each bucket. For each (ticker, month), the bucket with the highest realized return "
                  "becomes the ground truth label."),
                P("Payoff = premium received + capped stock P&L (capped at strike if assigned). "
                  "The 3-class moneyness target (ATM/OTM5/OTM10) is used for the production model, "
                  "with maturity selected post-hoc by an IV-rank rule.", cls="mt-2"),
                P("Class weights are computed via inverse-frequency to handle imbalance. "
                  "This prevents frequent classes (ATM_DTE60) from dominating minority ones (OTM10_DTE90).", cls="mt-2"),
            ),
            _doc_fig("feature_label_relationships.png", "Feature-label relationships — how features correlate with bucket selection."),
        ),
    )


def _docs_models():
    """Section 5: Model architectures and progression."""
    return _doc_section("doc-models", "Models",
        # Baseline
        _doc_row(
            Div(
                H4("Baseline Models (9-Class)", style=f"color:{_IMMACULATA};"),
                P("Random Forest and XGBoost trained on the full 9-class target with 80/20 "
                  "stratified train/test split. 1,112 train / 279 test samples, 30 features, "
                  "StandardScaler normalization, median imputation."),
                P("RF: 37.3% accuracy / 0.358 macro F1. XGB: 38.7% / 0.356 macro F1. "
                  "5-fold CV confirms: RF 0.373, XGB 0.370.", cls="mt-2"),
                P("Both models lean heavily on adjusted_close and volume — price-level features "
                  "that were later identified as leaky (encode ticker identity, not strategy signal).", cls="mt-2"),
            ),
            _doc_fig("feature_importance_comparison.png", "Baseline feature importance — price-level features dominate (later pruned)."),
        ),
        # Improved
        _doc_row(
            Div(
                H4("Improved Models (6-Class, Time-Split)", style=f"color:{_IMMACULATA};"),
                P("Key improvements over baseline: merged DTE90 into DTE60 (limited samples), "
                  "pruned price-level features, switched to time-based train/test split (no future leakage), "
                  "added class weights, and tuned hyperparameters with Optuna + TimeSeriesSplit CV."),
                P("RF, XGBoost, and LightGBM all trained. Scores drop vs baseline — expected when "
                  "moving from stratified to temporal split (realistic evaluation).", cls="mt-2"),
            ),
            _doc_fig("improved_confusion_matrices.png", "6-class confusion matrices for tuned RF, XGB, and LGBM."),
        ),
        _doc_row(
            P("Switching from stratified to time-based split causes scores to drop — expected, since the model "
              "can no longer memorize future patterns. This is the honest evaluation."),
            _doc_fig("old_vs_new_comparison.png", "Baseline vs improved — temporal split is harder but honest."),
        ),
        _doc_row(
            P("With proper temporal validation, all three tree-based models (RF, XGB, LGBM) perform similarly. "
              "No single model dominates — the real gains come from feature engineering and class reduction."),
            _doc_fig("model_comparison.png", "Time-based split — all models perform similarly."),
        ),
        # Walk-forward
        _doc_row(
            Div(
                H4("Walk-Forward 3-Class (Production)", style=f"color:{_IMMACULATA};"),
                P("The production model simplifies to 3-class moneyness (ATM/OTM5/OTM10). "
                  "Maturity is selected by an IV-rank rule: high IV → SHORT, low IV → LONG."),
                P("Walk-forward validation retrains annually on expanding windows — year N uses "
                  "only data from years 1 to N-1. This eliminates any future leakage and simulates "
                  "real deployment.", cls="mt-2"),
                P("Overall macro F1 = 0.474, consistently above random baseline (0.333). "
                  "Strong ATM recall (415/619), OTM10 reasonably separated (156/261).", cls="mt-2"),
            ),
            _doc_fig("walkforward_confusion_matrix.png", "Walk-forward confusion matrix (LGBM, 3-class)."),
        ),
        _doc_row(
            P("Walk-forward F1 varies by year — peaks around 0.55 in 2014-2015 and 2018, dips to ~0.36 "
              "in 2024. The model beats random in every year tested. IV features (iv_change, iv_skew, "
              "iv_term_structure, iv_rank) consistently appear in the top 20."),
            _doc_fig("walkforward_yearly_f1.png", "Macro F1 by year — always above random baseline."),
        ),
        # LSTM
        _doc_row(
            Div(
                H4("LSTM Sequence Model", style=f"color:{_IMMACULATA};"),
                P("LSTM with temporal attention mechanism using multi-step historical sequences. "
                  "Tested on both 6-class direct and two-stage (moneyness + maturity) targets."),
                P("Validation loss diverges after ~20 epochs indicating overfitting — expected with "
                  "~1,300 samples. Tree-based models outperform on this dataset size.", cls="mt-2"),
            ),
            _doc_fig("lstm_training_curves.png", "LSTM training — validation diverges after ~20 epochs."),
        ),
        _doc_row(
            P("Two-stage LSTM collapses predictions toward ATM_LONG — the dominant class absorbs most "
              "predictions. The 6-class LSTM shows slightly better spread but still underperforms "
              "tree-based models. With ~1,300 samples, sequence models lack enough data to generalize."),
            _doc_fig("lstm_confusion_matrices.png", "LSTM confusion matrices — 6-class and two-stage variants."),
        ),
    )


def _docs_results():
    """Section 6: All models comparison and financial returns."""
    return _doc_section("doc-results", "Results",
        _doc_row(
            Div(
                P("The 6-class two-stage approach achieves highest raw accuracy (83.5%) but uses "
                  "non-temporal validation — inflated by data leakage. The walk-forward LGBM 3-class "
                  "model (production) achieves 47.4% macro F1 with realistic temporal evaluation."),
                P("Tree-based models (RF, XGB, LGBM) consistently outperform LSTMs on this tabular "
                  "dataset. LightGBM is selected for production due to fast inference, native "
                  "categorical support, and best walk-forward stability.", cls="mt-2"),
            ),
            _doc_fig("all_models_comparison.png", "All models — two-stage inflated by non-temporal split."),
        ),
        _doc_row(
            P("Cumulative returns comparison under walk-forward evaluation. The Oracle strategy "
              "(perfect foresight) shows theoretical ceiling. LGBM, static ATM, and random strategies "
              "are tightly clustered — the model's edge is in moneyness selection, not magnitude. "
              "Real alpha comes from avoiding the worst buckets rather than always picking the best."),
            _doc_fig("financial_returns_comparison.png", "Cumulative returns — Oracle vs LGBM vs static vs random."),
        ),
    )


def _docs_navbar(sections):
    """Top navigation bar for docs with scrollspy anchors."""
    return NavBar(
        *[A(name, href=f"#{sid}") for sid, name in sections],
        A("Trader", href="/trading"),
        brand=A(
            DivLAligned(
                UkIcon("book-open", height=20, width=20),
                H3("USD Trader", style=f"color:{_FOUNDERS}; margin:0;"),
                Span("| Docs", style=f"color:{_IMMACULATA}; font-size:0.85rem; margin-left:0.5rem;"),
            ),
            href="/",
        ),
        sticky=True,
        uk_scrollspy_nav=True,
        cls="p-3",
        style=f"border-bottom:2px solid {_IMMACULATA};",
    )


def docs_screen():
    """Documentation screen: scrollspy navbar + content with notebook findings and figures.

    Returns:
        Div with docs layout.
    """
    try:
        sections = [
            ("doc-overview", "Overview"),
            ("doc-data-pipeline", "Data Pipeline"),
            ("doc-exploratory-analysis", "Exploratory Analysis"),
            ("doc-feature-engineering", "Feature Engineering"),
            ("doc-models", "Models"),
            ("doc-results", "Results"),
        ]
        return Div(
            _docs_navbar(sections),
            Div(
                _docs_overview(),
                _docs_data_pipeline(),
                _docs_eda(),
                _docs_features(),
                _docs_models(),
                _docs_results(),
                cls="px-6 py-4",
            ),
        )
    except Exception:
        return _fallback("documentation")
