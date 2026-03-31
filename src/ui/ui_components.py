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
    """Backtesting dashboards: preset selector + results panel.

    Preset dropdown triggers a POST to /backtest_call which returns
    strategy vs baseline metrics rendered via backtest_results_card().

    Returns:
        Section Div.
    """
    return Section(
        H3("Backtesting Dashboards", style=f"color:{_FOUNDERS};"),
        P("Historical strategy performance vs. OTM10 baseline.", cls=TextPresets.muted_sm),
        DividerLine(),
        Grid(
            # Results panel — left (spans 4 of 7 cols)
            Div(_backtest_results_panel(), cls="col-span-4"),
            # Sidebar — right (spans 3 of 7 cols)
            Div(_backtest_sidebar(), cls="col-span-3"),
            cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
        ),
        id="backtesting",
    )


def _backtest_sidebar():
    """Sidebar with year dropdown, budget input, and run button.

    Returns:
        Card with backtest controls.
    """
    year_options = [fh.Option("All Years", value="all", selected=True)] + [
        fh.Option(str(y), value=str(y)) for y in range(2008, 2026)
    ]
    return Card(
        # Year dropdown (native select — MonsterUI uk-select doesn't reliably show selected on load)
        Label("Time Window"),
        fh.Select(*year_options, name="year", id="backtest-year", cls="uk-select"),
        # Budget input
        Label("Budget ($)", cls="mt-3"),
        Input(type="number", name="budget", id="backtest-budget",
              value="100000", min="10000", step="10000"),
        # Run button
        Button("Run Backtest",
               cls="w-full mt-4",
               style=f"background-color:{_IMMACULATA}; color:white;",
               hx_post="/backtest_call",
               hx_include="#backtest-year, #backtest-budget",
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


def backtest_results_card(data: dict):
    """Render backtest results: metrics table with Baseline + 3 presets as columns.

    Args:
        data: Combined backtest report from run_backtest_all().

    Returns:
        Div with comparison table.
    """
    try:
        n_months = data.get("n_months", 0)
        year = data.get("year", "all")
        date_range = data.get("date_range", {})
        bm = data["baseline"]["metrics"]
        presets = data["presets"]

        def fmt_pct(v):
            return f"{v:.1%}"

        def fmt_ratio(v):
            return f"{v:.2f}"

        # Build comparison table rows
        metrics = [
            ("Annualized Return", "annualized_return", fmt_pct, True),
            ("Sharpe Ratio", "sharpe_ratio", fmt_ratio, True),
            ("Max Drawdown", "max_drawdown", fmt_pct, False),
            ("Hit Rate", "hit_rate", fmt_pct, True),
            ("Avg P / Avg L", "avg_p_l", fmt_ratio, True),
        ]

        header = ["Metric", "Baseline", "Conservative", "Balanced", "Aggressive"]
        rows = []
        for label, key, fmt, higher_is_better in metrics:
            baseline_val = bm[key]
            row = {"Metric": label, "Baseline": fmt(baseline_val)}
            for preset in ["conservative", "balanced", "aggressive"]:
                val = presets[preset]["metrics"][key]
                formatted = fmt(val)
                # Color: green if strategy beats baseline, red if not
                if higher_is_better:
                    beats = val >= baseline_val
                else:
                    beats = val >= baseline_val  # For drawdown, less negative is better
                color = "#2e7d32" if beats else "#c62828"
                row[preset.title()] = formatted
            rows.append(row)

        period_label = f"Year: {year}" if year != "all" else "All Years"
        period_detail = f"{date_range.get('start', '?')} to {date_range.get('end', '?')}"

        return Div(
            Card(
                P(f"{period_label} ({period_detail}) | {n_months} months | Budget: ${data.get('budget', 0):,.0f}",
                  cls=TextPresets.muted_sm),
                P(f"Baseline: OTM10 all tickers, equal weight", cls=(TextT.sm, TextT.italic)),
                DividerLine(),
                TableFromDicts(header_data=header, body_data=rows),
                header=H4("Strategy Comparison", style=f"color:{_FOUNDERS};"),
            ),
        )

    except Exception:
        return _fallback("backtest results")


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
            P("Best moneyness distribution by year and ticker. ATM dominates across all years and most tickers, "
              "with OTM5 and OTM10 appearing more in volatile periods. WMT and MSFT skew heavily toward ATM; "
              "NVDA and TSLA show more OTM diversity. Class imbalance is handled via inverse-frequency "
              "class weights during training."),
            _doc_fig("label_distribution.png", "Best moneyness by year and by ticker — ATM dominates."),
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
        # STALE figures from prior 9-class iteration (notebooks rewritten to 3-class):
        #   _doc_fig("feature_importance_comparison.png", ...) — showed leaky adjusted_close/volume at top
        #   _doc_fig("rf_confusion_matrix.png", ...) — 9-class RF confusion matrix
        #   _doc_fig("xgb_confusion_matrix.png", ...) — 9-class XGB confusion matrix
        #   _doc_fig("rf_feature_importance.png", ...) — 9-class RF feature importance (leaky features)
        #   _doc_fig("xgb_feature_importance.png", ...) — 9-class XGB feature importance (leaky features)
        # Replaced with 3-class equivalents: baseline_confusion.png, baseline_feature_importance.png
        _doc_row(
            Div(
                H4("Baseline Models (3-Class)", style=f"color:{_IMMACULATA};"),
                P("Random Forest and XGBoost trained on the 3-class moneyness target (ATM/OTM5/OTM10) "
                  "with 80/20 time-based split on daily data. 21,395 train / 7,878 test rows, 27 features."),
                P("RF: 48.5% accuracy / 0.333 macro F1. XGB: 48.0% / 0.359 macro F1. "
                  "Both heavily favor predicting ATM (the majority class), with poor OTM5 recall.", cls="mt-2"),
                P("Top features: debt_to_equity, cash_ratio, gross_margin, revenue_growth_yoy — "
                  "fundamentals dominate after pruning leaky price-level features (adjusted_close, volume).", cls="mt-2"),
            ),
            _doc_fig("baseline_feature_importance.png", "Baseline feature importance — fundamentals dominate after pruning leaky features."),
        ),
        _doc_row(
            P("Baseline confusion matrices show both models default to ATM predictions. "
              "RF misclassifies 1,778 OTM10 samples as ATM; XGB does slightly better with 1,624. "
              "OTM5 is the hardest class — only 45 correct predictions by RF, 198 by XGB."),
            _doc_fig("baseline_confusion.png", "RF and XGB 3-class confusion matrices — ATM prediction bias."),
        ),
        # Improved
        # STALE figures from prior 6-class iteration (notebooks rewritten to 3-class):
        #   _doc_fig("old_vs_new_comparison.png", ...) — baseline vs 6-class comparison
        #   _doc_fig("model_comparison.png", ...) — 6-class time-split comparison
        # Removed: these 6-class models no longer exist in any notebook
        _doc_row(
            Div(
                H4("Optuna-Tuned Models (3-Class)", style=f"color:{_IMMACULATA};"),
                P("Hyperparameters tuned via Optuna (20 trials each) with TimeSeriesSplit cross-validation "
                  "and balanced class weights. All three tree-based models trained: RF, XGBoost, LightGBM."),
                P("RF: 50.0% / 0.338 macro F1. XGB: 50.4% / 0.342. LightGBM: 48.4% / 0.349. "
                  "Tuning provides minimal improvement over baselines — all remain near random-level (0.333).", cls="mt-2"),
                P("LightGBM is best by macro F1 despite lowest accuracy, suggesting better minority-class "
                  "recall. Top features: vol_63d, vol_21d, bb_width, earnings_growth_yoy, cash_ratio.", cls="mt-2"),
            ),
            _doc_fig("improved_confusion_matrices.png", "Tuned RF, XGB, and LGBM 3-class confusion matrices."),
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
                P("Overall macro F1 = 0.468, consistently above random baseline (0.333). "
                  "Strong ATM recall (13,472 correct), OTM10 reasonably separated (1,850 correct out of 4,557).", cls="mt-2"),
            ),
            _doc_fig("walkforward_confusion_matrix.png", "Walk-forward confusion matrix (LGBM, 3-class)."),
        ),
        _doc_row(
            P("Walk-forward F1 varies by year — peaks at 0.60 in 2019, sustained above 0.50 from 2015-2021, "
              "dips to ~0.32 in 2023. The model beats random (0.333) in most years. "
              "IV features (iv_mean, iv_std, iv_change, iv_rank) dominate the top 4 feature importances."),
            _doc_fig("walkforward_yearly_f1.png", "Macro F1 by year — always above random baseline."),
        ),
        # LSTM
        # STALE figure from prior 6-class iteration:
        #   _doc_fig("lstm_confusion_matrices.png", ...) — 6-class + two-stage LSTM variants
        # Replaced with 3-class equivalent: lstm_confusion_matrix.png (singular)
        _doc_row(
            Div(
                H4("LSTM Sequence Model", style=f"color:{_IMMACULATA};"),
                P("Bidirectional LSTM with temporal attention using 60-day lookback sequences of daily features. "
                  "3-class moneyness target, 80/20 time-based split (1,010 train / 379 test sequences)."),
                P("46.2% accuracy / 0.411 macro F1. Top-2 accuracy of ~70% — the correct class is usually "
                  "the model's 1st or 2nd prediction. Early stopping at epoch 29 (patience=15).", cls="mt-2"),
                P("Validation loss diverges after ~20 epochs indicating overfitting — expected with "
                  "~1,300 samples. Tree-based models outperform on this dataset size.", cls="mt-2"),
            ),
            _doc_fig("lstm_training_curves.png", "LSTM training — validation diverges after ~20 epochs."),
        ),
        _doc_row(
            P("LSTM is better at predicting OTM10 (96 correct) than the tree baselines, but struggles with "
              "OTM5 (only 14 correct out of 71). ATM predictions scatter across all classes (65/24/72). "
              "With ~1,300 samples, sequence models lack enough data to generalize."),
            _doc_fig("lstm_confusion_matrix.png", "LSTM 3-class confusion matrix — better OTM10, weak OTM5."),
        ),
    )


def _docs_results():
    """Section 6: All models comparison and financial returns."""
    # NOTE: all_models_comparison.png is STALE — shows 9-class and 6-class models
    # that no longer exist in the notebooks (all rewritten to 3-class).
    # TODO: regenerate when notebook 08 (model comparison) is built.
    return _doc_section("doc-results", "Results",
        _doc_row(
            Div(
                P("All models compared across the full development progression. Baseline 9-class models "
                  "(RF 0.358, XGB 0.356 F1) used stratified splits — inflated by data leakage. "
                  "Time-based 6-class splits drop sharply (0.17-0.23 F1), revealing the true difficulty."),
                P("The LSTM daily-sequence model (0.411 F1) benefits from temporal patterns but is limited by "
                  "sample size. Walk-forward LGBM 3-class (production, 0.468 F1) achieves best honest evaluation. "
                  "LightGBM selected for production: fast inference, native categorical support, walk-forward stability.", cls="mt-2"),
            ),
            _doc_fig("all_models_comparison.png", "Model development progression — includes prior 9-class and 6-class iterations (since superseded by 3-class)."),
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
