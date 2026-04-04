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


def _tip(text: str):
    """Render a small tooltip bubble (?) with hover explanation.

    Uses UIKit's uk-tooltip — no JS, just a data attribute.
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
        # batch checkbox — grays out ticker dropdown when checked
        Div(
            Label(
                CheckboxX(id="batch-check", name="batch",
                          hx_get="/toggle_ticker",
                          hx_include="#batch-check",
                          hx_target="#inference-ticker",
                          hx_swap="outerHTML"),
                " All Stocks",
            ),
            cls="mt-2",
        ),
        # compute button — posts to /inference_call, swaps results panel
        Button("Compute Inference",
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
        # Tuples: (label, value, tooltip_or_None)
        display_rows = [
            ("Ticker", data.get("ticker", "—"), None),
            ("Date", data.get("date", "—"), None),
            ("Month", data.get("month", "—"), None),
            ("Model", "LGBM 3-Class Moneyness", None),
            ("Prediction", data.get("model_bucket", "—"),
             "Recommended strike bucket: ATM (at the money), OTM5 (5% out of the money), OTM10 (10% out). SHORT = sell within 45 days, LONG = sell 46-120 days."),
            ("Confidence", f"{data.get('model_confidence', 0):.1%}",
             "How sure the model is about its top pick. Higher means the model sees a clearer signal."),
            ("Baseline", data.get("baseline", "—"),
             "What you'd pick if you ignored the model entirely: always sell 10% OTM short-dated calls."),
            ("Sample", data.get("sample_type", "—"),
             "Train Dataset = the model learned from this data. Test Dataset = the model has never seen this data."),
        ]

        # Build table rows with inline tooltips
        table_rows = []
        for label, value, tip_text in display_rows:
            metric_cell = Span(label)
            if tip_text:
                metric_cell = Span(label, _tip(tip_text))
            table_rows.append(Tr(Td(metric_cell), Td(value)))

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
        # Using a static alert Div instead of Toast — Toast needs UIKit JS init
        # which doesn't fire on htmx swap
        snap_warning = None
        if data.get("snapped"):
            snap_warning = Div(
                P(f"No data for requested month — showing nearest available: {data.get('month', '?')}",
                  style="margin:0;"),
                cls="uk-alert uk-alert-warning",
                style="padding:0.75rem 1rem; margin-bottom:0.5rem; border-radius:6px; "
                      f"background:{_TORERO}22; border-left:4px solid {_TORERO};",
            )

        return Div(
            snap_warning if snap_warning else "",
            Card(
                Div(
                    # table side — left ~40%
                    Div(
                        Table(
                            Thead(Tr(Th("Metric"), Th("Value"))),
                            Tbody(*table_rows),
                            cls="uk-table uk-table-small uk-table-divider",
                        ),
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


def _batch_ticker_chart_modal(ticker: str, date: str):
    """Render a per-ticker candlestick chart inside a modal via lazy loading.

    The chart is NOT pre-loaded — it fetches via htmx when the modal opens.
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
            header=f"{ticker} — Price Chart",
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
            "Date": summary.get("date", "—"),
            "Model": summary.get("model", "—"),
            "Tickers": str(summary.get("n_tickers", 0)),
            "Top Pick": f"{summary.get('top_ticker', '?')} ({summary.get('top_prediction', '?')})",
            "Top Confidence": f"{summary.get('top_confidence', 0):.1%}",
            "Avg Confidence": f"{summary.get('avg_confidence', 0):.1%}",
            "Accuracy": f"{summary.get('accuracy', 0):.1%}",
            "Sample": summary.get("sample_type", "—"),
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
                "Prediction": r.get("model_bucket", "—"),
                "Confidence": f"{r.get('model_confidence', 0):.1%}",
                "Correct": "Y" if r.get("model_correct") else "N",
                "Sample": r.get("sample_type", "—"),
            })

        detail_header = ["Ticker", "Prediction", "Confidence", "Correct", "Sample", "Chart"]

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
                    Td(r.get("model_bucket", "—")),
                    Td(f"{r.get('model_confidence', 0):.1%}"),
                    Td("Y" if r.get("model_correct") else "N"),
                    Td(r.get("sample_type", "—")),
                    Td(_batch_ticker_chart_modal(ticker, batch_date)),
                )
            )

        detail_table = Table(
            Thead(Tr(*[Th(h) for h in detail_header])),
            Tbody(*table_rows),
            cls="uk-table uk-table-small uk-table-divider",
        )

        # Snap warning for batch — show which tickers snapped
        snapped_tickers = [r.get("ticker") for r in results
                           if "error" not in r and r.get("snapped")]
        batch_snap_warning = None
        if snapped_tickers:
            batch_snap_warning = Div(
                P(f"Date snapped for: {', '.join(snapped_tickers)} — showing nearest available month.",
                  style="margin:0;"),
                cls="uk-alert uk-alert-warning",
                style="padding:0.75rem 1rem; margin-bottom:0.5rem; border-radius:6px; "
                      f"background:{_TORERO}22; border-left:4px solid {_TORERO};",
            )

        return Div(
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
                header=H4(f"Batch Inference — {summary.get('date', '?')} ({summary.get('n_tickers', 0)} tickers)",
                          style=f"color:{_FOUNDERS};"),
            ),
            # Detail modal with per-ticker rows + chart icons
            Modal(
                detail_table,
                header=f"All Results — {summary.get('date', '?')}",
                id=detail_modal_id,
                dialog_cls="uk-modal-dialog-large",
            ),
        )

    except Exception:
        return _fallback("batch inference results")


# ═══════════════════════════════════════════════════════════════════════════════
# backtesting dashboards
# ═══════════════════════════════════════════════════════════════════════════════

# the backtesting dashboards section should be based upon the MonsterUI dashboard example (https://monsterui.answer.ai/dashboard/)
# and allow the user to understand (four different sections):

# overall system performance (in terms of profit and risk), per stock performance (dropdown), baseline statistics
# (what calls the system made the most, etc), baseline statistics in terms of the EDA

def _backtesting_section():
    """Backtesting dashboards with tabbed layout.

    Three tabs: Strategy, Model Performance, MLflow.
    UIKit uk-tab + uk-switcher handles tab switching (zero JS).
    Each pane has its own sidebar + results grid.

    Returns:
        Section Div.
    """
    return Section(
        H3("Dashboards", style=f"color:{_FOUNDERS};"),
        P("Strategy backtesting, model evaluation, and experiment tracking.", cls=TextPresets.muted_sm),
        DividerLine(),
        # Tab headers
        Ul(
            Li(A("Strategy"), cls="uk-active"),
            Li(A("Model Performance")),
            Li(A("MLflow")),
            uk_tab="",
        ),
        # Tab panes
        Ul(
            # Tab 1: Strategy (existing)
            Li(
                Grid(
                    Div(_backtest_results_panel(), cls="col-span-5"),
                    Div(_backtest_sidebar(), cls="col-span-2"),
                    cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
                ),
                cls="uk-active",
            ),
            # Tab 2: Model Performance (lazy loaded)
            Li(
                Grid(
                    Div(_model_performance_panel(), cls="col-span-4"),
                    Div(_model_sidebar(), cls="col-span-3"),
                    cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1, gap=4,
                ),
            ),
            # Tab 3: MLflow (lazy-loaded on tab switch)
            Li(
                Div(
                    Loading(htmx_indicator=True, id="mlflow-spinner"),
                    id="mlflow-results",
                    hx_get="/mlflow_call",
                    hx_trigger="intersect once",
                    hx_swap="innerHTML",
                    hx_indicator="#mlflow-spinner",
                ),
            ),
            cls="uk-switcher mt-4",
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
        # Budget commented out — percentage-based metrics are invariant to budget scale
        # Label("Budget ($)", cls="mt-3"),
        # Input(type="number", name="budget", id="backtest-budget",
        #       value="100000", min="10000", step="10000"),
        # Run button
        Button("Run Backtest",
               cls="w-full mt-4",
               style=f"background-color:{_IMMACULATA}; color:white;",
               hx_post="/backtest_call",
               hx_include="#backtest-year",
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
        am = data.get("argmax", {}).get("metrics", {})
        ra = data.get("risk_adjusted", {}).get("metrics", {})
        presets = data["presets"]

        def fmt_pct(v):
            return f"{v:.1%}"

        def fmt_ratio(v):
            return f"{v:.2f}"

        # Build comparison table rows with tooltips
        metrics = [
            ("Annualized Return", "annualized_return", fmt_pct, True,
             "Total return converted to a yearly rate. What you'd earn per year on average."),
            ("Sharpe Ratio", "sharpe_ratio", fmt_ratio, True,
             "Return per unit of risk. Higher is better. Above 1 is good, above 2 is very good."),
            ("Max Drawdown", "max_drawdown", fmt_pct, False,
             "Biggest peak-to-trough drop. The worst-case loss you'd experience before recovery."),
            ("Hit Rate", "hit_rate", fmt_pct, True,
             "Percentage of months that ended with a positive return."),
            ("Avg P / Avg L", "avg_p_l", fmt_ratio, True,
             "Average winning month divided by average losing month. Above 1 means wins are bigger than losses."),
        ]

        # Columns: Baseline | Argmax | Risk-Adjusted | Conservative | Balanced | Aggressive
        bt_table_rows = []
        for label, key, fmt, higher_is_better, tip_text in metrics:
            cells = [
                Td(Span(label, _tip(tip_text))),
                Td(fmt(bm.get(key, 0))),
                Td(fmt(am.get(key, 0))),
                Td(fmt(ra.get(key, 0))),
            ]
            for preset in ["conservative", "balanced", "aggressive"]:
                val = presets[preset]["metrics"][key]
                cells.append(Td(fmt(val)))
            bt_table_rows.append(Tr(*cells))

        period_label = f"Year: {year}" if year != "all" else "All Years"
        period_detail = f"{date_range.get('start', '?')} to {date_range.get('end', '?')}"

        return Div(
            Card(
                P(f"{period_label} ({period_detail}) | {n_months} months",
                  cls=TextPresets.muted_sm),
                DividerLine(),
                Table(
                    Thead(Tr(
                        Th("Metric"),
                        Th(Span("Baseline", _tip("Always sell 10% OTM short-dated calls on all tickers, equal weight. No model."))),
                        Th(Span("Argmax", _tip("Model's single best prediction per ticker, all tickers traded, equal weight."))),
                        Th(Span("Risk-Adjusted", _tip("Picks the bucket that maximizes probability times expected return. Uses expanding historical averages."))),
                        Th(Span("Conservative", _tip("Wide diversification (7 positions), prioritizes low-cost trades."))),
                        Th(Span("Balanced", _tip("Equal weight to confidence, cost, and vol premium (5 positions)."))),
                        Th(Span("Aggressive", _tip("Concentrated bets (3 positions), chases highest model confidence."))),
                    )),
                    Tbody(*bt_table_rows),
                    cls="uk-table uk-table-small uk-table-divider",
                ),
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
        fh.Option("All", value="all", selected=True),
        fh.Option("Train Dataset", value="train"),
        fh.Option("Test Dataset", value="test"),
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


def model_performance_card(data: dict):
    """Render model performance metrics: summary, per-class, per-year, comparison.

    Args:
        data: Dict from compute_model_metrics() with metrics and breakdowns.

    Returns:
        Div with metrics cards and tables.
    """
    try:
        n = data.get("n_samples", 0)
        year_filter = data.get("year_filter", "all")
        sample_filter = data.get("sample_filter", "all")

        # ── Summary metrics row ──
        summary_items = [
            ("Accuracy", f"{data.get('accuracy', 0):.1%}",
             "How often the model picks the correct bucket overall."),
            ("Macro F1", f"{data.get('macro_f1', 0):.3f}",
             "Average F1 across all classes, weighted equally. Balances precision and recall. Random guessing = 0.33."),
            ("Top-2 Accuracy", f"{data.get('top2_accuracy', 0):.1%}",
             "How often the correct answer is the model's 1st or 2nd choice."),
            ("Samples", f"{n:,}", None),
        ]
        summary_cards = Div(
            *[
                Div(
                    P(Strong(label), _tip(tip) if tip else "",
                      cls=TextPresets.muted_sm, style="margin-bottom:0.25rem;"),
                    P(value, style="font-size:1.5rem; font-weight:600; margin:0;"),
                    style="text-align:center; padding:0.75rem;",
                )
                for label, value, tip in summary_items
            ],
            style="display:grid; grid-template-columns:repeat(4, 1fr); gap:0.75rem;",
        )

        # ── Per-class breakdown table ──
        per_class = data.get("per_class", {})
        pc_header = Tr(
            Th("Class"),
            Th(Span("Precision", _tip("Of all times the model predicted this class, how often was it right."))),
            Th(Span("Recall", _tip("Of all actual instances of this class, how many did the model catch."))),
            Th(Span("F1", _tip("Balance between precision and recall. 1.0 is perfect, 0.33 is random."))),
            Th(Span("Support", _tip("How many times this class actually appeared in the data."))),
            Th(Span("Predicted", _tip("How many times the model chose this class."))),
        )
        pc_rows = []
        for cls in ["ATM", "OTM5", "OTM10"]:
            m = per_class.get(cls, {})
            pc_rows.append(Tr(
                Td(Strong(cls)),
                Td(f"{m.get('precision', 0):.3f}"),
                Td(f"{m.get('recall', 0):.3f}"),
                Td(f"{m.get('f1', 0):.3f}"),
                Td(str(m.get("support", 0))),
                Td(str(m.get("predicted", 0))),
            ))

        # ── Per-year breakdown table ──
        per_year = data.get("per_year", [])
        year_header = ["Year", "Accuracy", "Macro F1", "Top-2", "Samples"]
        year_rows = [
            {
                "Year": str(y.get("year", "?")),
                "Accuracy": f"{y.get('accuracy', 0):.1%}",
                "Macro F1": f"{y.get('macro_f1', 0):.3f}",
                "Top-2": f"{y.get('top2', 0):.1%}",
                "Samples": str(y.get("n_samples", 0)),
            }
            for y in per_year
        ]

        # ── Confidence analysis ──
        conf = data.get("confidence", {})
        conf_items = [
            ("Avg Confidence (Correct)", f"{conf.get('avg_when_correct', 0):.1%}",
             "How confident the model is when it gets the answer right. Higher means the model knows when it's right."),
            ("Avg Confidence (Incorrect)", f"{conf.get('avg_when_incorrect', 0):.1%}",
             "How confident the model is when it gets it wrong. If close to 'correct', the model can't tell when it's guessing."),
            ("Overall Avg Confidence", f"{conf.get('overall_avg', 0):.1%}",
             "Average confidence across all predictions regardless of correctness."),
        ]

        # ── Model information (single column — LGBM production model) ──
        info_header = ["Metric", "Value"]
        info_rows = [
            {"Metric": "Task", "Value": "3-class moneyness (ATM / OTM5 / OTM10)"},
            {"Metric": "Macro F1", "Value": f"{data.get('macro_f1', 0):.3f}"},
            {"Metric": "Accuracy", "Value": f"{data.get('accuracy', 0):.1%}"},
            {"Metric": "Validation", "Value": "Walk-forward annual"},
            {"Metric": "Inference", "Value": "Row lookup (<1ms)"},
            {"Metric": "Status", "Value": "Production"},
        ]

        filter_label = f"Year: {year_filter}" if year_filter != "all" else "All Years"
        sample_labels = {"all": "All Data", "train": "Train Dataset", "test": "Test Dataset"}
        sample_label = sample_labels.get(sample_filter, sample_filter)

        return Div(
            Card(
                P(f"{filter_label} | {sample_label} | {data.get('model_name', '?')}",
                  cls=TextPresets.muted_sm),
                DividerLine(),
                # Summary metrics
                summary_cards,
                DividerLine(),
                # Per-class breakdown
                H4("Per-Class Breakdown", style=f"color:{_IMMACULATA}; font-size:1rem;"),
                Table(Thead(pc_header), Tbody(*pc_rows),
                      cls="uk-table uk-table-small uk-table-divider"),
                DividerLine(),
                # Confidence analysis
                H4("Confidence Analysis", style=f"color:{_IMMACULATA}; font-size:1rem;"),
                Div(
                    *[Div(
                        P(Strong(label), _tip(tip), cls=TextPresets.muted_sm),
                        P(value),
                    ) for label, value, tip in conf_items],
                    style="display:grid; grid-template-columns:repeat(3, 1fr); gap:0.75rem; text-align:center;",
                ),
                DividerLine(),
                # Model information
                H4("Model Information", style=f"color:{_IMMACULATA}; font-size:1rem;"),
                TableFromDicts(header_data=info_header, body_data=info_rows),
                header=H4("Model Performance", style=f"color:{_FOUNDERS};"),
            ),
            # Per-year breakdown in a collapsible section
            Card(
                TableFromDicts(header_data=year_header, body_data=year_rows),
                header=H4("Performance by Year", style=f"color:{_FOUNDERS};"),
                cls="mt-2",
            ) if year_filter == "all" and year_rows else "",
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
            params_cell = "—"
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
                        header=f"{run['run_name']} — Hyperparameters",
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
                        header=f"{run['run_name']} — Confusion Matrix",
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
                        header=f"{run['run_name']} — ROC Curves",
                        id=roc_modal_id,
                        dialog_cls="uk-modal-dialog-large",
                    )
                )

            rows.append(Tr(
                Td(Strong(run["run_name"])),
                Td(run.get("variant", "—")),
                Td(run.get("n_classes", "—")),
                Td(run.get("n_features", "—")),
                Td(f"{m.get('val_macro_f1', 0):.3f}" if m.get("val_macro_f1") else "—"),
                Td(f"{m.get('test_macro_f1', 0):.3f}" if m.get("test_macro_f1") else "—"),
                Td(f"{m.get('test_accuracy', 0):.1%}" if m.get("test_accuracy") else "—"),
                Td(f"{m.get('test_balanced_accuracy', 0):.1%}" if m.get("test_balanced_accuracy") else "—"),
                Td(params_cell),
                Td(Span(*plot_links) if plot_links else "—"),
            ))

        exp_name = experiments[0].get("experiment_name", "Unknown")
        total = experiments[0].get("total_runs", 0)
        unique = experiments[0].get("unique_runs", 0)

        return Div(
            Card(
                P(f"Experiment: {exp_name} | {total} total runs, {unique} unique",
                  cls=TextPresets.muted_sm),
                P("Deep learning and XGBoost runs tracked via MLflow (7-class task). "
                  "LGBM 3-class production model tracked separately via walk-forward validation — included for comparison.",
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
        P("This system uses machine learning to optimize covered call option selling across a universe of 10 large-cap "
          "US equities. Given a stock and a monthly decision date, the model predicts which moneyness bucket "
          "(ATM, OTM5%, or OTM10%) is most likely to yield the highest realized covered call return."),
        P("Two modeling approaches were developed in parallel: a LightGBM pipeline with walk-forward validation "
          "on a simplified 3-class target, and a deep learning pipeline (LSTM-CNN, PatchTST) on the full 7-class "
          "moneyness-maturity space. Both are documented here.", cls="mt-2"),
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
                Div(P(Strong("LightGBM Pipeline"), cls=TextPresets.muted_sm),
                    P("3-class moneyness + IV-rank maturity rule (production)")),
                Div(P(Strong("Deep Learning Pipeline"), cls=TextPresets.muted_sm),
                    P("7-class LSTM-CNN + PatchTST (experimental)")),
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
    """Section 4: Feature engineering — two-column comparison of both pipelines."""
    return _doc_section("doc-feature-engineering", "Feature Engineering",
        P("Both pipelines share the same raw data but engineer different feature sets and targets. "
          "Features are computed at monthly decision points; quarterly fundamentals are forward-filled "
          "via merge_asof to avoid lookahead bias."),
        _doc_columns(
            # Left: LGBM pipeline
            Div(
                H4("LightGBM Pipeline", style=f"color:{_IMMACULATA};"),
                P(Strong("27 + 8 IV features"), cls="mt-1"),
                P(Strong("Technical (15)"), cls="mt-2"),
                P("Volatility (10d/21d/63d), momentum (5d/21d/63d), price-to-SMA ratios, "
                  "SMA crossovers, drawdowns, volume ratio, vol regime", cls=TextPresets.muted_sm),
                P(Strong("Fundamental (10)"), cls="mt-2"),
                P("Margins, revenue/earnings growth, debt-to-equity, cash ratio, ROE, ROA, FCF", cls=TextPresets.muted_sm),
                P(Strong("Implied Volatility (8)"), cls="mt-2"),
                P("IV mean/median/skew, short/long-term IV, IV term structure, "
                  "IV rank, IV month-over-month change", cls=TextPresets.muted_sm),
                P(Strong("Target: 3-class moneyness"), cls="mt-3"),
                P("ATM / OTM5 / OTM10. Maturity selected post-hoc by IV-rank rule.", cls=TextPresets.muted_sm),
            ),
            # Right: DL pipeline
            Div(
                H4("Deep Learning Pipeline", style=f"color:{_IMMACULATA};"),
                P(Strong("Top 35 features (RF-selected)"), cls="mt-1"),
                P(Strong("Price & Returns"), cls="mt-2"),
                P("Open, High, Low, Close, Volume, daily/weekly/monthly returns", cls=TextPresets.muted_sm),
                P(Strong("Technical"), cls="mt-2"),
                P("Rolling volatility (5/20/60d), momentum, RSI, MACD, Bollinger Bands", cls=TextPresets.muted_sm),
                P(Strong("Valuation & Profitability"), cls="mt-2"),
                P("P/E, P/S, EV/EBITDA, Price/Book, margins, ROA, ROE, leverage ratios", cls=TextPresets.muted_sm),
                P(Strong("Target: 7-class buckets"), cls="mt-3"),
                P("ATM_30, ATM_60, ATM_90, OTM5_30, OTM5_60_90, OTM10_30, OTM10_60_90", cls=TextPresets.muted_sm),
            ),
        ),
        # Shared label construction — no figure, text only
        Div(
            H4("Label Construction (Shared)", style=f"color:{_IMMACULATA};"),
            P("Both pipelines construct labels by computing realized covered call payoffs for every contract "
              "in each bucket. For each (ticker, month), the bucket with the highest realized return "
              "becomes the ground truth label."),
            P("Payoff = premium received + capped stock P&L (capped at strike if assigned). "
              "Class weights are computed via inverse-frequency to handle imbalance.", cls="mt-2"),
            style=f"border-bottom:1px solid {_TORERO}; padding-bottom:0.75rem; margin-top:0.75rem;",
        ),
    )


def _docs_lgbm_pipeline():
    """Section 5: Tree-based pipeline — 3-class moneyness, RF → XGB → LGBM → walk-forward."""
    return _doc_section("doc-lgbm-pipeline", "Tree-Based Pipeline (3-Class)",
        P("This pipeline reformulates the problem as a simpler 3-class moneyness prediction "
          "(ATM / OTM5 / OTM10), decoupling maturity selection into a post-hoc IV-rank rule. "
          "Three tree-based architectures were evaluated — Random Forest, XGBoost, and LightGBM — "
          "each progressively refined through Optuna hyperparameter tuning and walk-forward validation. "
          "LightGBM with walk-forward annual retraining emerged as the production model.", cls="mb-2"),
        # Baselines
        _doc_row(
            Div(
                H4("Random Forest & XGBoost Baselines", style=f"color:{_IMMACULATA};"),
                P("Both models trained on the 3-class moneyness target with 80/20 time-based split. "
                  "21,395 training rows, 7,878 test rows, 27 features engineered from daily price "
                  "data and quarterly fundamentals."),
                P("Random Forest: 48.5% accuracy / 0.333 macro F1. Heavily biased toward ATM — "
                  "misclassifies 1,778 OTM10 samples as ATM. Only 45 correct OTM5 predictions out of "
                  "2,266 actual OTM5 instances.", cls="mt-2"),
                P("XGBoost: 48.0% accuracy / 0.359 macro F1. Slightly better OTM5 recall (198 correct) "
                  "but still dominated by ATM predictions. Top features at this stage: debt_to_equity, "
                  "cash_ratio, gross_margin, revenue_growth_yoy — fundamentals dominate after "
                  "pruning leaky price-level features (adjusted_close, volume).", cls="mt-2"),
            ),
            _doc_fig("baseline_confusion.png", "RF and XGB 3-class confusion matrices — both default to ATM."),
        ),
        Div(
            P("Feature importance analysis at the baseline stage revealed that raw price-level features "
              "(adjusted_close, volume) were leaking information. After removing them, fundamental "
              "features took over the importance rankings. This informed the feature selection for "
              "subsequent stages."),
            style=f"border-bottom:1px solid {_TORERO}; padding-bottom:0.75rem; margin-top:0.75rem;",
        ),
        # Optuna-tuned
        _doc_row(
            Div(
                H4("Optuna Hyperparameter Tuning", style=f"color:{_IMMACULATA};"),
                P("All three tree-based models (RF, XGBoost, LightGBM) were tuned via Optuna with "
                  "20 trials each, using TimeSeriesSplit cross-validation to respect temporal ordering. "
                  "Balanced class weights applied to all models to address ATM dominance."),
                P("Random Forest: 50.0% accuracy / 0.338 macro F1. XGBoost: 50.4% / 0.342. "
                  "LightGBM: 48.4% / 0.349. Tuning provided only marginal improvement over baselines — "
                  "all three remained near random-level F1 (0.333) on time-split evaluation.", cls="mt-2"),
                P("LightGBM achieved the highest macro F1 despite lowest raw accuracy, indicating better "
                  "minority-class recall. Key tuned hyperparameters: n_estimators=300, max_depth=8, "
                  "learning_rate=0.05, num_leaves=50, subsample=0.8. Top features shifted to volatility "
                  "measures: vol_63d, vol_21d, bb_width.", cls="mt-2"),
            ),
            _doc_fig("improved_confusion_matrices.png", "Tuned RF, XGB, and LGBM confusion matrices — LGBM shows best minority-class balance."),
        ),
        # Walk-forward (production)
        _doc_row(
            Div(
                H4("Walk-Forward Validation (Production)", style=f"color:{_IMMACULATA};"),
                P("The critical breakthrough came from walk-forward annual retraining: for each year N, "
                  "the model trains on all data from years 1 to N-1 and predicts year N. This expanding "
                  "window approach eliminates future leakage and simulates real-world deployment conditions."),
                P("Eight implied volatility features were added at this stage (IV mean, median, skew, "
                  "short/long-term IV, term structure, IV rank, IV change), computed from the full options "
                  "board. These immediately dominated the feature importance rankings.", cls="mt-2"),
                P("Overall macro F1 = 0.468 — a significant jump from the 0.349 of the tuned model. "
                  "Strong ATM recall (13,472 correct), OTM10 reasonably separated (1,850 of 4,557). "
                  "OTM5 remains the hardest class to predict.", cls="mt-2"),
                P("Maturity rule: IV rank > 0.5 → sell SHORT (capture elevated premium quickly), "
                  "IV rank ≤ 0.5 → sell LONG (collect more time value). This decoupled approach avoids "
                  "the curse of dimensionality that hampered 7-class joint prediction.", cls="mt-2"),
            ),
            _doc_fig("walkforward_confusion_matrix.png", "Walk-forward confusion matrix — IV features enabled a significant F1 jump."),
        ),
        _doc_row(
            P("Walk-forward F1 varies by year — peaks at 0.60 in 2019, sustained above 0.50 from "
              "2015-2021, dips to ~0.32 in 2023 (a regime-shift year with rapid rate hikes). "
              "The model beats the random baseline (0.333) in most years. IV features (iv_change, "
              "iv_rank, iv_mean, iv_std) consistently occupy the top 4 importance positions across "
              "all walk-forward folds."),
            _doc_fig("walkforward_yearly_f1.png", "Macro F1 by year — consistently above random baseline, with year-to-year variation."),
        ),
        # LSTM 3-class (also explored in this pipeline)
        _doc_row(
            Div(
                H4("LSTM on 3-Class (Explored)", style=f"color:{_IMMACULATA};"),
                P("A bidirectional LSTM with temporal attention was also evaluated on the 3-class target "
                  "using 60-day lookback sequences. Walk-forward annual retraining applied."),
                P("Overall macro F1 = 0.411. Competitive with LGBM in certain years (2017, 2020) "
                  "but less stable — degrades more sharply in recent years (2022-2025) where LGBM "
                  "maintains better consistency. Selected LGBM for production due to stability, "
                  "faster inference, and simpler deployment.", cls="mt-2"),
            ),
            _doc_fig("lstm_vs_lgbm_walkforward.png", "LSTM vs LGBM walk-forward F1 by year — both on 3-class target."),
        ),
    )


def _docs_dl_pipeline():
    """Section 6: Deep learning pipeline — XGBoost baseline, LSTM-CNN, PatchTST on 7-class."""
    return _doc_section("doc-dl-pipeline", "Deep Learning Pipeline (7-Class)",
        P("This pipeline tackles the full 7-class moneyness-maturity target (ATM_30, ATM_60, ATM_90, "
          "OTM5_30, OTM5_60_90, OTM10_30, OTM10_60_90), predicting both strike and expiry jointly. "
          "Models process 50-day sliding window sequences of the top 35 features selected by "
          "Random Forest importance. Train/val/test split: pre-2022 / 2022-2023 / 2024+. "
          "Three architectures were evaluated: XGBoost (baseline), LSTM-CNN with attention, "
          "and PatchTST transformer.", cls="mb-2"),
        # XGBoost baseline
        _doc_row(
            Div(
                H4("XGBoost Baseline (7-Class)", style=f"color:{_IMMACULATA};"),
                P("XGBoost classifier with Optuna hyperparameter tuning and partial class resampling "
                  "on the 7-class target. 225 estimators, max_depth=3, learning_rate=0.049. "
                  "Multi-class softmax objective with early stopping (50 rounds)."),
                P("Test accuracy: 25.5% / Macro F1: 0.117 / Balanced accuracy: 16.7%. "
                  "The model struggles with the granularity of 7 classes — severe class imbalance "
                  "causes predictions to concentrate on dominant buckets (ATM_30, ATM_60). "
                  "Minority classes (OTM10_60_90, OTM5_30) receive almost no predictions.", cls="mt-2"),
                P("This established the difficulty of the 7-class problem and motivated the "
                  "exploration of sequence-based deep learning architectures.", cls="mt-2"),
            ),
            _doc_fig("model_comparison.png", "7-class model comparison — XGBoost baseline performance."),
        ),
        # LSTM-CNN architecture
        _doc_row(
            Div(
                H4("LSTM-CNN + Bahdanau Attention", style=f"color:{_IMMACULATA};"),
                P("Hybrid deep learning architecture combining two parallel branches that capture "
                  "different aspects of the temporal input signal:"),
                P(Strong("CNN Branch"), " — Two-layer 1D convolution for local temporal pattern extraction. "
                  "Conv1d(35→128, kernel=7) → BatchNorm → ReLU → Dropout → second Conv1d → "
                  "AdaptiveAvgPool1d(1). Captures short-range features like momentum shifts and "
                  "volatility spikes within the 50-day window.", cls="mt-1"),
                P(Strong("BiLSTM Branch"), " — Bidirectional LSTM (2 layers, hidden=128) processes "
                  "the full sequence and outputs a context-weighted representation via Bahdanau "
                  "temporal attention. Captures long-range dependencies like trend reversals "
                  "and regime transitions.", cls="mt-1"),
                P(Strong("Fusion"), " — CNN output (128-dim) concatenated with LSTM attention output "
                  "(256-dim) → LayerNorm → Dropout → FC(384→192) → ReLU → Dropout → FC(192→7). "
                  "The fused representation benefits from both local and global temporal context.", cls="mt-1"),
                P("Hyperparameters tuned via Optuna: cnn_out=128, kernel=7, lstm_hidden=128, "
                  "lstm_layers=2, attn_dim=128, batch_size=32. Three training variants explored "
                  "with different regularization strengths.", cls="mt-2"),
            ),
            _doc_fig("lstm_training_curves.png", "LSTM-CNN training curves — validation loss monitored for early stopping."),
        ),
        # LSTM-CNN results
        _doc_row(
            Div(
                H4("LSTM-CNN Experiment Results", style=f"color:{_IMMACULATA};"),
                P("20 experiment runs tracked in MLflow across three model variants, each with "
                  "confusion matrices, ROC curves, and model checkpoints:"),
                P(Strong("Regularised"), " (dropout=0.5, weight_decay=0.01): 38.1% accuracy / 0.110 macro F1. "
                  "Heavy regularization improved generalization over the base model — the highest "
                  "test F1 of any deep learning variant.", cls="mt-1"),
                P(Strong("Best"), " (dropout=0.155, standard training): 24.5% accuracy / 0.091 macro F1. "
                  "Lower dropout led to overfitting on the training distribution.", cls="mt-1"),
                P(Strong("Full"), " (complete architecture, no checkpoint selection): saved for "
                  "deployment flexibility and MLflow registry integration.", cls="mt-1"),
                P("The 7-class target space proved challenging for the available data volume (~1,300 "
                  "monthly decision points). The model learns meaningful structure in the dominant "
                  "classes (ATM_30, ATM_60) but cannot reliably distinguish all 7 buckets.", cls="mt-2"),
            ),
            _doc_fig("comparison_confusion_matrices.png", "LSTM-CNN confusion matrices across training variants."),
        ),
        # PatchTST
        _doc_row(
            Div(
                H4("PatchTST Transformer", style=f"color:{_IMMACULATA};"),
                P("Patch-based time series transformer (HuggingFace PatchTSTForClassification) using "
                  "100-day sliding window sequences. The architecture divides input sequences into "
                  "fixed-length patches and applies multi-head self-attention across patches — "
                  "a technique adapted from Vision Transformers for time series."),
                P("Walk-forward annual retraining applied (same protocol as the tree-based pipeline). "
                  "Two variants explored: base model and a pretrained version fine-tuned with "
                  "FRED macro features (federal funds rate, unemployment, yield curve, VIX).", cls="mt-2"),
                P("Base PatchTST: 14.4% accuracy / 0.086 macro F1. Pretrained + FRED variant: "
                  "no measurable improvement. Transformers require substantially more data to "
                  "learn effective attention patterns — with ~1,300 decision points, the "
                  "self-attention mechanism overfits to spurious temporal correlations.", cls="mt-2"),
            ),
            _doc_fig("patchtst_walkforward_confusion_matrix.png", "PatchTST walk-forward confusion matrix — inconsistent class predictions."),
        ),
        _doc_row(
            P("PatchTST F1 by year shows high variance across evaluation windows, with performance "
              "oscillating between near-random and moderately above baseline. The model struggles "
              "to maintain stability across different market regimes (bull/bear/sideways)."),
            _doc_fig("patchtst_walkforward_yearly_f1.png", "PatchTST walk-forward F1 by year — high variance, unstable across regimes."),
        ),
        # Deployment — no figure available for the DL deployment specifically
        Div(
            Div(
                H4("Deployment & MLflow Integration", style=f"color:{_IMMACULATA};"),
                P("The LSTM-CNN model is independently deployed via Docker Compose on AWS EC2 with "
                  "three containerized services: Streamlit frontend (:8501) for batch CSV prediction, "
                  "FastAPI inference server (:8000) with /predict and /predict/csv endpoints, and "
                  "MLflow tracking server (:5000) for model registry and experiment management."),
                P("Model loading follows a priority chain: MLflow Model Registry "
                  "(models:/CoveredCallLSTMCNN/Champion) → local file fallback "
                  "(lstm_cnn_best_model.pth). Infrastructure provisioned via Terraform "
                  "on AWS EC2 t3.medium.", cls="mt-2"),
            ),
            style=f"border-bottom:1px solid {_TORERO}; padding-bottom:0.75rem; margin-top:0.75rem;",
        ),
    )


def _docs_strategy():
    """Section 7: Post-inference strategy — scoring engine, allocation, backtesting."""
    return _doc_section("doc-strategy", "Strategy & Post-Inference",
        P("The model predicts a moneyness bucket, but the deployed system adds a scoring and "
          "allocation layer between the model output and the trading decision. This layer ranks "
          "the 10-ticker universe each month, selects which positions to take, and sizes them "
          "according to configurable presets.", cls="mb-2"),
        # Scoring engine
        Div(
            H4("Composable Scoring Engine", style=f"color:{_IMMACULATA};"),
            P("Each ticker receives a composite score from three weighted components:"),
            P(Strong("1. Model Confidence"), " — the LGBM prediction probability for the chosen bucket. "
              "Higher confidence = the model sees a clearer signal for this ticker-month.", cls="mt-1"),
            P(Strong("2. Transaction Cost Score"), " — computed from the bid-ask spread of matching "
              "options contracts. Tighter spreads = lower cost = higher score. A turnover penalty "
              "doubles the cost if the bucket changed from the previous month (discouraging churn).", cls="mt-1"),
            P(Strong("3. Delta-Hedged Return Score"), " — monthly approximation of Bali et al. (2023). "
              "Isolates the volatility premium by removing directional stock exposure: "
              "DH_gain = option_pnl - delta * stock_move - financing_cost. Higher = more pure "
              "vol premium available.", cls="mt-1"),
            style=f"border-bottom:1px solid {_TORERO}; padding-bottom:0.75rem; margin-top:0.75rem;",
        ),
        # Presets
        Div(
            H4("Strategy Presets", style=f"color:{_IMMACULATA};"),
            P("Three presets control the weight given to each scoring component and "
              "the number of positions taken:"),
            _doc_columns(
                Div(
                    P(Strong("Conservative"), cls="mt-1"),
                    P("Weights: Confidence 30%, TC 50%, Delta-Hedge 20%", cls=TextPresets.muted_sm),
                    P("Max positions: 7 (wide diversification)", cls=TextPresets.muted_sm),
                    P("Sizing: Equal weight across positions", cls=TextPresets.muted_sm),
                    P("Philosophy: Spread capital wide, prioritize low-cost positions.", cls="mt-1"),
                    P(Strong("Balanced"), cls="mt-3"),
                    P("Weights: Confidence 33%, TC 33%, Delta-Hedge 34%", cls=TextPresets.muted_sm),
                    P("Max positions: 5", cls=TextPresets.muted_sm),
                    P("Sizing: Equal weight", cls=TextPresets.muted_sm),
                    P("Philosophy: Equal consideration to all signals.", cls="mt-1"),
                ),
                Div(
                    P(Strong("Aggressive"), cls="mt-1"),
                    P("Weights: Confidence 60%, TC 10%, Delta-Hedge 30%", cls=TextPresets.muted_sm),
                    P("Max positions: 3 (concentrated bets)", cls=TextPresets.muted_sm),
                    P("Sizing: Proportional to composite score", cls=TextPresets.muted_sm),
                    P("Philosophy: Chase the model's strongest convictions, tolerate higher costs.", cls="mt-1"),
                    P(Strong("Baseline (No Model)"), cls="mt-3"),
                    P("Always sell 10% OTM short-dated calls on all tickers, equal weight.", cls=TextPresets.muted_sm),
                    P("No scoring, no selection. Pure benchmark for comparison.", cls="mt-1"),
                ),
            ),
            style=f"border-bottom:1px solid {_TORERO}; padding-bottom:0.75rem; margin-top:0.75rem;",
        ),
        # Backtesting results
        _doc_row(
            P("Annual returns breakdown shows strategy performance across different market regimes. "
              "The model-guided strategies (Argmax, Risk-Adjusted) track close to Always-ATM in "
              "bull years (2017, 2019) and show differentiation during transitional periods. "
              "The Oracle (perfect foresight) confirms headroom exists for stronger classifiers."),
            _doc_fig("annual_returns_comparison.png", "Annual covered call returns by strategy — model vs static vs Oracle."),
        ),
        _doc_row(
            P("Equity curves show cumulative portfolio growth across all strategies. "
              "Model-Argmax and Model-Risk-Adj deliver similar trajectories, both outperforming "
              "static OTM strategies but trailing Always-ATM in sustained bull markets. "
              "The strategies diverge most in volatile periods (2020, 2022) where bucket "
              "selection has the largest impact on realized returns."),
            _doc_fig("equity_curves.png", "Equity curves — cumulative growth across all strategy variants."),
        ),
    )


def _docs_results():
    """Section 8: Side-by-side model results comparison of both pipelines."""
    return _doc_section("doc-results", "Results",
        P("Both pipelines evaluated on held-out test data with honest temporal splits. "
          "The two approaches target different problem formulations, making direct F1 comparison "
          "across pipelines misleading — but within each pipeline, the progression from baseline "
          "to final model tells a clear story.", cls="mb-2"),
        # Side-by-side metrics
        _doc_columns(
            # Left: Tree-based pipeline results
            Div(
                H4("Tree-Based (3-Class)", style=f"color:{_IMMACULATA};"),
                P(Strong("Production Model: LightGBM Walk-Forward")),
                Div(
                    P(Strong("Macro F1"), cls=TextPresets.muted_sm), P("0.468"),
                    P(Strong("Test Accuracy"), cls=TextPresets.muted_sm), P("63.0% (2025 held-out)"),
                    P(Strong("Top-2 Accuracy"), cls=TextPresets.muted_sm), P("85.0%"),
                    P(Strong("Validation"), cls=TextPresets.muted_sm), P("Walk-forward annual"),
                    cls="mt-2",
                ),
                P(Strong("Model Progression:"), cls="mt-3"),
                P("RF Baseline → 0.333 F1", cls=TextPresets.muted_sm),
                P("XGB Baseline → 0.359 F1", cls=TextPresets.muted_sm),
                P("LGBM Tuned → 0.349 F1", cls=TextPresets.muted_sm),
                P("LGBM + IV + Walk-Forward → 0.468 F1", cls=TextPresets.muted_sm),
                P("Key insight: IV features + walk-forward validation provided the "
                  "largest performance jump (+0.12 F1). Simplifying from 7 to 3 classes "
                  "made the problem tractable for tree-based models.", cls="mt-2"),
            ),
            # Right: DL pipeline results
            Div(
                H4("Deep Learning (7-Class)", style=f"color:{_IMMACULATA};"),
                P(Strong("Best Model: LSTM-CNN Regularised")),
                Div(
                    P(Strong("Macro F1"), cls=TextPresets.muted_sm), P("0.110"),
                    P(Strong("Test Accuracy"), cls=TextPresets.muted_sm), P("38.1%"),
                    P(Strong("MLflow Runs"), cls=TextPresets.muted_sm), P("20 tracked experiments"),
                    P(Strong("Validation"), cls=TextPresets.muted_sm), P("Time-based 80/20"),
                    cls="mt-2",
                ),
                P(Strong("Model Progression:"), cls="mt-3"),
                P("XGB 7-Class → 0.117 F1", cls=TextPresets.muted_sm),
                P("LSTM-CNN Best → 0.091 F1", cls=TextPresets.muted_sm),
                P("LSTM-CNN Regularised → 0.110 F1", cls=TextPresets.muted_sm),
                P("PatchTST → 0.086 F1", cls=TextPresets.muted_sm),
                P("Key insight: The 7-class joint prediction is fundamentally harder. "
                  "Regularization helped (0.091 → 0.110), but the data volume (~1,300 "
                  "monthly points) limits what sequence models can learn.", cls="mt-2"),
            ),
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


def _docs_sidebar(sections):
    """Sticky left sidebar with section links for docs navigation.

    Uses uk-scrollspy-nav so the active section highlights on scroll.
    """
    return Div(
        Ul(
            *[Li(A(name, href=f"#{sid}", style=f"color:{_FOUNDERS}; text-decoration:none; "
                   "display:block; padding:0.4rem 0.75rem; border-radius:4px; font-size:0.9rem;"))
              for sid, name in sections],
            uk_scrollspy_nav="closest: li; scroll: true; offset: 80",
            cls="uk-nav uk-nav-default",
            style="list-style:none; padding:0;",
        ),
        style=f"position:sticky; top:80px; padding-top:1rem; "
              f"border-right:1px solid {_TORERO}40; padding-right:1rem; min-height:300px;",
    )


def docs_screen():
    """Documentation screen: top navbar + sticky left sidebar + scrollable content.

    Returns:
        Div with docs layout.
    """
    try:
        sections = [
            ("doc-overview", "Overview"),
            ("doc-data-pipeline", "Data Pipeline"),
            ("doc-exploratory-analysis", "Exploratory Analysis"),
            ("doc-feature-engineering", "Features"),
            ("doc-lgbm-pipeline", "Tree-Based"),
            ("doc-dl-pipeline", "Deep Learning"),
            ("doc-results", "Results"),
            ("doc-strategy", "Strategy"),
        ]
        return Div(
            _docs_navbar(sections),
            Div(
                # Left sidebar (sticky)
                Div(
                    _docs_sidebar(sections),
                    style="flex:0 0 200px; min-width:180px;",
                ),
                # Right content (scrollable)
                Div(
                    _docs_overview(),
                    _docs_data_pipeline(),
                    _docs_eda(),
                    _docs_features(),
                    _docs_lgbm_pipeline(),
                    _docs_dl_pipeline(),
                    _docs_results(),
                    _docs_strategy(),
                    style="flex:1; min-width:0;",
                ),
                style="display:flex; gap:1.5rem;",
                cls="px-4 py-4",
            ),
        )
    except Exception:
        return _fallback("documentation")
