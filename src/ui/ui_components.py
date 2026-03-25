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
        A("Daily Inference", href="#daily-inference"),
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
        H3("Daily Inference", style=f"color:{_FOUNDERS};"),
        P("Run the model on a single day for a given ticker.", cls=TextPresets.muted_sm),
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
def docs_screen():
    """Documentation screen placeholder: sidebar nav + content area.

    Returns:
        Div with docs layout.
    """
    try:
        sections = ["Overview", "Data Pipeline", "Feature Engineering",
                    "Model Architectures", "Backtesting", "API Reference"]
        return Div(
            # top bar with back-to-home link
            Div(
                A(
                    DivLAligned(
                        UkIcon("arrow-left", height=16, width=16),
                        Span("Back to Trader"),
                    ),
                    href="/",
                    style=f"color:{_IMMACULATA};",
                ),
                cls="px-6 py-3",
                style=f"border-bottom:1px solid {_TORERO};",
            ),
            Div(
                # sidebar nav — fixed left, scrollspy-driven
                Div(
                    H4("Documentation", style=f"color:{_FOUNDERS};"),
                    NavContainer(
                        *[Li(A(s, href=f"#doc-{s.lower().replace(' ', '-')}")) for s in sections],
                        uk_scrollspy_nav=True,
                        cls=NavT.default,
                    ),
                    cls="sticky top-20",
                    style=f"flex:1; padding-right:2rem; border-right:1px solid {_TORERO};",
                ),
                # content area — scrollable right
                Div(
                    *[Section(
                        H3(s, style=f"color:{_FOUNDERS};"),
                        P("Content coming soon.", cls=TextPresets.muted_sm),
                        DividerLine(),
                        id=f"doc-{s.lower().replace(' ', '-')}",
                    ) for s in sections],
                    style="flex:3; padding-left:2rem;",
                ),
                style="display:flex;",
                cls="px-6 py-4",
            ),
        )
    except Exception:
        return _fallback("documentation")
