
# FAST HTML SERVER, here we have the routes and app launching (if __main.py__ triggered)
# App name should be  USD Trader, port 8008

# WE ARE HARD-CORE Hypermedia Driven system builders, no JS (except for htmx by means of FastHTML/MonsterUI  functionalitiesa)
# keep .htmx simple, (Div swapping on button triggers, rarely having to resort to hw-swap-oob functionalities, but if needed lets do it)
# tricks such as on click this.reset() are encouraged, ids for swapped components should match
# we want for the components to be defined on the ui_components.py, here we only ingest user input, redirect to ui_handler(actual processing
# or inter-service communication), and ui_components render the ui_components.

# we want to keep css stylings to monsterUI standards, no need for crazy stuff beyond flex width height, margin, padding, text wrapping stuff.

# ui_utils handle serialization, data handling, object handling, decorators stuff. there is an external utils file for project wide utils (loggers, decorators)

# lets begin with a CRUD we actually need

# @rts should have try except blocks that allow for failure to retrieves to let the user know there was an error (plus they should also logg to the a.py custom logger)

from datetime import date

from fasthtml.common import *
import fasthtml.common as fh
from monsterui.all import *
from src.ui.ui_components import (
    launcher_screen, trading_screen, inference_results_card,
    batch_results_card, backtest_results_card, model_performance_card,
    docs_screen, TICKERS,
)
from src.ui.ui_handler import (
    handle_inference_call, handle_batch_inference,
    handle_backtest_call, handle_model_metrics,
)

from src.utils import create_logger, log_call

logger = create_logger("ui")

# USD brand colors: Founders Blue #003b70, Immaculata Blue #0074c8, Torero Blue #75bee9
# USD brand fonts: Sofia Sans Extra Condensed (headlines), Spectral (body) — served locally from static/fonts
usd_brand_css = Style("""
    @font-face {
        font-family: 'Sofia Sans Extra Condensed';
        src: url('/static/fonts/sofia-sans-ec-600.ttf') format('truetype');
        font-weight: 600;
        font-display: swap;
    }
    @font-face {
        font-family: 'Spectral';
        src: url('/static/fonts/spectral-400.ttf') format('truetype');
        font-weight: 400;
        font-display: swap;
    }
    @font-face {
        font-family: 'Spectral';
        src: url('/static/fonts/spectral-600.ttf') format('truetype');
        font-weight: 600;
        font-display: swap;
    }
    :root { --usd-founders: #003b70; --usd-immaculata: #0074c8; --usd-torero: #75bee9; }
    h1, h2, h3, h4 { font-family: 'Sofia Sans Extra Condensed', sans-serif; }
    body, p, span, a, label, td, th { font-family: 'Spectral', serif; }
    .uk-container, .uk-section { max-width: 100% !important; width: 100% !important; }
    body { margin: 0; padding: 0; }
""")

import os as _os
from pathlib import Path as _Path
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount

_UI_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
_FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"

app, rt = fast_app(
    pico=False,
    hdrs=(
        *Theme.blue.headers(apex_charts=True),
        usd_brand_css,
    ),
    static_path=_UI_DIR,
)

# Serve reports/figures at /figures/ — mounted before catch-all, no symlinks, portable across clones
if _FIGURES_DIR.exists():
    app.routes.insert(0, Mount("/figures", app=StaticFiles(directory=str(_FIGURES_DIR))))


# a @rt for project root/index, should serve a launching screen, minimalistic, USD colors and .svg logo centered, with a continue button
# that sends you to the actual SPA-esque screen (not actually a SIngle Page Application in the traditional sense)
@rt("/")
@log_call(logger)
def get():
    """Serve the launcher screen."""
    try:
        return launcher_screen()
    except Exception as e:
        logger.error(f"Error rendering launcher: {e}")
        return Div(P("Something went wrong.", cls="uk-text-danger"))


# @rt loads the actual trading system screen, which is a ui_components function (that is a collection of components)
@rt("/trading")
@log_call(logger)
def get():
    """Serve the trading system screen."""
    try:
        return trading_screen()
    except Exception as e:
        logger.error(f"Error rendering trading screen: {e}")
        return Div(P("Error loading trading screen.", cls="uk-text-danger"))


# clear button @rt that should reload the trading system screen
@rt("/clear")
def get():
    """Clear and reload the trading screen."""
    try:
        return trading_screen()
    except Exception as e:
        logger.error(f"Error clearing trading screen: {e}")
        return Div(P("Error reloading.", cls="uk-text-danger"))


# pure htmx today-date swap — checkbox triggers this, replaces the date input with today's value pre-filled
@rt("/today_date")
def get():
    """Return a date input pre-filled with today's date (outerHTML swap)."""
    try:
        return Input(type="date", name="date", id="inference-date", value=date.today().isoformat())
    except Exception as e:
        logger.error(f"Error on today_date: {e}")
        return Input(type="date", name="date", id="inference-date")


# toggle ticker dropdown enabled/disabled based on batch checkbox
@rt("/toggle_ticker")
def get(batch: str = ""):
    """Return enabled or disabled ticker dropdown based on batch checkbox state."""
    if batch:
        return Select(
            *[Option(t, value=t) for t in TICKERS],
            name="ticker", id="inference-ticker",
            disabled=True, style="opacity:0.5;",
        )
    return Select(
        *[Option(t, value=t) for t in TICKERS],
        name="ticker", id="inference-ticker",
    )


@rt("/ticker_chart")
async def get(ticker: str = "", date: str = ""):
    """Lazy-load a candlestick chart for a given ticker and date.

    Called via hx-trigger='intersect once' when a batch modal becomes visible.
    Fetches OHLC data from the inference service and returns an ApexChart.
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:8009/chart_data?ticker={ticker}&date={date}"
            async with session.get(url) as resp:
                data = await resp.json()

        chart_data = data.get("chart_data", [])
        if chart_data:
            return ApexChart(opts={
                "chart": {"type": "candlestick", "height": 350},
                "series": [{"name": ticker, "data": chart_data}],
                "xaxis": {"type": "datetime"},
                "yaxis": {"tooltip": {"enabled": True}},
                "plotOptions": {"candlestick": {
                    "colors": {"upward": "#0074c8", "downward": "#003b70"},
                }},
            })
        return P("No chart data available.", style="color:#999; text-align:center;")
    except Exception as e:
        logger.error(f"Error on ticker_chart: {e}")
        return P("Chart load failed.", style="color:#c62828;")


@rt("/inference_call")
@log_call(logger)
async def post(date: str = "", ticker: str = "", batch: str = ""):
    """Handle inference call from the sidebar form.

    If batch checkbox is checked, runs inference for all tickers.
    Otherwise runs single-ticker inference.
    """
    try:
        if batch:
            result = await handle_batch_inference(date=date)
            if "error" in result:
                return Div(P(f"Error: {result['error']}", cls="uk-text-danger"))
            return batch_results_card(result)
        else:
            result = await handle_inference_call(ticker=ticker, date=date)
            if "error" in result:
                return Div(P(f"Error: {result['error']}", cls="uk-text-danger"))
            return inference_results_card(result)
    except Exception as e:
        logger.error(f"Error on inference_call: {e}")
        return Div(P("Inference request failed.", cls="uk-text-danger"))


@rt("/toggle_model_year")
def get(sample_type: str = "all"):
    """Enable/disable year dropdown based on sample type selection.

    When Train or Test is selected, year is irrelevant — gray it out.
    """
    year_options = [fh.Option("All Years", value="all", selected=True)] + [
        fh.Option(str(y), value=str(y)) for y in range(2008, 2026)
    ]
    disabled = sample_type in ("train", "test")
    return fh.Select(
        *year_options, name="year", id="model-year", cls="uk-select",
        disabled=disabled, style="opacity:0.5;" if disabled else "",
        hx_get="/toggle_model_sample",
        hx_include="#model-year",
        hx_target="#model-sample-type",
        hx_swap="outerHTML",
    )


@rt("/toggle_model_sample")
def get(year: str = "all"):
    """Enable/disable sample type dropdown based on year selection.

    When a specific year is selected, sample type is redundant — gray it out.
    """
    sample_options = [
        fh.Option("All", value="all", selected=True),
        fh.Option("Train Dataset", value="train"),
        fh.Option("Test Dataset", value="test"),
    ]
    disabled = year != "all"
    return fh.Select(
        *sample_options, name="sample_type", id="model-sample-type", cls="uk-select",
        disabled=disabled, style="opacity:0.5;" if disabled else "",
        hx_get="/toggle_model_year",
        hx_include="#model-sample-type",
        hx_target="#model-year",
        hx_swap="outerHTML",
    )


@rt("/model_metrics_call")
@log_call(logger)
async def get(sample_type: str = "all", year: str = "all"):
    """Handle model metrics request (GET for lazy load, POST for button)."""
    try:
        result = await handle_model_metrics(year=year, sample_type=sample_type)
        if "error" in result:
            return Div(P(f"Error: {result['error']}", cls="uk-text-danger"))
        return model_performance_card(result)
    except Exception as e:
        logger.error(f"Error on model_metrics_call: {e}")
        return Div(P("Model metrics request failed.", cls="uk-text-danger"))


@rt("/model_metrics_call")
@log_call(logger)
async def post(sample_type: str = "all", year: str = "all"):
    """Handle model metrics request from sidebar button."""
    try:
        result = await handle_model_metrics(year=year, sample_type=sample_type)
        if "error" in result:
            return Div(P(f"Error: {result['error']}", cls="uk-text-danger"))
        return model_performance_card(result)
    except Exception as e:
        logger.error(f"Error on model_metrics_call: {e}")
        return Div(P("Model metrics request failed.", cls="uk-text-danger"))


@rt("/backtest_call")
@log_call(logger)
async def post(year: str = "all", budget: str = "100000"):
    """Handle backtest request from the backtesting sidebar.

    Runs backtests for all 3 presets + baseline for the selected year.
    """
    try:
        budget_val = float(budget)
        result = await handle_backtest_call(year=year, budget=budget_val)
        if "error" in result:
            return Div(P(f"Error: {result['error']}", cls="uk-text-danger"))
        return backtest_results_card(result)
    except Exception as e:
        logger.error(f"Error on backtest_call: {e}")
        return Div(P("Backtest request failed.", cls="uk-text-danger"))


@rt("/docs")
@log_call(logger)
def get():
    """Serve the documentation screen."""
    try:
        return docs_screen()
    except Exception as e:
        logger.error(f"Error rendering docs: {e}")
        return Div(P("Error loading documentation.", cls="uk-text-danger"))


# if main.pyy launch the server, it is worth mentioning that the way we are going to be serving the service (there is going to be a parallel
# fast api microservice, is by means of a main.py that executes this app.py as a popen process.
if __name__ == "__main__":
    serve(port=8008)
