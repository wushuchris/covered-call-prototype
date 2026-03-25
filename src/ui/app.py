
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
from monsterui.all import *
from src.ui.ui_components import launcher_screen, trading_screen, inference_results_card, docs_screen
from src.ui.ui_handler import handle_inference_call
from src.utils import create_logger

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
_UI_DIR = _os.path.dirname(_os.path.abspath(__file__))

app, rt = fast_app(
    pico=False,
    hdrs=(
        *Theme.blue.headers(apex_charts=True),
        usd_brand_css,
    ),
    static_path=_UI_DIR,
)


# a @rt for project root/index, should serve a launching screen, minimalistic, USD colors and .svg logo centered, with a continue button
# that sends you to the actual SPA-esque screen (not actually a SIngle Page Application in the traditional sense)
@rt("/")
def get():
    """Serve the launcher screen."""
    try:
        return launcher_screen()
    except Exception as e:
        logger.error(f"Error rendering launcher: {e}")
        return Div(P("Something went wrong.", cls="uk-text-danger"))


# @rt loads the actual trading system screen, which is a ui_components function (that is a collection of components)
@rt("/trading")
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


# @ rt that handles user input, at first it should remove all html and return a plain hello world (testing boiler plate)
@rt("/inference_call")
async def post(date: str = "", ticker: str = ""):
    """Handle inference call from the sidebar form.

    Receives date and ticker from hx-include, forwards to
    ui_handler, and returns the rendered results card.
    """
    try:
        result = await handle_inference_call(ticker=ticker, date=date)
        if "error" in result:
            return Div(P(f"Error: {result['error']}", cls="uk-text-danger"))
        return inference_results_card(result)
    except Exception as e:
        logger.error(f"Error on inference_call: {e}")
        return Div(P("Inference request failed.", cls="uk-text-danger"))


@rt("/docs")
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
