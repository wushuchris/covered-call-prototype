The project is locally hosted, can be reached by a publicly exposed domain (translates by means of DDNS to reverse-proxy to FastHTML request to FAST API processing... and goes back all the way to the client to render info accordingly).

We want to keep it python-based, pythonic, and simple. No JS (FastHTML), even striving for no DBs (minimal side-effects/persistence).
One-shot calls that trigger a chain of functions and return self-contained Divs to the Client. Hence different sections could actually trigger and be run
concurrently,

We want to keep it as asynchronous as possible (to don't block the event-loop/user space if not needed). Multiple simultaneous requests to a same section should be limited however. This can be achieved by temporary blocks on the user's ability to interact with certain parts of the UI (e.g. we block a text box until the
previous response has finished rendering, it always renders, because if something failed there is generic div or result that can be sent to keep the chain going).

We want modularity, a given file/section/service does not know and does not care how the next on-chain function is handling/processing the information (zero-trust). There are of course exceptions, such as inter-service communications that need a shared data structure in other to exchange information. But note how they don't really know how the other component does stuff, they simply both "coincidentally" expect information to come in shared format.

We only care about post-inference stuff, building on top of the modularity anythin lower-level (when it comes to the backtesting strategy) is not relevant to the system (for now).

No need to do OOP where it is not needed, functional programming is encouraged but if things start getting to abstract then plain function calling works as well (teammates should be able to understand the code and follow along).

Docstrings should be included on all relevant functions, and try-catch (Except) blocks as well.

We keep additional libraries to a minimum. We are using FastHTML, MonsterUI, FastAPI, NautilusTrader, and dataclasses/pydantic base models.
