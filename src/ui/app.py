
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

# a @rt for project root/index, should serve a launching screen, minimalistic, USD colors and .svg logo centered, with a continue button
# that sends you to the actual SPA-esque screen (not actually a SIngle Page Application in the traditional sense)


# @rt loads the actual trading system screen, which is a ui_components function (that is a collection of components)


# clear button @rt that should reload the trading system screen

# @ rt that handles user input, at first it should remove all html and return a plain hello world (testing boiler plate)


# if main.pyy launch the server, it is worth mentioning that the way we are going to be serving the service (there is going to be a parallel
# fast api microservice, is by means of a main.py that executes this app.py as a popen process.