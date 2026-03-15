# for now the key main functions are going to be custom logger creation (we want to define a function that generates a new logger)
# that way the microservices can call this function and spawn per project loggers
# we want the logger to print to console errors only, log everything to .txt files (from INFO to ERROR), and delete logs more than a day old (we don't want to
# accumulate .txt files, just present day stuff)
# the log should be brief, date, description, file where it broke
# it is worth mentioning that INFO is mainly going to be handled by the decorator below, and errors will be handled by try catch (Exception)
# blocks, where the caught exception will simply do something like custom_logger.error(f"Error {e} on xyz. Please..."")

# we want a tool/function decorator that can be placed on  regular functions and acts as a wrapper that loggs the INFO
# date, time to completion, function name, file
# function takes care of only logging INFO for regular functions. handling generators, routes, functions where another decorator is present is handled differently
# basic log for basic functions