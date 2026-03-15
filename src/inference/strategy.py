# this should define the backtesting strategy, we want a shared backtest-live srategy using Nautilus trader.
# we want to keep it simple at first, simulating model inference.
# i.e we don't care about how we came to the results (we do post-inferencing stuff only)
# for placeholder info we are therefore setting up an equally weighted (each bucket gets an x % percent chance of being chosen)
# strategy. we do this by using random choice function