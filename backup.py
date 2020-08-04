y1 = nyse_hist[nyse_hist.index < 2020]["Volume"]
x1 = nyse_hist[nyse_hist.index < 2020]["Close"]
y2 = nyse_hist[nyse_hist.index >= 2020]["Volume"]
x2 = nyse_hist[nyse_hist.index >= 2020]["Close"]

