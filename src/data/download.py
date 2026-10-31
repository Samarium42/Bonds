"""Download weekly basket data. Run on a machine with internet:  python -m src.data.download

Writes data/raw/weekly_returns.csv with weekly (Friday) log-returns, one column per ticker.
TSLA is the target name; the others are pooled for extra training data.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

BASKET = ["TSLA", "NVDA", "AMD", "NFLX", "META", "AMZN", "AAPL", "MSFT", "GOOGL"]


def main(out: str = "data/raw/weekly_returns.csv", start: str = "2012-01-01") -> None:
    import yfinance as yf
    px = yf.download(BASKET, start=start, auto_adjust=True, progress=False)["Close"]
    wk = px.resample("W-FRI").last().dropna(how="all")
    lr = np.log(wk).diff()
    lr.to_csv(out)
    daily = np.log(px).diff()
    daily.to_csv(out.replace("weekly", "daily"))
    print(f"also saved daily: {daily.shape[0]} days")
    print(f"saved {out}: {lr.shape[0]} weeks x {lr.shape[1]} names, {lr.index[0].date()} to {lr.index[-1].date()}")
    print(lr.describe().T[["count", "std"]])


def load(path: str = "data/raw/weekly_returns.csv") -> dict:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return {c: df[c].dropna().to_numpy() for c in df.columns}


if __name__ == "__main__":
    main(*sys.argv[1:])
