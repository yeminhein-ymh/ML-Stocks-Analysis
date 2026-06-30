from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()
    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA_60"] = out["Close"].ewm(span=60, adjust=False).mean()
    out["EMA_250"] = out["Close"].ewm(span=250, adjust=False).mean()
    out["RSI"] = rsi(out["Close"])
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    out["BB_Mid"] = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_Upper"] = out["BB_Mid"] + 2 * bb_std
    out["BB_Lower"] = out["BB_Mid"] - 2 * bb_std
    out["ATR"] = atr(out)
    out["ATR_Pct"] = out["ATR"] / out["Close"] * 100
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    out["VWAP"] = (typical_price * out["Volume"]).cumsum() / out["Volume"].replace(0, np.nan).cumsum()
    out["Cumulative_VWAP"] = out["VWAP"]
    out["Avg_Volume_20"] = out["Volume"].rolling(20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Avg_Volume_20"]
    out["Dollar_Volume_20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    out["Momentum_1D"] = out["Close"].pct_change(1)
    out["Momentum_5D"] = out["Close"].pct_change(5)
    out["Momentum_20D"] = out["Close"].pct_change(20)
    out["Volatility_20D"] = out["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    out["Gap_Pct"] = (out["Open"] / out["Close"].shift(1) - 1) * 100
    return out


def add_relative_strength(df: pd.DataFrame, references: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    for symbol, ref in references.items():
        if ref.empty:
            out[f"RS_{symbol}_20D"] = np.nan
            continue
        aligned = ref["Close"].reindex(out.index).ffill()
        out[f"RS_{symbol}_20D"] = out["Close"].pct_change(20) - aligned.pct_change(20)
    return out
