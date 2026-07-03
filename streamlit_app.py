from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="Universal AI Stock Analysis Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

USING_EMBEDDED_FALLBACK = not (APP_ROOT / "app" / "core" / "analyzer.py").exists()

if not USING_EMBEDDED_FALLBACK:
    try:
        from app.core.analyzer import analyze_stock, prepare_stock, scan_stocks
        from app.core.backtesting import backtest_portfolio, backtest_signal_strategy
        from app.core.config import DEFAULT_WATCHLIST, RISK_RULES, SIGNAL_LOG_FILE, TRADE_LOG_FILE
        from app.core.data import fetch_history, fetch_last_prices, market_data_healthcheck, normalize_tickers
        from app.core.modeling import train_models
        from app.core.paper_trading import cancel_option_order, cancel_order, log_signal, open_option_orders_table, open_orders_table, option_order_requirement, place_option_paper_order, place_paper_order, position_size
        from app.core.screener import COMMON_UNIVERSE, screen_universe
        from app.core.watchlists import load_watchlists, parse_uploaded_watchlist, save_watchlist
    except ModuleNotFoundError:
        USING_EMBEDDED_FALLBACK = True

if USING_EMBEDDED_FALLBACK:
    import json
    import re
    from datetime import datetime

    import numpy as np
    import requests

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
        from alpaca.common.exceptions import APIError
    except Exception:
        TradingClient = None
        OrderSide = None
        QueryOrderStatus = None
        TimeInForce = None
        GetOrdersRequest = None
        LimitOrderRequest = None
        MarketOrderRequest = None
        APIError = Exception

    DEFAULT_WATCHLIST = [
        "NVDA", "AMZN", "META", "GOOGL", "PLTR", "MU", "SNDK", "NBIS",
        "AAPL", "MSFT", "TSLA", "AMD", "AVGO", "SMCI", "NFLX", "ARM",
        "TSM", "ORCL", "CRM", "QCOM", "INTC", "COIN", "MSTR", "SOFI",
        "HOOD", "JPM", "BAC", "XOM", "NEE", "ENPH", "FSLR",
    ]
    RISK_RULES = {
        "max_risk_per_trade": 0.02,
        "max_exposure_per_stock": 0.20,
        "max_total_exposure": 0.80,
        "min_reward_to_risk": 2.0,
    }
    SIGNAL_LOG_FILE = APP_ROOT / "logs" / "signals.csv"
    TRADE_LOG_FILE = APP_ROOT / "logs" / "paper_trades.csv"
    WATCHLIST_FILE = APP_ROOT / "custom_watchlists.json"
    COMMON_UNIVERSE = sorted(set(DEFAULT_WATCHLIST + [
        "ABBV", "ABNB", "ADBE", "AMAT", "BA", "CAT", "COST", "CVX", "DIS",
        "GE", "GS", "HD", "IBM", "LLY", "MA", "MRVL", "NOW", "PANW", "PEP",
        "PYPL", "SHOP", "SNOW", "UBER", "UNH", "V", "WFC", "WMT",
    ]))

    def normalize_ticker(ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9.\-]", "", str(ticker).upper().strip())[:12]

    def normalize_tickers(tickers) -> list[str]:
        seen = set()
        output = []
        for ticker in tickers:
            clean = normalize_ticker(ticker)
            if clean and clean not in seen:
                seen.add(clean)
                output.append(clean)
        return output

    @st.cache_data(ttl=300, show_spinner=False)
    def _yahoo_chart(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        response = requests.get(
            url,
            params={"range": period, "interval": interval, "includePrePost": "false"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        result = (response.json().get("chart", {}).get("result") or [None])[0]
        if not result:
            return pd.DataFrame()
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [None])[0] or {}
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [None])[0] or {}
        if not timestamps or not quote:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Adj Close": adjclose.get("adjclose", quote.get("close")),
                "Volume": quote.get("volume"),
            },
            index=pd.to_datetime(timestamps, unit="s"),
        )
        frame.index = frame.index.tz_localize(None)
        return frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        ticker = normalize_ticker(ticker)
        if not ticker:
            return pd.DataFrame()
        try:
            data = _yahoo_chart(ticker, period, interval)
        except Exception:
            data = pd.DataFrame()
        if data.empty:
            try:
                import yfinance as yf

                data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.rename(columns=str.title)
            except Exception:
                data = pd.DataFrame()
        return data.dropna(subset=["Open", "High", "Low", "Close", "Volume"]) if not data.empty else data

    def market_data_healthcheck() -> str:
        return "ok" if not fetch_history("AAPL", "5d").empty else "Market data provider returned no data. Check internet access or Yahoo Finance availability."

    def fetch_last_prices(tickers) -> pd.DataFrame:
        rows = []
        for ticker in normalize_tickers(tickers):
            hist = fetch_history(ticker)
            if hist.empty:
                continue
            last = hist.iloc[-1]
            prev = hist["Close"].iloc[-2] if len(hist) > 1 else last["Close"]
            rows.append({"Ticker": ticker, "Price": float(last["Close"]), "Daily Change %": float((last["Close"] / prev - 1) * 100), "Volume": int(last["Volume"])})
        return pd.DataFrame(rows)

    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        true_range = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    def prepare_stock(ticker: str) -> pd.DataFrame:
        out = fetch_history(ticker).copy()
        if out.empty:
            return out
        out["SMA_20"] = out["Close"].rolling(20).mean()
        out["SMA_50"] = out["Close"].rolling(50).mean()
        out["SMA_200"] = out["Close"].rolling(200).mean()
        out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
        out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()
        out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
        out["EMA_60"] = out["Close"].ewm(span=60, adjust=False).mean()
        out["EMA_250"] = out["Close"].ewm(span=250, adjust=False).mean()
        out["RSI"] = _rsi(out["Close"])
        ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = ema_12 - ema_26
        out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
        out["BB_Mid"] = out["Close"].rolling(20).mean()
        bb_std = out["Close"].rolling(20).std()
        out["BB_Upper"] = out["BB_Mid"] + 2 * bb_std
        out["BB_Lower"] = out["BB_Mid"] - 2 * bb_std
        out["ATR"] = _atr(out)
        out["ATR_Pct"] = out["ATR"] / out["Close"] * 100
        typical = (out["High"] + out["Low"] + out["Close"]) / 3
        out["VWAP"] = (typical * out["Volume"]).cumsum() / out["Volume"].replace(0, np.nan).cumsum()
        out["Cumulative_VWAP"] = out["VWAP"]
        out["Avg_Volume_20"] = out["Volume"].rolling(20).mean()
        out["Volume_Ratio"] = out["Volume"] / out["Avg_Volume_20"]
        out["Dollar_Volume_20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
        out["Momentum_1D"] = out["Close"].pct_change(1)
        out["Momentum_5D"] = out["Close"].pct_change(5)
        out["Momentum_20D"] = out["Close"].pct_change(20)
        out["Volatility_20D"] = out["Close"].pct_change().rolling(20).std() * np.sqrt(252)
        out["Gap_Pct"] = (out["Open"] / out["Close"].shift(1) - 1) * 100
        for ref in ["SPY", "QQQ"]:
            ref_hist = fetch_history(ref)
            aligned = ref_hist["Close"].reindex(out.index).ffill() if not ref_hist.empty else pd.Series(index=out.index, dtype=float)
            out[f"RS_{ref}_20D"] = out["Close"].pct_change(20) - aligned.pct_change(20)
        return out

    def _suitability(df: pd.DataFrame, allow_penny_stocks: bool = False) -> dict:
        if df.empty:
            return {"Suitable": False, "Score": 0, "Reasons": ["No market data available"]}
        row = df.dropna().iloc[-1] if not df.dropna().empty else df.iloc[-1]
        checks = {
            "Average daily volume > 1M": df["Volume"].tail(20).mean() > 1_000_000,
            "Price above $5": row["Close"] > 5 or allow_penny_stocks,
            "ATR percentage > 1.5%": row.get("ATR_Pct", 0) > 1.5,
            "20-day dollar volume > $50M": (df["Close"] * df["Volume"]).tail(20).mean() > 50_000_000,
            "Minimum 1 year history": len(df) >= 252,
        }
        return {"Suitable": all(checks.values()), "Score": round(sum(checks.values()) / len(checks) * 100), "Reasons": [k for k, v in checks.items() if not v] or ["Passes all suitability filters"]}

    def _ai_prob(row: pd.Series) -> dict:
        bullish_checks = [
            row["Close"] > row["VWAP"], row["Close"] > row["EMA_20"], row["EMA_20"] > row["EMA_50"],
            45 <= row["RSI"] <= 70, row["MACD"] > row["MACD_Signal"], row["Volume_Ratio"] > 1,
            row["ATR_Pct"] > 1.5, row.get("RS_SPY_20D", 0) > 0,
        ]
        bearish_checks = [
            row["Close"] < row["VWAP"], row["Close"] < row["EMA_20"], row["RSI"] < 40 or row["RSI"] > 75,
            row["MACD"] < row["MACD_Signal"], row["Volume_Ratio"] < 0.8,
        ]
        return {"bullish_probability": round(min(92, 30 + sum(bullish_checks) * 7.5), 1), "bearish_probability": round(min(90, 25 + sum(bearish_checks) * 11), 1), "confidence": "Medium"}

    def _risk(row: pd.Series) -> dict:
        atr = float(row.get("ATR", 0) or 0)
        close = float(row["Close"])
        stop = max(0.01, close - 1.5 * atr)
        target = close + 3.0 * atr
        return {"buy_zone": close, "stop_loss": stop, "take_profit": target, "risk_reward_ratio": (target - close) / max(0.01, close - stop)}

    def analyze_stock(ticker: str, allow_penny_stocks: bool = False) -> dict:
        ticker = normalize_ticker(ticker)
        df = prepare_stock(ticker)
        if df.empty:
            return {"Ticker": ticker, "Error": "No data found"}
        row = df.dropna().iloc[-1]
        suitability = _suitability(df, allow_penny_stocks)
        ai = _ai_prob(row)
        risk = _risk(row)
        buy_rules = [
            suitability["Suitable"], ai["bullish_probability"] > 60, row["Close"] > row["VWAP"],
            row["Close"] > row["EMA_20"], row["EMA_20"] > row["EMA_50"], 45 <= row["RSI"] <= 70,
            row["MACD"] > row["MACD_Signal"], row["Volume"] > row["Avg_Volume_20"],
            row["ATR_Pct"] > 1.5, risk["risk_reward_ratio"] >= 2,
        ]
        sell_rules = [
            ai["bearish_probability"] > 60, row["Close"] < row["VWAP"], row["Close"] < row["EMA_20"],
            row["RSI"] < 40 or row["RSI"] > 75, row["MACD"] < row["MACD_Signal"], row["Volume_Ratio"] < 0.8,
            not suitability["Suitable"], risk["risk_reward_ratio"] < 2,
        ]
        final = "Avoid" if not suitability["Suitable"] else "Strong Buy" if sum(buy_rules) >= 9 else "Buy" if sum(buy_rules) >= 7 else "Sell" if sum(sell_rules) >= 5 else "Hold"
        return {
            "Ticker": ticker, "Price": float(row["Close"]), "Daily Change %": float(row["Close"] / df["Close"].iloc[-2] - 1) * 100,
            "Volume": int(row["Volume"]), "RSI": float(row["RSI"]), "MACD signal": "Bullish" if row["MACD"] > row["MACD_Signal"] else "Bearish",
            "VWAP status": "Above VWAP" if row["Close"] > row["VWAP"] else "Below VWAP",
            "Trend": "Uptrend" if row["EMA_20"] > row["EMA_50"] and row["Close"] > row["EMA_20"] else "Downtrend" if row["Close"] < row["EMA_20"] else "Sideways",
            "AI bullish probability": ai["bullish_probability"], "AI bearish probability": ai["bearish_probability"],
            "Buy zone": risk["buy_zone"], "Stop loss": risk["stop_loss"], "Take profit": risk["take_profit"], "Risk-reward ratio": risk["risk_reward_ratio"],
            "Suitability score": suitability["Score"], "Suitability": "Suitable" if suitability["Suitable"] else "Not Suitable",
            "Suitability reasons": "; ".join(suitability["Reasons"]), "Final signal": final,
            "Data": df, "Suitability detail": suitability, "AI detail": ai,
        }

    def scan_stocks(tickers: list[str], limit: int, allow_penny_stocks: bool = False) -> pd.DataFrame:
        rows = []
        for ticker in tickers[:limit]:
            result = analyze_stock(ticker, allow_penny_stocks)
            if "Error" not in result:
                rows.append({key: value for key, value in result.items() if key not in {"Data", "Suitability detail", "AI detail"}})
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        ranks = {"Strong Buy": 5, "Buy": 4, "Hold": 3, "Sell": 2, "Avoid": 1}
        table["_rank"] = table["Final signal"].map(ranks)
        return table.sort_values(["_rank", "Suitability score", "AI bullish probability"], ascending=False).drop(columns="_rank")

    def screen_universe(universe: list[str] | None = None, limit: int = 100, allow_penny_stocks: bool = False) -> pd.DataFrame:
        result = scan_stocks(universe or COMMON_UNIVERSE, limit, allow_penny_stocks)
        return result[result["Suitability"] == "Suitable"] if not result.empty else result

    def load_watchlists() -> dict[str, list[str]]:
        if "watchlists" not in st.session_state:
            st.session_state.watchlists = {"Default": DEFAULT_WATCHLIST}
        return st.session_state.watchlists

    def save_watchlist(name: str, tickers: list[str]) -> None:
        st.session_state.watchlists = load_watchlists()
        st.session_state.watchlists[name.strip() or "Custom"] = normalize_tickers(tickers)
        try:
            WATCHLIST_FILE.write_text(json.dumps(st.session_state.watchlists, indent=2), encoding="utf-8")
        except Exception:
            pass

    def parse_uploaded_watchlist(file) -> list[str]:
        df = pd.read_csv(file)
        column = next((col for col in df.columns if col.lower() in {"ticker", "symbol", "tickers", "symbols"}), df.columns[0])
        return normalize_tickers(df[column].dropna().astype(str).tolist())

    def backtest_signal_strategy(df: pd.DataFrame, transaction_cost: float = 0.001, slippage: float = 0.001) -> dict:
        data = df.dropna().copy()
        if len(data) < 80:
            return {"error": "Not enough clean history for backtest."}
        position = ((data["Close"] > data["VWAP"]) & (data["Close"] > data["EMA_20"]) & (data["EMA_20"] > data["EMA_50"]) & data["RSI"].between(45, 70) & (data["MACD"] > data["MACD_Signal"]) & (data["Volume_Ratio"] > 1)).astype(float)
        returns = position.shift(1).fillna(0) * data["Close"].pct_change().fillna(0) - position.diff().abs().fillna(0) * (transaction_cost + slippage)
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        trade_returns = returns[position.diff().abs() > 0].tolist()
        wins = [x for x in trade_returns if x > 0]
        losses = [x for x in trade_returns if x <= 0]
        return {"total_return": float(equity.iloc[-1] - 1), "win_rate": float(len(wins) / len(trade_returns)) if trade_returns else 0, "sharpe_ratio": float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() else 0, "max_drawdown": float(drawdown.min()), "profit_factor": float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0, "average_gain": float(np.mean(wins)) if wins else 0, "average_loss": float(np.mean(losses)) if losses else 0, "number_of_trades": int(position.diff().abs().sum() / 2), "equity_curve": equity, "drawdown_curve": drawdown}

    def backtest_portfolio(stock_frames: dict[str, pd.DataFrame], transaction_cost: float, slippage: float) -> dict:
        curves = {}
        stats = {}
        for ticker, frame in stock_frames.items():
            result = backtest_signal_strategy(frame, transaction_cost, slippage)
            if "error" not in result:
                curves[ticker] = result["equity_curve"]
                stats[ticker] = result
        if not curves:
            return {"error": "No symbols had enough data for portfolio backtest."}
        returns = pd.concat(curves, axis=1).pct_change().mean(axis=1).fillna(0)
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        return {"total_return": float(equity.iloc[-1] - 1), "sharpe_ratio": float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() else 0, "max_drawdown": float(drawdown.min()), "equity_curve": equity, "drawdown_curve": drawdown, "per_stock": stats}

    FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_20", "EMA_50", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Lower", "ATR", "ATR_Pct", "VWAP", "Volume_Ratio", "Momentum_1D", "Momentum_5D", "Momentum_20D", "Volatility_20D", "Gap_Pct", "RS_SPY_20D", "RS_QQQ_20D"]

    def train_models(df: pd.DataFrame) -> dict:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return {"error": "scikit-learn is not installed. Add it to requirements.txt and reboot the app."}
        data = df.copy()
        data["Future_Return_5D"] = data["Close"].shift(-5) / data["Close"] - 1
        data["Label"] = np.select([data["Future_Return_5D"] > 0.015, data["Future_Return_5D"] < -0.015], [1, -1], default=0)
        data = data.dropna(subset=[*FEATURE_COLUMNS, "Label"])
        if len(data) < 260:
            return {"error": "Not enough clean rows for time-series training."}
        x = data[FEATURE_COLUMNS]
        y = data["Label"]
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, class_weight="balanced")
        scores = []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(x_scaled):
            model.fit(x_scaled[train_idx], y.iloc[train_idx])
            scores.append(accuracy_score(y.iloc[test_idx], model.predict(x_scaled[test_idx])))
        model.fit(x_scaled, y)
        return {"models": {"Random Forest": {"model": model, "cv_accuracy": float(np.mean(scores))}}, "scaler": scaler, "features": FEATURE_COLUMNS, "rows": len(data)}

    def _secret(name: str, default: str = "") -> str:
        try:
            return st.secrets.get(name, os.getenv(name, default))
        except Exception:
            return os.getenv(name, default)

    def alpaca_client(account_type: str = "stock"):
        if TradingClient is None:
            return None
        if account_type == "options":
            key = _secret("ALPACA_OPTIONS_API_KEY") or _secret("ALPACA_API_KEY")
            secret = _secret("ALPACA_OPTIONS_SECRET_KEY") or _secret("ALPACA_SECRET_KEY")
            endpoint = (_secret("ALPACA_OPTIONS_ENDPOINT") or _secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")).removesuffix("/").removesuffix("/v2")
        else:
            key = _secret("ALPACA_API_KEY")
            secret = _secret("ALPACA_SECRET_KEY")
            endpoint = _secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2").removesuffix("/").removesuffix("/v2")
        return TradingClient(key, secret, paper=True, url_override=endpoint) if key and secret else None

    def open_orders_table(ticker: str | None = None) -> pd.DataFrame:
        client = alpaca_client()
        return _open_orders_table_for_client(client, ticker)

    def open_option_orders_table(ticker: str | None = None) -> pd.DataFrame:
        client = alpaca_client("options")
        return _open_orders_table_for_client(client, ticker)

    def _open_orders_table_for_client(client, ticker: str | None = None) -> pd.DataFrame:
        if client is None or GetOrdersRequest is None:
            return pd.DataFrame()
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker] if ticker else None)
        return pd.DataFrame([{"id": str(order.id), "symbol": order.symbol, "side": str(order.side).replace("OrderSide.", "").lower(), "qty": order.qty, "type": str(order.type).replace("OrderType.", "").lower(), "status": str(order.status).replace("OrderStatus.", "").lower(), "submitted_at": order.submitted_at} for order in client.get_orders(filter=request)])

    def cancel_order(order_id: str) -> dict:
        client = alpaca_client()
        if client is None:
            return {"ok": False, "message": "Alpaca paper API is not configured."}
        try:
            client.cancel_order_by_id(order_id)
            return {"ok": True, "message": f"Canceled paper order {order_id}."}
        except APIError as exc:
            return {"ok": False, "message": f"Could not cancel order: {exc}"}

    def cancel_option_order(order_id: str) -> dict:
        client = alpaca_client("options")
        if client is None:
            return {"ok": False, "message": "Alpaca options paper API is not configured."}
        try:
            client.cancel_order_by_id(order_id)
            return {"ok": True, "message": f"Canceled option paper order {order_id}."}
        except APIError as exc:
            return {"ok": False, "message": f"Could not cancel option order: {exc}"}

    def position_size(capital: float, entry: float, stop: float, current_exposure: float = 0) -> int:
        shares_by_risk = int((capital * RISK_RULES["max_risk_per_trade"]) / max(0.01, entry - stop))
        shares_by_exposure = int((capital * RISK_RULES["max_exposure_per_stock"] - current_exposure) / entry)
        return max(0, min(shares_by_risk, shares_by_exposure))

    def place_paper_order(ticker: str, qty: int, side: str = "buy") -> dict:
        client = alpaca_client()
        if client is None:
            return {"ok": False, "message": "Alpaca paper API is not configured."}
        if qty <= 0:
            return {"ok": False, "message": "Quantity is zero after risk controls."}
        opposite = "sell" if side.lower() == "buy" else "buy"
        open_orders = open_orders_table(ticker)
        blockers = open_orders[open_orders["side"].astype(str).str.contains(opposite, case=False, na=False)] if not open_orders.empty else pd.DataFrame()
        if not blockers.empty:
            return {"ok": False, "message": f"Open opposite-side paper order exists: {', '.join(blockers['id'].astype(str))}. Cancel it before submitting."}
        try:
            order = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
            submitted = client.submit_order(order)
            return {"ok": True, "message": f"Submitted paper {side} order for {qty} {ticker}. Alpaca id: {submitted.id}"}
        except APIError as exc:
            return {"ok": False, "message": f"Alpaca rejected the paper order: {exc}"}

    def place_option_paper_order(option_symbol: str, qty: int, limit_price: float, side: str = "buy") -> dict:
        client = alpaca_client("options")
        if client is None:
            return {"ok": False, "message": "Alpaca options paper API is not configured."}
        if LimitOrderRequest is None:
            return {"ok": False, "message": "Installed alpaca-py version does not support limit option orders."}
        if qty <= 0 or limit_price <= 0:
            return {"ok": False, "message": "Quantity and limit price must be greater than zero."}
        requirement = option_order_requirement(option_symbol, qty, limit_price, side)
        if not requirement["ok"]:
            return {"ok": False, "message": requirement["message"]}
        available = account_buying_power("options")
        if requirement["required"] > available:
            return {
                "ok": False,
                "message": (
                    f"Insufficient options buying power. Required about ${requirement['required']:,.2f}; "
                    f"available about ${available:,.2f}. For sell puts, Alpaca requires cash-secured collateral "
                    "based on strike x 100 x contracts, not the limit premium."
                ),
            }
        open_orders = open_option_orders_table(option_symbol)
        opposite = "sell" if side.lower() == "buy" else "buy"
        blockers = open_orders[open_orders["side"].astype(str).str.contains(opposite, case=False, na=False)] if not open_orders.empty else pd.DataFrame()
        if not blockers.empty:
            return {"ok": False, "message": f"Open opposite-side option order exists: {', '.join(blockers['id'].astype(str))}. Cancel it before submitting."}
        try:
            order = LimitOrderRequest(
                symbol=option_symbol.strip().upper(),
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(float(limit_price), 2),
            )
            submitted = client.submit_order(order)
            return {"ok": True, "message": f"Submitted paper option {side} limit order for {qty} {option_symbol} at ${limit_price:.2f}. Alpaca id: {submitted.id}"}
        except APIError as exc:
            return {"ok": False, "message": f"Alpaca rejected the option paper order: {exc}"}

    def log_signal(row: dict) -> None:
        try:
            SIGNAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{**{"timestamp": datetime.utcnow().isoformat()}, **row}]).to_csv(SIGNAL_LOG_FILE, mode="a", header=not SIGNAL_LOG_FILE.exists(), index=False)
        except Exception:
            pass

load_dotenv()

DAILY_ANALYSIS_FILE = APP_ROOT / "logs" / "daily_signal_analysis.csv"
SCANNER_HISTORY_FILE = APP_ROOT / "logs" / "multi_stock_ai_scanner_history.csv"
TECHNICAL_SNAPSHOT_FILE = APP_ROOT / "logs" / "technical_indicator_snapshots.csv"
DAILY_EXCEL_DIR = APP_ROOT / "logs" / "daily_excel_reports"
ARCHIVE_BACKUP_DIR = APP_ROOT / "logs" / "archive_backups"
AUTO_DAILY_SOURCE = "Auto Top 50 Daily Scanner"
RETENTION_DAYS = 365
CALENDAR_ARCHIVE_START = pd.Timestamp("2026-06-26")
CALENDAR_ARCHIVE_END = pd.Timestamp("2026-12-31")


def app_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def alpaca_rest_base(account_type: str = "stock") -> str:
    if account_type == "options":
        return (app_secret("ALPACA_OPTIONS_ENDPOINT") or app_secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")).removesuffix("/")
    return app_secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2").removesuffix("/")


def alpaca_headers(account_type: str = "stock") -> dict:
    if account_type == "options":
        key = app_secret("ALPACA_OPTIONS_API_KEY") or app_secret("ALPACA_API_KEY")
        secret = app_secret("ALPACA_OPTIONS_SECRET_KEY") or app_secret("ALPACA_SECRET_KEY")
    else:
        key = app_secret("ALPACA_API_KEY")
        secret = app_secret("ALPACA_SECRET_KEY")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def alpaca_options_data_base() -> str:
    return "https://data.alpaca.markets/v1beta1/options"


def parse_option_symbol(option_symbol: str) -> dict:
    match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", option_symbol.strip().upper())
    if not match:
        return {}
    root, expiry, option_type, strike_raw = match.groups()
    return {
        "underlying": root,
        "expiration": f"20{expiry[:2]}-{expiry[2:4]}-{expiry[4:6]}",
        "type": "call" if option_type == "C" else "put",
        "strike": int(strike_raw) / 1000,
    }


def option_order_requirement(option_symbol: str, qty: int, limit_price: float, side: str) -> dict:
    details = parse_option_symbol(option_symbol)
    if qty <= 0:
        return {"ok": False, "required": 0.0, "message": "Quantity must be greater than zero."}
    if limit_price <= 0:
        return {"ok": False, "required": 0.0, "message": "Limit price must be greater than zero."}
    if side.lower() == "buy":
        return {"ok": True, "required": qty * limit_price * 100, "message": "Long option premium required."}
    if not details:
        return {"ok": False, "required": 0.0, "message": "Could not parse option symbol for sell-side collateral check."}
    if details["type"] == "call":
        return {
            "ok": False,
            "required": 0.0,
            "message": "Selling uncovered calls is disabled in this paper bot. Use buy orders or close an existing long call manually.",
        }
    return {
        "ok": True,
        "required": details["strike"] * 100 * qty,
        "message": "Cash-secured put collateral required.",
    }


def account_buying_power(account_type: str = "stock") -> float:
    if account_type == "options":
        has_credentials = bool((app_secret("ALPACA_OPTIONS_API_KEY") or app_secret("ALPACA_API_KEY")) and (app_secret("ALPACA_OPTIONS_SECRET_KEY") or app_secret("ALPACA_SECRET_KEY")))
    else:
        has_credentials = bool(app_secret("ALPACA_API_KEY") and app_secret("ALPACA_SECRET_KEY"))
    if not has_credentials:
        return 0.0
    try:
        response = requests.get(f"{alpaca_rest_base(account_type)}/account", headers=alpaca_headers(account_type), timeout=20)
        response.raise_for_status()
        account = response.json()
    except Exception:
        return 0.0
    for key in ("options_buying_power", "buying_power", "cash"):
        if key in account and account[key] is not None:
            try:
                return float(account[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def masked_account_key(account_type: str = "stock") -> str:
    if account_type == "options":
        key = app_secret("ALPACA_OPTIONS_API_KEY") or app_secret("ALPACA_API_KEY")
    else:
        key = app_secret("ALPACA_API_KEY")
    if not key:
        return "not configured"
    return f"...{key[-6:]}"


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        .risk-note {
            border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 14px;
            background: #f8fafc; color: #0f172a; font-size: 0.94rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<div class="risk-note"><b>Education and research only.</b> This dashboard is not financial advice, does not guarantee profit, and defaults to analysis plus Alpaca paper trading only. Live trading is intentionally not implemented.</div>',
        unsafe_allow_html=True,
    )


def ticker_selector(max_default: int = 50) -> list[str]:
    watchlists = load_watchlists()
    selected_watchlist = st.sidebar.selectbox("Watchlist", list(watchlists.keys()))
    manual = st.sidebar.text_input("Add any US ticker", placeholder="Example: AAPL, MSFT, AMD")
    uploaded = st.sidebar.file_uploader("Upload CSV watchlist", type=["csv"])
    tickers = list(watchlists[selected_watchlist])
    tickers.extend(COMMON_UNIVERSE)
    if uploaded is not None:
        tickers.extend(parse_uploaded_watchlist(uploaded))
    if manual:
        tickers.extend(manual.split(","))
    tickers = normalize_tickers(tickers)
    selected = st.sidebar.multiselect("Selected stocks", tickers, default=tickers[:max_default])
    if st.sidebar.button("Save selected as custom watchlist"):
        save_watchlist(f"Custom {pd.Timestamp.now().strftime('%Y-%m-%d %H%M')}", selected)
        st.sidebar.success("Watchlist saved.")
    return selected


def scanner_universe(selected: list[str], limit: int) -> list[str]:
    universe = normalize_tickers([*selected, *COMMON_UNIVERSE, *DEFAULT_WATCHLIST])
    return universe[:limit]


def signal_direction(signal: str) -> str:
    signal = str(signal)
    if signal in {"Strong Buy", "Buy"}:
        return "Bullish"
    if signal in {"Sell", "Avoid"}:
        return "Bearish"
    return "Neutral"


def signal_synergy_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool, index=rows.index)
    rsi = pd.to_numeric(rows.get("RSI", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    macd = rows.get("MACD signal", pd.Series(index=rows.index, dtype=str)).astype(str)
    vwap = rows.get("VWAP status", pd.Series(index=rows.index, dtype=str)).astype(str)
    final_signal = rows.get("Final signal", pd.Series(index=rows.index, dtype=str)).astype(str)
    return (
        final_signal.eq("Strong Buy")
        & rsi.between(60, 75, inclusive="both")
        & macd.str.contains("Bullish", case=False, na=False)
        & vwap.str.contains("Above", case=False, na=False)
    )


def add_signal_synergy_columns(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    output = rows.copy()
    output["Synergy Strong Buy"] = signal_synergy_mask(output)
    bullish_prob = pd.to_numeric(output.get("AI bullish probability", pd.Series(index=output.index, dtype=float)), errors="coerce")
    output["AI conviction band"] = pd.cut(
        bullish_prob,
        bins=[-1, 60, 70, 82.5, 90, 100],
        labels=["Low", "Watch", "Conviction", "Strong Buy sweet spot", "Very high"],
    ).astype("object")
    output["Signal quality note"] = ""
    weak_buy = output.get("Final signal", pd.Series(index=output.index, dtype=str)).astype(str).eq("Buy") & (bullish_prob < 70)
    output.loc[weak_buy, "Signal quality note"] = "Buy signal but AI probability is below 70%"
    output.loc[output["Synergy Strong Buy"], "Signal quality note"] = "RSI/MACD/VWAP synergy confirmed"
    return output


def apply_retention(records: pd.DataFrame, date_col: str, days: int = RETENTION_DAYS) -> pd.DataFrame:
    if records.empty or date_col not in records:
        return records
    output = records.copy()
    dates = pd.to_datetime(output[date_col], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    return output[(dates.isna()) | (dates >= cutoff)]


def save_csv_archive(records: pd.DataFrame, path: Path, date_col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = apply_retention(records, date_col)
    records.to_csv(path, index=False)
    try:
        ARCHIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = ARCHIVE_BACKUP_DIR / f"{path.stem}_{pd.Timestamp.now().date().isoformat()}.csv"
        records.to_csv(backup_path, index=False)
    except Exception:
        pass


def record_daily_rows(rows: pd.DataFrame, source: str) -> int:
    if rows.empty:
        return 0
    DAILY_ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().date().isoformat()
    records = rows.copy()
    records["Record date"] = today
    records["Recorded at"] = datetime.utcnow().isoformat()
    records["Source"] = source
    records["Signal direction"] = records["Final signal"].map(signal_direction)
    records = records.rename(columns={"Price": "Entry price"})
    keep_cols = [
        "Record date", "Recorded at", "Source", "Ticker", "Entry price", "Daily Change %",
        "Volume", "RSI", "MACD signal", "VWAP status", "Trend", "AI bullish probability",
        "AI bearish probability", "Buy zone", "Stop loss", "Take profit", "Risk-reward ratio",
        "Suitability score", "Suitability", "Suitability reasons", "Final signal", "Signal direction",
    ]
    for col in keep_cols:
        if col not in records.columns:
            records[col] = None
    records = records[keep_cols]
    if DAILY_ANALYSIS_FILE.exists():
        existing = pd.read_csv(DAILY_ANALYSIS_FILE)
        existing = existing[
            ~(
                (existing["Record date"].astype(str) == today)
                & (existing["Source"].astype(str) == source)
                & (existing["Ticker"].astype(str).isin(records["Ticker"].astype(str)))
            )
        ]
        records = pd.concat([existing, records], ignore_index=True)
    save_csv_archive(records, DAILY_ANALYSIS_FILE, "Record date")
    return len(rows)


def load_daily_records() -> pd.DataFrame:
    if not DAILY_ANALYSIS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(DAILY_ANALYSIS_FILE)


def record_scanner_history(rows: pd.DataFrame, scan_size: int) -> int:
    if rows.empty:
        return 0
    SCANNER_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().date().isoformat()
    records = rows.copy()
    records.insert(0, "Archive date", today)
    records.insert(1, "Recorded at", datetime.utcnow().isoformat())
    records.insert(2, "Source", "Multi-Stock AI Scanner")
    records.insert(3, "Scan size", int(scan_size))
    if SCANNER_HISTORY_FILE.exists():
        existing = pd.read_csv(SCANNER_HISTORY_FILE)
        existing = existing[
            ~(
                (existing["Archive date"].astype(str) == today)
                & (existing["Ticker"].astype(str).isin(records["Ticker"].astype(str)))
            )
        ]
        records = pd.concat([existing, records], ignore_index=True)
    save_csv_archive(records, SCANNER_HISTORY_FILE, "Archive date")
    return len(rows)


def load_scanner_history() -> pd.DataFrame:
    if not SCANNER_HISTORY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(SCANNER_HISTORY_FILE)


def _today_exists(path: Path, date_col: str, source: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        records = pd.read_csv(path)
    except Exception:
        return False
    if records.empty or date_col not in records:
        return False
    today = pd.Timestamp.now().date().isoformat()
    mask = records[date_col].astype(str) == today
    if source and "Source" in records:
        mask &= records["Source"].astype(str) == source
    return bool(mask.any())


def export_signal_workbook(rows: pd.DataFrame, report_date: str, prefix: str) -> Path | None:
    if rows.empty:
        return None
    DAILY_EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    path = DAILY_EXCEL_DIR / f"{prefix}_{report_date}.xlsx"
    sheet_order = ["Strong Buy", "Buy", "Hold", "Sell", "Avoid"]
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            rows.to_excel(writer, sheet_name="All Signals", index=False)
            for signal in sheet_order:
                filtered = rows[rows.get("Final signal", pd.Series(dtype=str)).astype(str) == signal]
                if not filtered.empty:
                    filtered.to_excel(writer, sheet_name=signal[:31], index=False)
    except Exception:
        fallback = DAILY_EXCEL_DIR / f"{prefix}_{report_date}.csv"
        rows.to_csv(fallback, index=False)
        return fallback
    return path


def run_daily_top50_autosave(selected: list[str], allow_penny: bool, force: bool = False) -> dict:
    today = pd.Timestamp.now().date().isoformat()
    already_saved = (
        _today_exists(DAILY_ANALYSIS_FILE, "Record date", AUTO_DAILY_SOURCE)
        and _today_exists(SCANNER_HISTORY_FILE, "Archive date")
        and _today_exists(TECHNICAL_SNAPSHOT_FILE, "Record date", "Technical Indicator Dashboard")
    )
    if already_saved and not force:
        return {"status": "skipped", "message": f"Top 50 autosave already exists for {today}."}
    universe = scanner_universe(selected, 50)
    table = scan_stocks(universe, limit=50, allow_penny_stocks=allow_penny)
    if table.empty:
        return {"status": "empty", "message": "No top 50 scanner data was available to autosave."}
    signal_rows = table[table["Final signal"].isin(["Strong Buy", "Buy", "Hold", "Sell", "Avoid"])].copy()
    signal_count = record_daily_rows(signal_rows, AUTO_DAILY_SOURCE)
    scanner_count = record_scanner_history(table, 50)
    technical_rows, technical_errors = record_technical_snapshots(table["Ticker"].astype(str).tolist()[:50], ["Daily"])
    workbook_path = export_signal_workbook(signal_rows, today, "daily_top50_signal_analysis")
    return {
        "status": "saved",
        "message": f"Autosaved {scanner_count} scanner row(s), {len(technical_rows)} technical row(s), and {signal_count} signal journal row(s).",
        "table": table,
        "technical_errors": technical_errors,
        "workbook_path": workbook_path,
    }


def filter_history_range(records: pd.DataFrame, date_col: str, preset: str) -> pd.DataFrame:
    if records.empty or date_col not in records:
        return records
    output = records.copy()
    output[date_col] = pd.to_datetime(output[date_col], errors="coerce")
    today = pd.Timestamp.now().normalize()
    if preset == "Last 1 month":
        start = today - pd.DateOffset(months=1)
    elif preset == "Last 3 months":
        start = today - pd.DateOffset(months=3)
    elif preset == "Last 6 months":
        start = today - pd.DateOffset(months=6)
    elif preset == "Last 1 year":
        start = today - pd.DateOffset(years=1)
    elif preset == "Last 2 years":
        start = today - pd.DateOffset(years=2)
    elif preset == "Last 5 years":
        start = today - pd.DateOffset(years=5)
    else:
        return output
    return output[output[date_col] >= start]


def _period_slice(df: pd.DataFrame, period: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    if as_of is not None:
        as_of = pd.Timestamp(as_of).normalize()
        data = data[pd.to_datetime(data.index).normalize() <= as_of]
        if data.empty:
            return data
    last_day = pd.Timestamp(data.index.max()).normalize()
    if period == "Daily":
        return data.tail(1)
    if period == "Weekly":
        start_day = last_day.to_period("W").start_time.normalize()
    elif period == "Monthly":
        start_day = last_day.to_period("M").start_time.normalize()
    else:
        start_day = last_day
    return data[data.index >= start_day]


def _technical_snapshot_row(ticker: str, analysis: dict, period: str, record_date: str | pd.Timestamp | None = None) -> dict:
    record_day = pd.Timestamp(record_date).normalize() if record_date is not None else pd.Timestamp.now().normalize()
    df = analysis.get("Data", pd.DataFrame()).dropna()
    window = _period_slice(df, period, as_of=record_day)
    if window.empty:
        raise ValueError("No technical data available")
    latest = window.iloc[-1]
    latest_market_day = pd.Timestamp(window.index.max()).normalize()
    first_close = float(window["Close"].iloc[0])
    last_close = float(window["Close"].iloc[-1])
    period_return = ((last_close / first_close) - 1) * 100 if first_close else 0
    market_status = "Trading day"
    if record_day.weekday() >= 5:
        market_status = "Weekend / market closed"
    elif latest_market_day < record_day:
        market_status = "Market closed / no new bar"
    return {
        "Record date": record_day.date().isoformat(),
        "Recorded at": datetime.utcnow().isoformat(),
        "Source": "Technical Indicator Dashboard",
        "Market status": market_status,
        "Latest market date": latest_market_day.date().isoformat(),
        "Period": period,
        "Period start": pd.Timestamp(window.index.min()).date().isoformat(),
        "Period end": pd.Timestamp(window.index.max()).date().isoformat(),
        "Ticker": ticker,
        "Price": analysis.get("Price"),
        "Period return %": period_return,
        "Period high": float(window["High"].max()),
        "Period low": float(window["Low"].min()),
        "Period volume": int(window["Volume"].sum()),
        "RSI": float(latest.get("RSI", 0)),
        "MACD": float(latest.get("MACD", 0)),
        "MACD signal value": float(latest.get("MACD_Signal", 0)),
        "MACD signal": analysis.get("MACD signal"),
        "VWAP status": analysis.get("VWAP status"),
        "Trend": analysis.get("Trend"),
        "ATR %": float(latest.get("ATR_Pct", 0)),
        "Volume ratio": float(latest.get("Volume_Ratio", 0)),
        "AI bullish probability": analysis.get("AI bullish probability"),
        "AI bearish probability": analysis.get("AI bearish probability"),
        "Suitability score": analysis.get("Suitability score"),
        "Suitability": analysis.get("Suitability"),
        "Final signal": analysis.get("Final signal"),
    }


def record_technical_snapshots(tickers: list[str], periods: list[str], record_date: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors = []
    record_day = pd.Timestamp(record_date).normalize() if record_date is not None else pd.Timestamp.now().normalize()
    today = record_day.date().isoformat()
    clean_tickers = normalize_tickers(tickers)
    clean_periods = [period for period in periods if period in {"Daily", "Weekly", "Monthly"}]
    for ticker in clean_tickers:
        try:
            analysis = analyze_stock(ticker)
            if "Error" in analysis:
                errors.append(f"{ticker}: {analysis['Error']}")
                continue
            for period in clean_periods:
                rows.append(_technical_snapshot_row(ticker, analysis, period, record_date=record_day))
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
    records = pd.DataFrame(rows)
    if records.empty:
        return records, errors
    TECHNICAL_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TECHNICAL_SNAPSHOT_FILE.exists():
        existing = pd.read_csv(TECHNICAL_SNAPSHOT_FILE)
        existing = existing[
            ~(
                (existing["Record date"].astype(str) == today)
                & (existing["Ticker"].astype(str).isin(records["Ticker"].astype(str)))
                & (existing["Period"].astype(str).isin(records["Period"].astype(str)))
            )
        ]
        records = pd.concat([existing, records], ignore_index=True)
    save_csv_archive(records, TECHNICAL_SNAPSHOT_FILE, "Record date")
    saved_rows = records.tail(len(rows)).reset_index(drop=True)
    daily_signal_rows = technical_snapshots_to_signal_records(saved_rows)
    if not daily_signal_rows.empty:
        record_daily_rows(daily_signal_rows, "Daily Technical Snapshot Journal")
    return saved_rows, errors


def load_technical_snapshots() -> pd.DataFrame:
    if not TECHNICAL_SNAPSHOT_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(TECHNICAL_SNAPSHOT_FILE)


def calendar_archive_dates(end_date: pd.Timestamp | None = None) -> list[str]:
    end = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.now().normalize()
    end = min(end, CALENDAR_ARCHIVE_END)
    if end < CALENDAR_ARCHIVE_START:
        return []
    return [date.date().isoformat() for date in pd.date_range(CALENDAR_ARCHIVE_START, end, freq="D")]


def missing_calendar_dates(history: pd.DataFrame, tickers: list[str] | None = None) -> list[str]:
    expected = set(calendar_archive_dates())
    if history.empty or "Record date" not in history:
        return sorted(expected)
    daily = history[history.get("Period", pd.Series(dtype=str)).astype(str) == "Daily"].copy()
    daily["Record date"] = pd.to_datetime(daily["Record date"], errors="coerce").dt.date.astype(str)
    if tickers:
        clean_tickers = set(normalize_tickers(tickers))
        complete_dates = set()
        for date, group in daily.groupby("Record date"):
            saved_tickers = set(group["Ticker"].astype(str))
            if clean_tickers.issubset(saved_tickers):
                complete_dates.add(date)
        return sorted(expected - complete_dates)
    saved = set(daily["Record date"].dropna().astype(str))
    return sorted(expected - saved)


def backfill_calendar_technical_snapshots(tickers: list[str], dates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    all_rows = []
    all_errors = []
    for date in dates:
        rows, errors = record_technical_snapshots(tickers, ["Daily"], record_date=date)
        if not rows.empty:
            all_rows.append(rows)
        all_errors.extend(errors)
    if all_rows:
        return pd.concat(all_rows, ignore_index=True), all_errors
    return pd.DataFrame(), all_errors


def technical_snapshots_to_signal_records(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or "Period" not in snapshots:
        return pd.DataFrame()
    daily = snapshots[snapshots["Period"].astype(str) == "Daily"].copy()
    if daily.empty:
        return pd.DataFrame()
    records = pd.DataFrame(
        {
            "Record date": daily.get("Record date"),
            "Recorded at": daily.get("Recorded at"),
            "Source": "Daily Technical Snapshot Journal",
            "Ticker": daily.get("Ticker"),
            "Entry price": daily.get("Price"),
            "Daily Change %": daily.get("Period return %"),
            "Volume": daily.get("Period volume"),
            "RSI": daily.get("RSI"),
            "MACD signal": daily.get("MACD signal"),
            "VWAP status": daily.get("VWAP status"),
            "Trend": daily.get("Trend"),
            "AI bullish probability": daily.get("AI bullish probability"),
            "AI bearish probability": daily.get("AI bearish probability"),
            "Buy zone": None,
            "Stop loss": None,
            "Take profit": None,
            "Risk-reward ratio": None,
            "Suitability score": daily.get("Suitability score"),
            "Suitability": daily.get("Suitability"),
            "Suitability reasons": None,
            "Final signal": daily.get("Final signal"),
        }
    )
    records["Signal direction"] = records["Final signal"].map(signal_direction)
    return records


def load_signal_accuracy_records() -> pd.DataFrame:
    records = load_daily_records()
    technical_records = technical_snapshots_to_signal_records(load_technical_snapshots())
    combined = pd.concat([records, technical_records], ignore_index=True) if not technical_records.empty else records
    if combined.empty:
        return combined
    if "Source" in combined:
        combined = combined[combined["Source"].astype(str) != "Technical Indicator Dashboard"].copy()
    combined["Record date"] = pd.to_datetime(combined["Record date"], errors="coerce")
    combined["Recorded at"] = pd.to_datetime(combined["Recorded at"], errors="coerce")
    combined = combined.sort_values("Recorded at", na_position="first")
    combined = combined.drop_duplicates(["Record date", "Source", "Ticker"], keep="last")
    return combined


def import_signal_journal_file(uploaded_file) -> int:
    if uploaded_file is None:
        return 0
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".xlsx"):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        imported = pd.concat(sheets.values(), ignore_index=True)
    else:
        imported = pd.read_csv(uploaded_file)
    if imported.empty:
        return 0
    if "Record date" not in imported or "Ticker" not in imported:
        raise ValueError("Imported journal must include Record date and Ticker columns.")
    if "Source" not in imported:
        imported["Source"] = "Imported Signal Journal"
    if "Final signal" not in imported:
        imported["Final signal"] = "Hold"
    imported["Signal direction"] = imported["Final signal"].map(signal_direction)
    if "Recorded at" not in imported:
        imported["Recorded at"] = datetime.utcnow().isoformat()
    existing = load_daily_records()
    combined = pd.concat([existing, imported], ignore_index=True) if not existing.empty else imported
    combined["Record date"] = pd.to_datetime(combined["Record date"], errors="coerce").dt.date.astype(str)
    combined = combined.sort_values("Recorded at", na_position="first")
    combined = combined.drop_duplicates(["Record date", "Source", "Ticker"], keep="last")
    save_csv_archive(combined, DAILY_ANALYSIS_FILE, "Record date")
    return len(imported)


def evaluate_signal_records(records: pd.DataFrame, horizon_days: int = 5, threshold_pct: float = 1.5) -> pd.DataFrame:
    if records.empty:
        return records
    output = records.copy()
    output["Record date"] = pd.to_datetime(output["Record date"], errors="coerce")
    output["Entry price"] = pd.to_numeric(output["Entry price"], errors="coerce")
    rows = []
    today = pd.Timestamp.now().normalize()
    for _, row in output.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        record_date = row.get("Record date")
        entry = row.get("Entry price")
        if not ticker or pd.isna(record_date) or pd.isna(entry):
            rows.append({**row.to_dict(), "Evaluation status": "Invalid", "Wait reason": "Missing ticker, date, or entry price", "Correct": None})
            continue
        record_day = pd.Timestamp(record_date).normalize()
        age_days = int((today - record_day).days)
        hist = fetch_history(ticker, period="1y")
        if hist.empty:
            rows.append({**row.to_dict(), "Evaluation status": "No price data", "Wait reason": "Could not download price history", "Days since signal": age_days, "Correct": None})
            continue
        hist = hist.copy()
        hist["Trade Date"] = pd.to_datetime(hist.index).normalize()
        future_prices = hist[hist["Trade Date"] >= record_day]
        available_bars = max(0, len(future_prices) - 1)
        if available_bars < horizon_days:
            rows.append(
                {
                    **row.to_dict(),
                    "Evaluation status": "Waiting",
                    "Wait reason": f"Needs {horizon_days} later trading bar(s); available {available_bars}",
                    "Days since signal": age_days,
                    "Trading bars available": available_bars,
                    "Correct": None,
                }
            )
            continue
        exit_price = float(future_prices["Close"].iloc[horizon_days])
        realized = (exit_price / float(entry) - 1) * 100
        direction = signal_direction(row.get("Final signal", "Hold"))
        if direction == "Bullish":
            correct = realized >= threshold_pct
        elif direction == "Bearish":
            correct = realized <= -threshold_pct
        else:
            correct = abs(realized) < threshold_pct
        rows.append(
            {
                **row.to_dict(),
                "Evaluation status": "Evaluated",
                "Wait reason": "",
                "Days since signal": age_days,
                "Trading bars available": available_bars,
                "Exit price": exit_price,
                "Realized return %": realized,
                "Correct": bool(correct),
            }
        )
    return pd.DataFrame(rows)


def evaluate_manual_daily_change_records(records: pd.DataFrame, hold_threshold_pct: float = 1.5) -> pd.DataFrame:
    if records.empty:
        return records
    output = records.copy()
    output["Daily Change %"] = pd.to_numeric(output["Daily Change %"], errors="coerce")
    rows = []
    for _, row in output.iterrows():
        signal = str(row.get("Final signal", ""))
        daily_change = row.get("Daily Change %")
        if pd.isna(daily_change):
            rows.append({**row.to_dict(), "Evaluation status": "Invalid", "Correct": None, "Manual rule": "Missing Daily Change %"})
            continue
        if signal in {"Strong Buy", "Buy"}:
            correct = daily_change > 0
            rule = "Bullish signal correct when Daily Change % > 0"
        elif signal in {"Sell", "Avoid"}:
            correct = daily_change < 0
            rule = "Bearish signal correct when Daily Change % < 0"
        elif signal == "Hold":
            correct = abs(daily_change) <= hold_threshold_pct
            rule = f"Hold correct when abs(Daily Change %) <= {hold_threshold_pct:.2f}%"
        else:
            correct = False
            rule = "Unknown signal"
        rows.append(
            {
                **row.to_dict(),
                "Evaluation status": "Evaluated",
                "Realized return %": daily_change,
                "Correct": bool(correct),
                "Manual rule": rule,
            }
        )
    return pd.DataFrame(rows)


def fetch_option_contracts(underlying: str, option_type: str, min_dte: int, max_dte: int, limit: int = 1000) -> pd.DataFrame:
    if not (app_secret("ALPACA_OPTIONS_API_KEY") or app_secret("ALPACA_API_KEY")) or not (app_secret("ALPACA_OPTIONS_SECRET_KEY") or app_secret("ALPACA_SECRET_KEY")):
        return pd.DataFrame()
    normalized_underlying = normalize_tickers([underlying])[0] if normalize_tickers([underlying]) else ""
    if not normalized_underlying:
        return pd.DataFrame()
    today = datetime.utcnow().date()
    url = f"{alpaca_rest_base('options')}/options/contracts"
    params = {
        "underlying_symbols": normalized_underlying,
        "type": option_type.lower(),
        "status": "active",
        "expiration_date_gte": (today + timedelta(days=min_dte)).isoformat(),
        "expiration_date_lte": (today + timedelta(days=max_dte)).isoformat(),
        "limit": limit,
    }
    try:
        response = requests.get(url, params=params, headers=alpaca_headers("options"), timeout=20)
        response.raise_for_status()
    except Exception:
        return pd.DataFrame()
    contracts = response.json().get("option_contracts", [])
    if not contracts:
        return pd.DataFrame()
    rows = []
    for contract in contracts:
        rows.append(
            {
                "Option Symbol": contract.get("symbol"),
                "Underlying": contract.get("underlying_symbol") or underlying,
                "Type": str(contract.get("type", option_type)).title(),
                "Strike": float(contract.get("strike_price") or 0),
                "Expiration": contract.get("expiration_date"),
                "Status": contract.get("status"),
                "Tradable": contract.get("tradable", True),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Expiration"] = pd.to_datetime(frame["Expiration"], errors="coerce")
    frame["DTE"] = (frame["Expiration"].dt.date - today).apply(lambda value: value.days if pd.notna(value) else None)
    frame = frame.dropna(subset=["Option Symbol", "Strike", "Expiration"])
    return frame[(frame["Status"].astype(str).str.lower() == "active") & (frame["Tradable"].astype(bool))]


def fetch_option_snapshots(symbols: list[str]) -> pd.DataFrame:
    clean_symbols = [symbol for symbol in dict.fromkeys([str(s).strip().upper() for s in symbols]) if symbol]
    if not clean_symbols:
        return pd.DataFrame()
    rows = []
    for start in range(0, len(clean_symbols), 100):
        chunk = clean_symbols[start:start + 100]
        try:
            response = requests.get(
                f"{alpaca_options_data_base()}/snapshots",
                params={"symbols": ",".join(chunk), "feed": "indicative", "limit": len(chunk)},
                headers=alpaca_headers("options"),
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        snapshots = payload.get("snapshots") or payload
        if isinstance(snapshots, list):
            snapshot_items = [(item.get("symbol"), item) for item in snapshots]
        else:
            snapshot_items = snapshots.items()
        for symbol, snapshot in snapshot_items:
            if not symbol or not isinstance(snapshot, dict):
                continue
            greeks = snapshot.get("greeks") or {}
            quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
            trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
            bid = quote.get("bp") or quote.get("bid_price")
            ask = quote.get("ap") or quote.get("ask_price")
            last = trade.get("p") or trade.get("price")
            bid = float(bid) if bid is not None else None
            ask = float(ask) if ask is not None else None
            last = float(last) if last is not None else None
            mid = (bid + ask) / 2 if bid and ask and ask >= bid else last
            rows.append(
                {
                    "Option Symbol": symbol,
                    "Delta": greeks.get("delta"),
                    "IV": greeks.get("iv") or greeks.get("implied_volatility"),
                    "Bid": bid,
                    "Ask": ask,
                    "Mid": mid,
                    "Last": last,
                }
            )
    frame = pd.DataFrame(rows)
    for col in ["Delta", "IV", "Bid", "Ask", "Mid", "Last"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def estimate_put_delta(underlying_price: float, strike: float, dte: float) -> float:
    if underlying_price <= 0:
        return -0.15
    otm_pct = max(0.0, (underlying_price - strike) / underlying_price)
    base = 0.50 - otm_pct * 7.5
    dte_adjustment = min(0.08, max(-0.05, (float(dte or 30) - 30) / 365))
    return -round(min(0.50, max(0.05, base + dte_adjustment)), 3)


def enrich_option_contracts_with_market_data(contracts: pd.DataFrame) -> pd.DataFrame:
    if contracts.empty:
        return contracts
    snapshots = fetch_option_snapshots(contracts["Option Symbol"].dropna().astype(str).tolist())
    if snapshots.empty:
        return contracts
    enriched = contracts.merge(snapshots, on="Option Symbol", how="left")
    return enriched


def options_signal_from_stock(row: pd.Series) -> str:
    if row["Final signal"] in {"Strong Buy", "Buy"}:
        return "call"
    if row["Final signal"] in {"Sell", "Avoid"}:
        return "put"
    return "skip"


def build_options_ai_scanner(
    selected: list[str],
    scan_size: int,
    min_dte: int,
    max_dte: int,
    max_moneyness_pct: float,
    allow_penny: bool,
) -> pd.DataFrame:
    stock_table = scan_stocks(scanner_universe(selected, scan_size), limit=scan_size, allow_penny_stocks=allow_penny)
    if stock_table.empty:
        return pd.DataFrame()
    rows = []
    actionable = stock_table[stock_table["Final signal"].isin(["Strong Buy", "Buy", "Sell", "Avoid"])].copy()
    for _, stock in actionable.iterrows():
        option_type = options_signal_from_stock(stock)
        contracts = fetch_option_contracts(stock["Ticker"], option_type, min_dte, max_dte)
        if contracts.empty:
            continue
        price = float(stock["Price"])
        contracts["Moneyness %"] = (contracts["Strike"] / price - 1) * 100
        if option_type == "call":
            candidates = contracts[(contracts["Strike"] >= price * 0.97) & (contracts["Strike"] <= price * (1 + max_moneyness_pct / 100))]
        else:
            candidates = contracts[(contracts["Strike"] <= price * 1.03) & (contracts["Strike"] >= price * (1 - max_moneyness_pct / 100))]
        if candidates.empty:
            candidates = contracts.iloc[(contracts["Strike"] - price).abs().argsort()].head(3)
        candidates = candidates.copy()
        candidates["Underlying Price"] = price
        candidates["Underlying Signal"] = stock["Final signal"]
        candidates["AI bullish probability"] = stock["AI bullish probability"]
        candidates["AI bearish probability"] = stock["AI bearish probability"]
        candidates["Suitability score"] = stock["Suitability score"]
        candidates["Risk-reward ratio"] = stock["Risk-reward ratio"]
        signal_probability = stock["AI bullish probability"] if option_type == "call" else stock["AI bearish probability"]
        candidates["Options AI score"] = (
            candidates["Suitability score"]
            + candidates["Risk-reward ratio"].clip(upper=5) * 8
            + candidates["DTE"].between(min_dte, max_dte).astype(int) * 10
            - candidates["Moneyness %"].abs().clip(upper=50) * 0.7
            + signal_probability * 0.35
        )
        candidates["Suggested side"] = "buy"
        candidates["Estimated max loss"] = "Premium paid x 100 x contracts"
        rows.append(candidates.sort_values("Options AI score", ascending=False).head(1))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values("Options AI score", ascending=False)


def build_wheel_strategy_candidates(
    selected: list[str],
    scan_size: int,
    target_dte: int = 30,
    min_dte: int = 14,
    max_dte: int = 45,
    target_delta: float = 0.15,
    min_delta: float = 0.10,
    max_delta: float = 0.20,
    allow_penny: bool = False,
) -> pd.DataFrame:
    stock_table = scan_stocks(scanner_universe(selected, scan_size), limit=scan_size, allow_penny_stocks=allow_penny)
    if stock_table.empty:
        return pd.DataFrame()
    rows = []
    options_bp = account_buying_power("options")
    wheel_stocks = stock_table[
        (stock_table["Suitability"] == "Suitable")
        & (~stock_table["Final signal"].isin(["Sell", "Avoid"]))
    ].copy()
    for _, stock in wheel_stocks.iterrows():
        contracts = fetch_option_contracts(stock["Ticker"], "put", min_dte, max_dte)
        if contracts.empty:
            continue
        contracts = enrich_option_contracts_with_market_data(contracts)
        price = float(stock["Price"])
        contracts["Underlying Price"] = price
        contracts["Moneyness %"] = (contracts["Strike"] / price - 1) * 100
        contracts = contracts[contracts["Strike"] < price].copy()
        if contracts.empty:
            continue
        if "Delta" not in contracts:
            contracts["Delta"] = None
        contracts["Delta Source"] = contracts["Delta"].notna().map({True: "Alpaca snapshot", False: "Estimated"})
        contracts["Delta"] = contracts.apply(
            lambda row: float(row["Delta"]) if pd.notna(row["Delta"]) else estimate_put_delta(price, row["Strike"], row["DTE"]),
            axis=1,
        )
        contracts["Abs Delta"] = contracts["Delta"].abs()
        candidates = contracts[contracts["Abs Delta"].between(min_delta, max_delta)].copy()
        if candidates.empty:
            candidates = contracts.iloc[(contracts["Abs Delta"] - target_delta).abs().argsort()].head(3).copy()
        candidates["Underlying Signal"] = stock["Final signal"]
        candidates["AI bullish probability"] = stock["AI bullish probability"]
        candidates["Suitability score"] = stock["Suitability score"]
        candidates["Collateral Required"] = candidates["Strike"] * 100
        candidates["Buying Power OK"] = candidates["Collateral Required"] <= options_bp
        candidates["Suggested side"] = "sell"
        candidates["Wheel Score"] = (
            candidates["Suitability score"]
            + candidates["AI bullish probability"] * 0.25
            - (candidates["DTE"] - target_dte).abs() * 1.2
            - (candidates["Abs Delta"] - target_delta).abs() * 180
            - candidates["Moneyness %"].abs() * 0.25
            + candidates["Buying Power OK"].astype(int) * 12
        )
        rows.append(candidates.sort_values("Wheel Score", ascending=False).head(1))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values("Wheel Score", ascending=False)


def format_scan_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    numeric_cols = {
        "Price": "${:.2f}",
        "Daily Change %": "{:.2f}%",
        "RSI": "{:.1f}",
        "AI bullish probability": "{:.1f}%",
        "AI bearish probability": "{:.1f}%",
        "Buy zone": "${:.2f}",
        "Stop loss": "${:.2f}",
        "Take profit": "${:.2f}",
        "Risk-reward ratio": "{:.2f}",
        "Suitability score": "{:.0f}",
    }
    styler = df.style.format({k: v for k, v in numeric_cols.items() if k in df.columns})

    def highlight_conviction(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        bullish = pd.to_numeric(row.get("AI bullish probability"), errors="coerce")
        is_synergy = bool(row.get("Synergy Strong Buy", False))
        note = str(row.get("Signal quality note", ""))
        for idx, col in enumerate(row.index):
            if col == "AI bullish probability" and pd.notna(bullish) and 82.5 <= float(bullish) <= 90:
                styles[idx] = "background-color: #e7f7ef; color: #126b3f; font-weight: 700;"
            if col == "Synergy Strong Buy" and is_synergy:
                styles[idx] = "background-color: #dff5ff; color: #075985; font-weight: 700;"
            if col == "Signal quality note" and note:
                styles[idx] = "background-color: #fff7d6; color: #7a4b00; font-weight: 600;"
        return styles

    return styler.apply(highlight_conviction, axis=1)


def render_scanner_history_reference() -> None:
    history = load_scanner_history()
    if history.empty:
        st.info("No scanner reference archive yet. Run the scanner with daily archive enabled to start saving history.")
        return
    st.subheader("Multi-Stock Scanner Reference Archive")
    c1, c2, c3 = st.columns(3)
    range_preset = c1.selectbox(
        "Scanner archive range",
        ["Last 1 month", "Last 3 months", "Last 6 months", "Last 1 year", "Last 2 years", "Last 5 years", "All history"],
        index=3,
    )
    ticker_options = sorted(history["Ticker"].dropna().astype(str).unique().tolist()) if "Ticker" in history else []
    selected_tickers = c2.multiselect("Archive tickers", ticker_options, default=ticker_options[:20])
    signal_options = sorted(history["Final signal"].dropna().astype(str).unique().tolist()) if "Final signal" in history else []
    selected_signals = c3.multiselect("Archive signals", signal_options, default=signal_options)
    shown = filter_history_range(history, "Archive date", range_preset)
    if selected_tickers:
        shown = shown[shown["Ticker"].astype(str).isin(selected_tickers)]
    if selected_signals:
        shown = shown[shown["Final signal"].astype(str).isin(selected_signals)]
    if shown.empty:
        st.info("No scanner archive rows match the selected filters.")
        return
    shown = shown.sort_values(["Archive date", "Ticker"], ascending=[False, True])
    m1, m2, m3 = st.columns(3)
    m1.metric("Archived rows", f"{len(shown):,}")
    m2.metric("Archived dates", shown["Archive date"].astype(str).nunique())
    m3.metric("Archived tickers", shown["Ticker"].astype(str).nunique() if "Ticker" in shown else 0)
    st.dataframe(format_scan_table(shown.head(500)), use_container_width=True, hide_index=True)
    st.download_button(
        "Download scanner reference archive",
        shown.to_csv(index=False).encode("utf-8"),
        file_name="multi_stock_ai_scanner_history.csv",
        mime="text/csv",
    )


def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.copy()
    if "EMA_20" not in df and "Close" in df:
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    if "EMA_50" not in df and "Close" in df:
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    if "VWAP" not in df and {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (typical * df["Volume"]).cumsum() / df["Volume"].replace(0, pd.NA).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker))
    if "EMA_20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(width=1.5)))
    if "EMA_50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50", line=dict(width=1.5)))
    if "VWAP" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", line=dict(width=1.2, dash="dot")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False)
    return fig


def add_vwap_column(df: pd.DataFrame, column_name: str = "VWAP") -> pd.DataFrame:
    out = df.copy()
    if out.empty or not {"High", "Low", "Close", "Volume"}.issubset(out.columns):
        return out
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    volume = pd.to_numeric(out["Volume"], errors="coerce").fillna(0)
    out[column_name] = (typical * volume).cumsum() / volume.replace(0, float("nan")).cumsum()
    return out


def session_vwap_frame(ticker: str) -> pd.DataFrame:
    try:
        intraday = fetch_history(ticker, period="1d", interval="5m")
    except Exception:
        intraday = pd.DataFrame()
    if intraday.empty:
        return intraday
    return add_vwap_column(intraday, "Session_VWAP")


def anchored_vwap_series(df: pd.DataFrame, anchor_date: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp | None]:
    if df.empty:
        return pd.Series(dtype=float), None
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    anchor_date = pd.Timestamp(anchor_date).normalize()
    anchored = data[pd.to_datetime(data.index).normalize() >= anchor_date].copy()
    if anchored.empty:
        anchored = data.tail(1).copy()
    anchored = add_vwap_column(anchored, "Anchored_VWAP")
    return anchored["Anchored_VWAP"], pd.Timestamp(anchored.index.min()).normalize()


def individual_analysis_chart(
    df: pd.DataFrame,
    ticker: str,
    session_vwap_value: float | None,
    anchored_vwap: pd.Series,
) -> go.Figure:
    data = df.copy()
    for span in [20, 60, 250]:
        col = f"EMA_{span}"
        if col not in data and "Close" in data:
            data[col] = data["Close"].ewm(span=span, adjust=False).mean()
    if "Cumulative_VWAP" not in data:
        if "VWAP" in data:
            data["Cumulative_VWAP"] = data["VWAP"]
        else:
            data = add_vwap_column(data, "Cumulative_VWAP")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name=ticker))
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA_20"], name="EMA 20", line=dict(width=1.3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA_60"], name="EMA 60", line=dict(width=1.3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA_250"], name="EMA 250", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=data.index, y=data["Cumulative_VWAP"], name="Cumulative VWAP", line=dict(width=1.2, dash="dot")))
    if not anchored_vwap.empty:
        fig.add_trace(go.Scatter(x=anchored_vwap.index, y=anchored_vwap, name="Anchored VWAP", line=dict(width=2)))
    if session_vwap_value is not None and pd.notna(session_vwap_value):
        fig.add_hline(y=session_vwap_value, line_dash="dash", annotation_text="Daily Session VWAP")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False)
    return fig


SHORT_DEFAULT_UNIVERSE = [
    "SOUN", "OPEN", "RGTI", "BBAI", "IONQ", "QUBT", "LUNR", "ACHR", "JOBY", "SERV",
    "WULF", "RIOT", "MARA", "HUT", "BTDR", "CIFR", "LAZR", "OUST", "RKLB", "ASTS",
    "DNA", "PLUG", "BE", "BLNK", "QS", "MVIS", "SPCE", "KULR", "PSNY", "GRAB",
]

SHORT_PROFILE_OVERRIDES = {
    "SOUN": {"Sector": "Technology", "Country": "United States", "Market Cap": 180_000_000, "Float": 62_000_000},
    "OPEN": {"Sector": "Real Estate", "Country": "United States", "Market Cap": 190_000_000, "Float": 98_000_000},
    "RGTI": {"Sector": "Technology", "Country": "United States", "Market Cap": 120_000_000, "Float": 47_000_000},
    "BBAI": {"Sector": "Technology", "Country": "United States", "Market Cap": 95_000_000, "Float": 31_000_000},
    "LUNR": {"Sector": "Industrials", "Country": "United States", "Market Cap": 145_000_000, "Float": 34_000_000},
}


def short_mock_profile(ticker: str) -> dict:
    seed = sum(ord(ch) for ch in ticker)
    sectors = ["Technology", "Industrials", "Consumer Cyclical", "Financial Services", "Communication Services"]
    return {
        "Sector": sectors[seed % len(sectors)],
        "Country": "United States",
        "Market Cap": float(35_000_000 + (seed % 150) * 1_000_000),
        "Float": float(8_000_000 + (seed % 70) * 1_000_000),
    }


def short_profile(ticker: str) -> dict:
    return {**short_mock_profile(ticker), **SHORT_PROFILE_OVERRIDES.get(ticker, {})}


def short_market_frame(ticker: str) -> pd.DataFrame:
    hist = fetch_history(ticker, period="1y")
    if not hist.empty:
        return hist
    seed = sum(ord(ch) for ch in ticker)
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=260)
    base = 3.5 + (seed % 70) / 10
    rows = []
    for idx, date in enumerate(dates):
        drift = 1 + (idx / 260) * ((seed % 9) / 10)
        pulse = 1 + ((idx + seed) % 17) / 100
        close = base * drift * pulse
        open_price = close * (0.96 + ((idx + seed) % 8) / 100)
        high = max(open_price, close) * 1.06
        low = min(open_price, close) * 0.94
        volume = 400_000 + ((idx * 13 + seed) % 90) * 80_000
        rows.append({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume})
    return pd.DataFrame(rows, index=dates)


def short_base_metrics(ticker: str) -> dict | None:
    ticker = normalize_tickers([ticker])[0] if normalize_tickers([ticker]) else ""
    if not ticker:
        return None
    hist = short_market_frame(ticker).dropna()
    if len(hist) < 5:
        return None
    profile = short_profile(ticker)
    latest = hist.iloc[-1]
    previous = hist.iloc[-2]
    price = float(latest["Close"])
    change_pct = (price / float(previous["Close"]) - 1) * 100
    avg_volume = float(hist["Volume"].tail(20).mean())
    volume = float(latest["Volume"])
    rel_volume = volume / avg_volume if avg_volume else 0
    seed = sum(ord(ch) for ch in ticker)
    gap_pct = max(change_pct, 25 + (seed % 150))
    premarket_price = float(previous["Close"]) * (1 + gap_pct / 100)
    open_price = premarket_price * (0.94 + (seed % 9) / 100)
    high_after_open = open_price * (1.12 + (seed % 30) / 100)
    one_hour_high = high_after_open * 0.98
    one_hour_low = open_price * (0.88 + (seed % 8) / 100)
    current_price = max(price, premarket_price * (0.9 + (seed % 25) / 100))
    return {
        "Ticker": ticker,
        "Current Price": current_price,
        "% Change": change_pct,
        "Volume": volume,
        "Relative Volume": rel_volume,
        "Dollar Volume": current_price * volume,
        "Market Cap": float(profile["Market Cap"]),
        "Float": float(profile["Float"]),
        "Sector": profile["Sector"],
        "Country": profile["Country"],
        "Previous Close": float(previous["Close"]),
        "Pre-market Price": premarket_price,
        "Opening Price": open_price,
        "High After Open": high_after_open,
        "1H High": one_hour_high,
        "1H Low": one_hour_low,
        "Gap %": gap_pct,
        "History": hist,
    }


def short_passes_global_filters(row: dict, min_price: float, max_market_cap: float, excluded_sectors: list[str], excluded_countries: list[str]) -> bool:
    return (
        row["Current Price"] > min_price
        and row["Market Cap"] < max_market_cap
        and row["Sector"] not in excluded_sectors
        and row["Country"] not in excluded_countries
    )


def short_scan_universe(tickers: list[str], min_price: float, max_market_cap: float, excluded_sectors: list[str], excluded_countries: list[str]) -> list[dict]:
    rows = []
    for ticker in normalize_tickers(tickers):
        metrics = short_base_metrics(ticker)
        if metrics and short_passes_global_filters(metrics, min_price, max_market_cap, excluded_sectors, excluded_countries):
            rows.append(metrics)
    return rows


def gap_up_short_scanner(rows: list[dict]) -> pd.DataFrame:
    output = []
    for row in rows:
        premarket_volume = min(50_000_000, max(1_000_000, row["Volume"] * (1.2 + row["Relative Volume"] / 2)))
        post_open_push = (row["High After Open"] / row["Opening Price"] - 1) * 100
        consolidation = "Target push range" if 20 <= post_open_push <= 35 else "Watching"
        crack_alert = row["Current Price"] < row["1H Low"] and row["Relative Volume"] > 1.5
        score = min(100, row["Gap %"] * 0.25 + post_open_push * 1.2 + row["Relative Volume"] * 12 + (25 if crack_alert else 0))
        if row["Gap %"] >= 100 and 1_000_000 <= premarket_volume <= 50_000_000:
            output.append({
                "Ticker": row["Ticker"], "Current Price": row["Current Price"], "Gap %": row["Gap %"],
                "Pre-market Volume": premarket_volume, "Market Cap": row["Market Cap"], "Float": row["Float"],
                "Post-Open Push %": post_open_push, "1H High": row["1H High"], "1H Low": row["1H Low"],
                "Consolidation Status": consolidation, "Crack Alert": "Alert" if crack_alert else "Watching",
                "Short Setup Quality Score": score,
            })
    return pd.DataFrame(output).sort_values("Short Setup Quality Score", ascending=False) if output else pd.DataFrame()


def bounce_short_scanner(rows: list[dict]) -> pd.DataFrame:
    output = []
    for row in rows:
        hist = row["History"].copy()
        hist["Dollar Block"] = hist["Close"] * hist["Volume"]
        block = hist.sort_values("Dollar Block", ascending=False).iloc[0]
        resistance = float(block["High"])
        distance = (resistance / row["Current Price"] - 1) * 100
        ratio = float(block["Volume"]) / max(row["Volume"], 1)
        alert = distance <= 20 and ratio >= 2
        score = min(100, max(0, 45 - abs(distance)) + ratio * 12 + row["Gap %"] * 0.12 + (20 if alert else 0))
        output.append({
            "Ticker": row["Ticker"], "Current Price": row["Current Price"], "Historical Resistance Price": resistance,
            "Distance to Resistance %": distance, "Historical Dollar Block": float(block["Dollar Block"]),
            "Historical Volume": float(block["Volume"]), "Current Estimated Daily Volume": row["Volume"],
            "Historical Volume / Current Volume Ratio": ratio, "Gap %": row["Gap %"],
            "Resistance Alert": "Alert" if alert else "Watching", "Short Setup Quality Score": score,
        })
    return pd.DataFrame(output).sort_values("Short Setup Quality Score", ascending=False) if output else pd.DataFrame()


def first_red_day_scanner(rows: list[dict]) -> pd.DataFrame:
    output = []
    for row in rows:
        hist = row["History"].copy()
        closes = hist["Close"].tail(8)
        green = closes.diff() > 0
        consecutive = 0
        for value in reversed(green.tolist()):
            if value:
                consecutive += 1
            else:
                break
        last4 = hist.tail(4)
        price_increase = (last4["Close"].iloc[-1] / last4["Close"].iloc[0] - 1) * 100 if len(last4) >= 4 else 0
        increasing_volume = bool(last4["Volume"].is_monotonic_increasing) if len(last4) >= 4 else False
        total_run_dollar_volume = float((last4["Close"] * last4["Volume"]).sum())
        initial_market_cap = max(row["Market Cap"], 1)
        dv_ratio = total_run_dollar_volume / initial_market_cap
        previous_day_low = float(hist["Low"].iloc[-2])
        first_red_trigger = row["Current Price"] < previous_day_low or hist["Close"].iloc[-1] < hist["Close"].iloc[-2]
        exhaustion_score = min(100, dv_ratio * 30 + price_increase * 0.18 + (15 if not increasing_volume else 0))
        score = min(100, exhaustion_score + (20 if first_red_trigger else 0) + max(0, consecutive - 2) * 8)
        if consecutive >= 3 or price_increase > 120:
            output.append({
                "Ticker": row["Ticker"], "Current Price": row["Current Price"], "Consecutive Green Days": consecutive,
                "3-Day Price Increase %": price_increase, "Increasing Volume Status": "Increasing" if increasing_volume else "Fading/Mixed",
                "Total Run Dollar Volume": total_run_dollar_volume, "Initial Market Cap": initial_market_cap,
                "Dollar Volume / Initial Market Cap": dv_ratio, "Previous Day Low": previous_day_low,
                "First Red Day Trigger": "Trigger" if first_red_trigger else "Watching",
                "Exhaustion Score": exhaustion_score, "Short Setup Quality Score": score,
            })
    return pd.DataFrame(output).sort_values("Short Setup Quality Score", ascending=False) if output else pd.DataFrame()


def format_short_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    formats = {
        "Current Price": "${:.2f}", "Gap %": "{:.1f}%", "% Change": "{:.1f}%", "Pre-market Volume": "{:,.0f}",
        "Market Cap": "${:,.0f}", "Float": "{:,.0f}", "Post-Open Push %": "{:.1f}%", "1H High": "${:.2f}",
        "1H Low": "${:.2f}", "Short Setup Quality Score": "{:.1f}", "Historical Resistance Price": "${:.2f}",
        "Distance to Resistance %": "{:.1f}%", "Historical Dollar Block": "${:,.0f}", "Historical Volume": "{:,.0f}",
        "Current Estimated Daily Volume": "{:,.0f}", "Historical Volume / Current Volume Ratio": "{:.2f}",
        "3-Day Price Increase %": "{:.1f}%", "Total Run Dollar Volume": "${:,.0f}", "Initial Market Cap": "${:,.0f}",
        "Dollar Volume / Initial Market Cap": "{:.2f}", "Previous Day Low": "${:.2f}", "Exhaustion Score": "{:.1f}",
    }
    return df.style.format({col: fmt for col, fmt in formats.items() if col in df.columns})


def short_strategy_chart(ticker: str, rows: list[dict]) -> go.Figure:
    match = next((row for row in rows if row["Ticker"] == ticker), None)
    hist = match["History"] if match else short_market_frame(ticker)
    fig = price_chart(hist.tail(90), ticker)
    if match:
        overlays = {
            "Previous close": match["Previous Close"],
            "Pre-market high": match["Pre-market Price"],
            "Opening price": match["Opening Price"],
            "1H high": match["1H High"],
            "1H low / crack": match["1H Low"],
        }
        for label, value in overlays.items():
            fig.add_hline(y=value, line_dash="dot", annotation_text=label)
    fig.update_layout(height=460)
    return fig


def page_short_strategy_scanner(selected: list[str]) -> None:
    st.header("Short Strategy Stock Scanner")
    st.warning("Educational and research use only. This scanner does not provide financial advice and does not execute trades.")
    universe_text = st.text_area("Short scanner universe", value=", ".join(normalize_tickers([*SHORT_DEFAULT_UNIVERSE, *selected])[:50]), height=80)
    c1, c2, c3, c4 = st.columns(4)
    min_price = c1.number_input("Minimum price", min_value=0.5, value=3.0, step=0.5)
    max_market_cap = c2.number_input("Maximum market cap", min_value=10_000_000.0, value=200_000_000.0, step=10_000_000.0, format="%.0f")
    refresh_seconds = c3.selectbox("Refresh interval", [0, 60, 120, 300], index=0, format_func=lambda x: "Manual" if x == 0 else f"{x}s")
    c4.metric("API mode", "Mock/Yahoo fallback")
    if refresh_seconds and st_autorefresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="short_strategy_refresh")
    excluded_sectors = st.multiselect("Excluded sectors", ["Biotech", "Energy", "Healthcare", "Technology", "Industrials", "Consumer Cyclical"], default=["Biotech", "Energy"])
    excluded_countries = st.multiselect("Excluded countries", ["China", "Hong Kong", "Cayman Islands", "Singapore"], default=["China", "Hong Kong", "Cayman Islands"])
    tickers = normalize_tickers(universe_text.split(","))
    if st.button("Refresh short scanners", type="primary"):
        st.session_state["short_strategy_rows"] = short_scan_universe(tickers, min_price, max_market_cap, excluded_sectors, excluded_countries)
    rows = st.session_state.get("short_strategy_rows")
    if rows is None:
        rows = short_scan_universe(tickers, min_price, max_market_cap, excluded_sectors, excluded_countries)
        st.session_state["short_strategy_rows"] = rows
    if not rows:
        st.info("No symbols passed the global short-strategy filters. Broaden the universe or filters.")
        return
    gap = gap_up_short_scanner(rows)
    bounce = bounce_short_scanner(rows)
    first_red = first_red_day_scanner(rows)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered symbols", len(rows))
    m2.metric("Gap candidates", len(gap))
    m3.metric("Bounce candidates", len(bounce))
    m4.metric("First red candidates", len(first_red))
    tab_gap, tab_bounce, tab_red, tab_watch = st.tabs(["Gap Up Short", "Bounce Short", "First Red Day", "Watchlist"])
    with tab_gap:
        st.subheader("Gap Up Short Scanner")
        if gap.empty:
            st.info("No gap-up short candidates passed the current thresholds.")
        else:
            st.dataframe(format_short_table(gap), use_container_width=True, hide_index=True)
            st.download_button("Export Gap Up CSV", gap.to_csv(index=False), "gap_up_short_scanner.csv", "text/csv")
    with tab_bounce:
        st.subheader("Bounce Short Scanner")
        st.dataframe(format_short_table(bounce), use_container_width=True, hide_index=True)
        st.download_button("Export Bounce CSV", bounce.to_csv(index=False), "bounce_short_scanner.csv", "text/csv")
    with tab_red:
        st.subheader("First Red Day Scanner")
        if first_red.empty:
            st.info("No first-red-day candidates matched yet.")
        else:
            st.dataframe(format_short_table(first_red), use_container_width=True, hide_index=True)
            st.download_button("Export First Red Day CSV", first_red.to_csv(index=False), "first_red_day_scanner.csv", "text/csv")
    with tab_watch:
        st.subheader("Active Watchlist Monitor")
        watch_text = st.text_area("Watchlist input", value=", ".join(tickers[:10]), height=80, key="short_watchlist_input")
        strategy = st.selectbox("Strategy type for new rows", ["Gap Up Short", "Bounce Short", "First Red Day"])
        watch_rows = []
        for ticker in normalize_tickers(watch_text.split(",")):
            match = next((row for row in rows if row["Ticker"] == ticker), short_base_metrics(ticker))
            if not match:
                continue
            crack = match["Current Price"] < match["1H Low"]
            push = (match["High After Open"] / match["Opening Price"] - 1) * 100
            status = "Crack/trigger alert" if crack else "Active opportunity" if 20 <= push <= 35 else "Watching"
            watch_rows.append({
                "Ticker": ticker, "Strategy Type": strategy, "Entry Watch Level": match["Opening Price"],
                "Consolidation High": match["1H High"], "Consolidation Low": match["1H Low"],
                "Crack Level": match["1H Low"], "Current Price": match["Current Price"],
                "Volume Trend": "Strong RVOL" if match["Relative Volume"] > 1.5 else "Normal",
                "Alert Status": status, "Notes": "Monitor only. No automatic trading.",
            })
        watch_df = pd.DataFrame(watch_rows)
        if not watch_df.empty:
            st.dataframe(format_short_table(watch_df), use_container_width=True, hide_index=True)
            st.download_button("Export Watchlist CSV", watch_df.to_csv(index=False), "short_watchlist_monitor.csv", "text/csv")
            chart_ticker = st.selectbox("Chart ticker", watch_df["Ticker"].tolist())
            st.plotly_chart(short_strategy_chart(chart_ticker, rows), use_container_width=True)


def page_market_overview(selected: list[str]) -> None:
    st.header("Market Overview")
    disclaimer()
    overview = fetch_last_prices(["SPY", "QQQ", "DIA", "IWM", *selected[:12]])
    if overview.empty:
        st.warning(market_data_healthcheck())
        st.caption("The dashboard is working, but live price data could not be downloaded.")
        return
    cols = st.columns(4)
    for idx, row in overview.head(4).iterrows():
        cols[idx % 4].metric(row["Ticker"], f"${row['Price']:.2f}", f"{row['Daily Change %']:.2f}%")
    st.dataframe(format_scan_table(overview), use_container_width=True, hide_index=True)


def page_multi_stock_scanner(selected: list[str], allow_penny: bool) -> pd.DataFrame:
    st.header("Multi-Stock AI Scanner")
    scan_size = st.select_slider("Scan size", options=[5, 10, 20, 50, 100], value=50)
    universe = scanner_universe(selected, scan_size)
    st.caption(f"Ready to scan {len(universe)} stocks. Selected symbols are prioritized, then the built-in liquid-stock universe fills the rest.")
    f1, f2, f3 = st.columns(3)
    high_conviction_only = f1.toggle(
        "High Conviction Only",
        value=False,
        help="Shows only Strong Buy rows where RSI is 60-75, MACD is Bullish, and price is Above VWAP.",
    )
    hide_weak_buys = f2.toggle(
        "Skip Buy below 70% AI",
        value=True,
        help="Hides Buy rows when AI bullish probability is below 70%.",
    )
    ai_sort = f3.toggle("Sort by AI conviction", value=True)
    record_run = st.checkbox("Record this scanner run for signal accuracy analysis", value=True)
    archive_run = st.checkbox("Save full scanner table to daily one-year reference archive", value=True)
    run = st.button("Run scanner", type="primary")
    if not run:
        st.info("Choose a scan size and run the scanner.")
        render_scanner_history_reference()
        return pd.DataFrame()
    with st.spinner("Scanning selected stocks..."):
        table = scan_stocks(universe, limit=scan_size, allow_penny_stocks=allow_penny)
    if table.empty:
        st.warning("No scan results were available.")
        render_scanner_history_reference()
        return table
    table = add_signal_synergy_columns(table)
    display_table = table.copy()
    weak_buy_mask = display_table["Signal quality note"].astype(str).str.contains("below 70", case=False, na=False)
    weak_buy_count = int(weak_buy_mask.sum())
    if hide_weak_buys:
        display_table = display_table[~weak_buy_mask].copy()
    if high_conviction_only:
        display_table = display_table[display_table["Synergy Strong Buy"]].copy()
    if ai_sort and "AI bullish probability" in display_table:
        display_table = display_table.sort_values(["AI bullish probability", "Suitability score"], ascending=False)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scanned", len(table))
    m2.metric("Displayed", len(display_table))
    m3.metric("Synergy Strong Buys", int(table["Synergy Strong Buy"].sum()))
    m4.metric("Buy below 70% AI", weak_buy_count)
    if weak_buy_count:
        st.warning(f"{weak_buy_count} Buy signal(s) have AI bullish probability below 70%. Keep the skip toggle enabled to hide lower-conviction buys.")
    if display_table.empty:
        st.info("No rows match the current scanner filters.")
    else:
        st.dataframe(format_scan_table(display_table), use_container_width=True, hide_index=True)
    if record_run:
        saved = record_daily_rows(display_table if not display_table.empty else table, "Multi-Stock AI Scanner")
        st.success(f"Recorded {saved} scanner signal(s) for today's analysis journal.")
    if archive_run:
        archived = record_scanner_history(table, scan_size)
        st.success(f"Saved {archived} scanner row(s) to the daily reference archive.")
    st.download_button("Download scanner results", display_table.to_csv(index=False), "scanner_results.csv", "text/csv")
    render_scanner_history_reference()
    return display_table


def page_options_ai_scanner(selected: list[str], allow_penny: bool) -> None:
    st.header("Options AI Scanner")
    st.warning("Options scanner is for education and Alpaca paper trading only. It does not guarantee fills or profit.")
    c1, c2, c3, c4 = st.columns(4)
    scan_size = c1.selectbox("Underlying scan size", [5, 10, 20, 50], index=2)
    min_dte = c2.number_input("Minimum DTE", min_value=1, max_value=180, value=14, step=1)
    max_dte = c3.number_input("Maximum DTE", min_value=2, max_value=365, value=45, step=1)
    max_moneyness = c4.number_input("Max OTM/ITM distance %", min_value=1.0, max_value=50.0, value=8.0, step=0.5)
    st.caption("The scanner maps bullish stock signals to call candidates and bearish/avoid signals to put candidates, then ranks near-the-money active Alpaca contracts.")
    if st.button("Run options AI scanner", type="primary"):
        with st.spinner("Scanning stocks and matching option contracts..."):
            table = build_options_ai_scanner(selected, scan_size, int(min_dte), int(max_dte), float(max_moneyness), allow_penny)
        if table.empty:
            st.warning("No option candidates found. Check Alpaca credentials/secrets, option permissions, or widen the DTE/moneyness filters.")
            return
        st.session_state["last_options_scan"] = table
        display_cols = [
            "Option Symbol", "Underlying", "Underlying Price", "Type", "Strike", "Expiration", "DTE",
            "Moneyness %", "Underlying Signal", "AI bullish probability", "AI bearish probability",
            "Suitability score", "Risk-reward ratio", "Options AI score", "Suggested side",
        ]
        st.dataframe(
            table[display_cols].style.format(
                {
                    "Underlying Price": "${:.2f}",
                    "Strike": "${:.2f}",
                    "Moneyness %": "{:.2f}%",
                    "AI bullish probability": "{:.1f}%",
                    "AI bearish probability": "{:.1f}%",
                    "Risk-reward ratio": "{:.2f}",
                    "Options AI score": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button("Download option candidates", table.to_csv(index=False), "options_ai_scanner.csv", "text/csv")


def page_individual_analysis(selected: list[str], allow_penny: bool) -> None:
    st.header("Individual Stock Analysis")
    ticker = st.selectbox("Ticker", selected or DEFAULT_WATCHLIST)
    with st.spinner(f"Analyzing {ticker}..."):
        result = analyze_stock(ticker, allow_penny_stocks=allow_penny)
    if "Error" in result:
        st.error(result["Error"])
        return
    suitability = result["Suitability detail"]
    ai = result["AI detail"]
    cols = st.columns(5)
    cols[0].metric("Price", f"${result['Price']:.2f}", f"{result['Daily Change %']:.2f}%")
    cols[1].metric("Suitability", result["Suitability"], f"{result['Suitability score']}/100")
    cols[2].metric("Bullish Probability", f"{ai['bullish_probability']:.1f}%")
    cols[3].metric("Bearish Probability", f"{ai['bearish_probability']:.1f}%")
    cols[4].metric("Final Signal", result["Final signal"])
    st.write("Suitability check:", result["Suitability reasons"])
    df = result["Data"].copy()
    latest = df.dropna().iloc[-1]
    session_frame = session_vwap_frame(ticker)
    session_vwap_value = None
    if not session_frame.empty and "Session_VWAP" in session_frame:
        session_vwap_value = float(session_frame["Session_VWAP"].dropna().iloc[-1])
    anchor_mode = st.selectbox("Anchored VWAP anchor", ["Year start", "Swing low", "Earnings/manual date"])
    if anchor_mode == "Year start":
        anchor_date = pd.Timestamp.now().replace(month=1, day=1)
    elif anchor_mode == "Swing low":
        clean = df.dropna(subset=["Low"])
        anchor_date = pd.Timestamp(clean["Low"].tail(252).idxmin()) if not clean.empty else pd.Timestamp.now()
    else:
        anchor_date = pd.Timestamp(
            st.date_input(
                "Earnings/manual anchor date",
                value=(pd.Timestamp.now() - pd.Timedelta(days=90)).date(),
            )
        )
    anchored_vwap, actual_anchor = anchored_vwap_series(df, anchor_date)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EMA 20", f"${latest.get('EMA_20', 0):.2f}")
    c2.metric("EMA 60", f"${latest.get('EMA_60', 0):.2f}")
    c3.metric("EMA 250", f"${latest.get('EMA_250', 0):.2f}")
    c4.metric("Daily Session VWAP", f"${session_vwap_value:.2f}" if session_vwap_value is not None else "Unavailable")
    c5.metric("Anchored VWAP", f"${anchored_vwap.dropna().iloc[-1]:.2f}" if not anchored_vwap.dropna().empty else "Unavailable")
    st.caption(
        "The old daily cumulative VWAP is shown as Cumulative VWAP. Daily Session VWAP uses intraday bars when the data provider returns them."
        + (f" Anchored VWAP start: {actual_anchor.date().isoformat()}." if actual_anchor is not None else "")
    )
    st.plotly_chart(individual_analysis_chart(df.tail(300), ticker, session_vwap_value, anchored_vwap), use_container_width=True)
    if not session_frame.empty:
        with st.expander("Intraday session VWAP data"):
            st.dataframe(session_frame.tail(120), use_container_width=True)


def render_sector_divergence_panel(history: pd.DataFrame) -> None:
    if history.empty or "Ticker" not in history:
        return
    semiconductors = ["AMAT", "SNDK", "INTC"]
    software_consumer = ["AAPL", "ABBV"]
    data = history.copy()
    data["Record date"] = pd.to_datetime(data["Record date"], errors="coerce")
    if "Period" in data:
        data = data[data["Period"].astype(str) == "Daily"].copy()
    data = data[data["Ticker"].astype(str).isin([*semiconductors, *software_consumer])]
    if data.empty:
        st.info("Sector divergence needs saved Daily rows for AMAT, SNDK, INTC, AAPL, and ABBV.")
        return
    latest_date = data["Record date"].max()
    latest = data[data["Record date"] == latest_date].copy()
    latest["Group"] = latest["Ticker"].map(lambda ticker: "Semiconductors" if ticker in semiconductors else "Software / Consumer Tech")
    macd_series = latest.get("MACD signal", pd.Series(index=latest.index, dtype=str)).astype(str)
    trend_series = latest.get("Trend", pd.Series(index=latest.index, dtype=str)).astype(str)
    latest["MACD bearish"] = macd_series.str.contains("Bearish", case=False, na=False)
    latest["MACD bullish"] = macd_series.str.contains("Bullish", case=False, na=False)
    latest["Downtrend"] = trend_series.str.contains("Downtrend", case=False, na=False)
    latest["Uptrend"] = trend_series.str.contains("Uptrend", case=False, na=False)
    summary = (
        latest.groupby("Group")
        .agg(
            Tickers=("Ticker", "count"),
            Bearish_MACD=("MACD bearish", "mean"),
            Bullish_MACD=("MACD bullish", "mean"),
            Downtrend=("Downtrend", "mean"),
            Uptrend=("Uptrend", "mean"),
            Avg_AI_Bullish=("AI bullish probability", "mean"),
        )
        .reset_index()
    )
    st.subheader("Sector Divergence")
    st.caption("Compares Semiconductor tickers AMAT/SNDK/INTC against Software/Consumer Tech tickers AAPL/ABBV using the latest saved Daily technical snapshots.")
    st.dataframe(
        summary.style.format(
            {
                "Bearish_MACD": "{:.0%}",
                "Bullish_MACD": "{:.0%}",
                "Downtrend": "{:.0%}",
                "Uptrend": "{:.0%}",
                "Avg_AI_Bullish": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    semi = summary[summary["Group"] == "Semiconductors"]
    software = summary[summary["Group"] == "Software / Consumer Tech"]
    if not semi.empty and not software.empty:
        semi_bearish = float(semi["Bearish_MACD"].iloc[0]) >= 0.5 and float(semi["Downtrend"].iloc[0]) >= 0.5
        software_bullish = float(software["Bullish_MACD"].iloc[0]) >= 0.5 and float(software["Uptrend"].iloc[0]) >= 0.5
        if semi_bearish and software_bullish:
            st.warning("Divergence alert: Semiconductors are mostly Bearish/Downtrend while Software/Consumer Tech remains Bullish/Uptrend. Treat semiconductor Buy signals as dip buys that need confirmation.")
    st.dataframe(latest[["Record date", "Group", "Ticker", "MACD signal", "Trend", "VWAP status", "Final signal", "AI bullish probability"]], use_container_width=True, hide_index=True)


def page_technical_dashboard(selected: list[str]) -> None:
    st.header("Technical Indicator Dashboard")
    available = selected or DEFAULT_WATCHLIST
    ticker = st.selectbox("Technical ticker", available)
    analysis = analyze_stock(ticker)
    if "Error" in analysis:
        st.warning(analysis["Error"])
        return
    df = analysis["Data"].dropna()
    if df.empty:
        st.warning("No technical data available.")
        return
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI", f"{latest['RSI']:.1f}")
    c2.metric("MACD", f"{latest['MACD']:.2f}", f"Signal {latest['MACD_Signal']:.2f}")
    c3.metric("ATR %", f"{latest['ATR_Pct']:.2f}%")
    c4.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig.add_hline(y=70, line_dash="dot")
    fig.add_hline(y=30, line_dash="dot")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.tail(60), use_container_width=True)

    st.subheader("Technical Snapshot Recorder")
    record_tickers = st.multiselect(
        "Stocks to record",
        available,
        default=available,
        help="Uses the stocks selected in the sidebar. Records are saved locally in logs/technical_indicator_snapshots.csv.",
    )
    record_periods = st.multiselect(
        "Record periods",
        ["Daily", "Weekly", "Monthly"],
        default=["Daily", "Weekly", "Monthly"],
    )
    if st.button("Record selected stocks technical snapshots"):
        if not record_tickers:
            st.warning("Select at least one stock to record.")
        elif not record_periods:
            st.warning("Select at least one period to record.")
        else:
            with st.spinner(f"Recording {len(record_tickers)} stock(s) across {len(record_periods)} period(s)..."):
                saved_rows, errors = record_technical_snapshots(record_tickers, record_periods)
            if not saved_rows.empty:
                st.success(f"Recorded {len(saved_rows)} technical snapshot rows.")
                st.dataframe(saved_rows, use_container_width=True, hide_index=True)
            if errors:
                st.warning("Some tickers could not be recorded: " + "; ".join(errors[:8]))

    history = load_technical_snapshots()
    st.subheader("Daily Calendar Archive")
    st.caption("Target: every calendar date from 2026-06-26 through 2026-12-31, including weekends. Weekend rows use the latest available market data and are marked as market closed.")
    calendar_tickers = st.multiselect("Calendar archive tickers", available, default=available[:50])
    calendar_history = load_technical_snapshots()
    missing_dates = missing_calendar_dates(calendar_history, calendar_tickers[:50])
    expected_dates = calendar_archive_dates()
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Expected calendar dates", len(expected_dates))
    cc2.metric("Saved calendar dates", max(0, len(expected_dates) - len(missing_dates)))
    cc3.metric("Missing calendar dates", len(missing_dates))
    auto_calendar = st.checkbox("Auto save today's calendar technical row", value=True)
    today_iso = pd.Timestamp.now().date().isoformat()
    if auto_calendar and today_iso in missing_dates and calendar_tickers:
        with st.spinner("Saving today's calendar technical rows..."):
            saved_rows, errors = record_technical_snapshots(calendar_tickers[:50], ["Daily"], record_date=today_iso)
        if not saved_rows.empty:
            st.success(f"Saved today's calendar archive rows: {len(saved_rows)}.")
        if errors:
            st.warning("Some calendar rows could not be saved: " + "; ".join(errors[:8]))
        history = load_technical_snapshots()
        missing_dates = missing_calendar_dates(history, calendar_tickers[:50])
    if st.button("Backfill missing calendar dates from 2026-06-26 to today"):
        if not calendar_tickers:
            st.warning("Select at least one ticker for the calendar archive.")
        elif not missing_dates:
            st.success("No missing calendar dates to backfill.")
        else:
            with st.spinner(f"Backfilling {len(missing_dates)} calendar date(s) for up to 50 tickers..."):
                filled_rows, errors = backfill_calendar_technical_snapshots(calendar_tickers[:50], missing_dates)
            if not filled_rows.empty:
                st.success(f"Backfilled {len(filled_rows)} technical calendar rows.")
            if errors:
                st.warning("Some calendar rows could not be saved: " + "; ".join(errors[:8]))
            history = load_technical_snapshots()
            missing_dates = missing_calendar_dates(history, calendar_tickers[:50])
    if missing_dates:
        st.caption("Missing dates: " + ", ".join(missing_dates[:20]) + ("..." if len(missing_dates) > 20 else ""))

    if not history.empty:
        st.subheader("Technical Snapshot Journal")
        c1, c2, c3 = st.columns(3)
        range_preset = c1.selectbox(
            "Technical journal range",
            ["Last 1 month", "Last 3 months", "Last 6 months", "Last 1 year", "Last 2 years", "Last 5 years", "All history"],
            index=3,
        )
        period_filter = c2.multiselect("Journal period filter", ["Daily", "Weekly", "Monthly"], default=["Daily", "Weekly", "Monthly"])
        ticker_options = sorted(history["Ticker"].dropna().astype(str).unique().tolist()) if "Ticker" in history else []
        ticker_filter = c3.multiselect("Journal tickers", ticker_options, default=ticker_options[:20])
        shown = filter_history_range(history, "Record date", range_preset)
        shown = shown[shown["Period"].isin(period_filter)] if period_filter else shown
        if ticker_filter:
            shown = shown[shown["Ticker"].astype(str).isin(ticker_filter)]
        if shown.empty:
            st.info("No technical snapshot rows match the selected filters.")
            return
        shown = shown.sort_values(["Record date", "Period", "Ticker"], ascending=[False, True, True])
        render_sector_divergence_panel(shown)
        row_limit = st.selectbox("Rows to display", [300, 1000, 5000, "All"], index=1)
        display_rows = shown if row_limit == "All" else shown.head(int(row_limit))
        m1, m2, m3 = st.columns(3)
        m1.metric("Archived rows", f"{len(shown):,}")
        m2.metric("Archived dates", shown["Record date"].astype(str).nunique())
        m3.metric("Archived tickers", shown["Ticker"].astype(str).nunique() if "Ticker" in shown else 0)
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Download technical snapshot journal",
            shown.to_csv(index=False).encode("utf-8"),
            file_name="technical_indicator_snapshots.csv",
            mime="text/csv",
        )


def page_ai_prediction(selected: list[str]) -> None:
    st.header("AI Prediction Model")
    ticker = st.selectbox("Model ticker", selected or DEFAULT_WATCHLIST)
    df = prepare_stock(ticker)
    with st.spinner("Training with time-series split..."):
        result = train_models(df)
    if "error" in result:
        st.warning(result["error"])
        return
    rows = [{"Model": name, "CV Accuracy": detail["cv_accuracy"]} for name, detail in result["models"].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"CV Accuracy": "{:.2%}"}), use_container_width=True, hide_index=True)
    rf = result["models"].get("Random Forest", {}).get("model")
    if rf is not None:
        importance = pd.DataFrame({"Feature": result["features"], "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature").head(15))
    st.caption("Target: 5-day future return labels of bullish > 1.5%, bearish < -1.5%, otherwise neutral. Training uses time-series splits to reduce look-ahead bias.")


def page_backtesting(selected: list[str]) -> None:
    st.header("Backtesting")
    tickers = st.multiselect("Backtest stocks", selected or DEFAULT_WATCHLIST, default=(selected or DEFAULT_WATCHLIST)[:5])
    cost = st.number_input("Transaction cost", min_value=0.0, max_value=0.02, value=0.001, step=0.0005, format="%.4f")
    slippage = st.number_input("Slippage", min_value=0.0, max_value=0.02, value=0.001, step=0.0005, format="%.4f")
    if not st.button("Run backtest", type="primary"):
        return
    frames = {ticker: prepare_stock(ticker) for ticker in tickers}
    stats = []
    for ticker, df in frames.items():
        result = backtest_signal_strategy(df, cost, slippage)
        if "error" not in result:
            stats.append({"Ticker": ticker, **{k: v for k, v in result.items() if not k.endswith("_curve")}})
    if stats:
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
    portfolio = backtest_portfolio(frames, cost, slippage)
    if "error" not in portfolio:
        st.subheader("Portfolio Strategy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", f"{portfolio['total_return']:.2%}")
        c2.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
        c3.metric("Maximum Drawdown", f"{portfolio['max_drawdown']:.2%}")
        st.line_chart(pd.DataFrame({"Equity": portfolio["equity_curve"], "Drawdown": portfolio["drawdown_curve"]}))


def page_paper_trading(selected: list[str], allow_penny: bool) -> None:
    st.header("Paper Trading Bot")
    st.warning("Paper trading only. Emergency stop disables scan/order actions in this session.")
    st.caption(f"Stock paper account key: {masked_account_key('stock')}")
    active = st.multiselect("Active bot stocks", selected or DEFAULT_WATCHLIST, default=(selected or DEFAULT_WATCHLIST)[:5])
    capital = st.number_input("Paper capital", min_value=1000.0, value=25000.0, step=1000.0)
    auto_scan = st.toggle("Auto-scan every 5 minutes", value=False)
    emergency_stop = st.toggle("Emergency stop", value=False)
    if auto_scan and st_autorefresh is not None:
        st_autorefresh(interval=5 * 60 * 1000, key="paper_bot_refresh")
    elif auto_scan:
        st.info("Install streamlit-autorefresh to enable timed browser refreshes.")
    if st.button("Scan and log paper signals", type="primary", disabled=emergency_stop):
        table = scan_stocks(active, limit=len(active), allow_penny_stocks=allow_penny)
        st.dataframe(format_scan_table(table), use_container_width=True, hide_index=True)
        for _, row in table.iterrows():
            log_signal(row.to_dict())
            if row["Final signal"] in {"Strong Buy", "Buy"}:
                qty = position_size(capital, row["Buy zone"], row["Stop loss"])
                st.write(f"{row['Ticker']}: eligible paper buy size = {qty} shares.")
    ticker = st.selectbox("Manual paper order ticker", active or DEFAULT_WATCHLIST)
    qty = st.number_input("Manual paper quantity", min_value=0, value=0, step=1)
    if st.button("Submit Alpaca paper buy", disabled=emergency_stop):
        result = place_paper_order(ticker, qty, side="buy")
        (st.success if result["ok"] else st.error)(result["message"])
    st.subheader("Open Paper Orders")
    open_orders = open_orders_table(ticker)
    if open_orders.empty:
        st.caption("No open paper orders for the selected ticker.")
    else:
        st.dataframe(open_orders, use_container_width=True, hide_index=True)
        cancel_id = st.selectbox("Cancel open order", open_orders["id"].tolist())
        if st.button("Cancel selected paper order", disabled=emergency_stop):
            result = cancel_order(cancel_id)
            (st.success if result["ok"] else st.error)(result["message"])
            st.rerun()
    if TRADE_LOG_FILE.exists():
        st.subheader("Trade Log")
        st.dataframe(pd.read_csv(TRADE_LOG_FILE).tail(100), use_container_width=True)
    if SIGNAL_LOG_FILE.exists():
        st.subheader("Signal Log")
        st.dataframe(pd.read_csv(SIGNAL_LOG_FILE).tail(100), use_container_width=True)


def page_options_paper_trading() -> None:
    st.header("Options Paper Trading")
    st.warning("Paper options only. Use limit orders. Options can expire worthless and may be illiquid.")
    st.caption(f"Options paper account key: {masked_account_key('options')}")
    available_options_bp = account_buying_power("options")
    st.metric("Available options buying power", f"${available_options_bp:,.2f}")
    order_tab, wheel_tab = st.tabs(["Single-Leg Option Order", "Wheel Strategy"])
    with order_tab:
        render_single_leg_option_order(available_options_bp)
    with wheel_tab:
        render_wheel_strategy(selected=st.session_state.get("selected_stocks_for_options", DEFAULT_WATCHLIST[:20]), available_options_bp=available_options_bp)


def render_single_leg_option_order(available_options_bp: float) -> None:
    last_scan = st.session_state.get("last_options_scan", pd.DataFrame())
    scanned_symbols = last_scan["Option Symbol"].dropna().astype(str).tolist() if not last_scan.empty and "Option Symbol" in last_scan else []
    mode = st.radio("Contract source", ["Use scanner result", "Manual option symbol"], horizontal=True)
    if mode == "Use scanner result" and scanned_symbols:
        option_symbol = st.selectbox("Option contract", scanned_symbols)
        selected_row = last_scan[last_scan["Option Symbol"] == option_symbol].iloc[0]
        st.dataframe(pd.DataFrame([selected_row]), use_container_width=True, hide_index=True)
    else:
        option_symbol = st.text_input("Option symbol", placeholder="Example: AAPL260116C00310000").strip().upper()
        if mode == "Use scanner result" and not scanned_symbols:
            st.info("Run the Options AI Scanner first, or switch to manual option symbol.")
    c1, c2, c3 = st.columns(3)
    qty = c1.number_input("Contracts", min_value=0, value=0, step=1)
    limit_price = c2.number_input("Limit price per contract", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    side = c3.selectbox("Side", ["buy", "sell"], index=0)
    requirement = option_order_requirement(option_symbol, int(qty), float(limit_price), side) if option_symbol else {"ok": False, "required": 0.0, "message": "Enter an option symbol."}
    m1, m2 = st.columns(2)
    m1.metric("Estimated requirement", f"${requirement['required']:,.2f}")
    m2.metric("Requirement type", requirement["message"])
    if side == "sell":
        st.info("Selling puts requires cash-secured collateral of strike x 100 x contracts. Selling uncovered calls is disabled by this app.")
    if option_symbol and qty > 0 and limit_price > 0 and requirement["required"] > available_options_bp:
        st.error(f"Not enough options buying power: required about ${requirement['required']:,.2f}, available about ${available_options_bp:,.2f}.")
    emergency_stop = st.toggle("Emergency stop", value=False, key="options_emergency_stop")
    can_submit = not emergency_stop and bool(option_symbol) and requirement["ok"] and requirement["required"] <= available_options_bp
    if st.button("Submit Alpaca paper option order", type="primary", disabled=not can_submit):
        result = place_option_paper_order(option_symbol, int(qty), float(limit_price), side=side)
        (st.success if result["ok"] else st.error)(result["message"])
    st.subheader("Open Option Paper Orders")
    open_orders = open_option_orders_table(option_symbol if option_symbol else None)
    if open_orders.empty:
        st.caption("No open paper orders for this option symbol.")
    else:
        st.dataframe(open_orders, use_container_width=True, hide_index=True)
        cancel_id = st.selectbox("Cancel option order", open_orders["id"].tolist())
        if st.button("Cancel selected option order", disabled=emergency_stop):
            result = cancel_option_order(cancel_id)
            (st.success if result["ok"] else st.error)(result["message"])
            st.rerun()


def render_wheel_strategy(selected: list[str], available_options_bp: float) -> None:
    st.subheader("Wheel Strategy: Cash-Secured Put Scanner")
    st.caption("Step 1 of the wheel: sell cash-secured puts on suitable stocks you are willing to own. Defaults target 30 DTE and 0.15 absolute delta.")
    c1, c2, c3, c4 = st.columns(4)
    scan_size = c1.selectbox("Wheel underlying scan size", [5, 10, 20, 50], index=2)
    target_dte = c2.number_input("Target DTE", min_value=14, max_value=45, value=30, step=1)
    min_dte = c3.number_input("Min DTE", min_value=1, max_value=44, value=14, step=1)
    max_dte = c4.number_input("Max DTE", min_value=15, max_value=90, value=45, step=1)
    d1, d2, d3 = st.columns(3)
    target_delta = d1.number_input("Target abs delta", min_value=0.10, max_value=0.20, value=0.15, step=0.01, format="%.2f")
    min_delta = d2.number_input("Min abs delta", min_value=0.05, max_value=0.19, value=0.10, step=0.01, format="%.2f")
    max_delta = d3.number_input("Max abs delta", min_value=0.11, max_value=0.50, value=0.20, step=0.01, format="%.2f")
    if min_dte > target_dte or target_dte > max_dte:
        st.error("Target DTE must be between Min DTE and Max DTE.")
        return
    if min_delta > target_delta or target_delta > max_delta:
        st.error("Target delta must be between Min abs delta and Max abs delta.")
        return
    if st.button("Scan wheel cash-secured puts", type="primary"):
        with st.spinner("Finding wheel strategy candidates..."):
            wheel = build_wheel_strategy_candidates(
                selected,
                scan_size=scan_size,
                target_dte=int(target_dte),
                min_dte=int(min_dte),
                max_dte=int(max_dte),
                target_delta=float(target_delta),
                min_delta=float(min_delta),
                max_delta=float(max_delta),
            )
        st.session_state["last_wheel_scan"] = wheel
    wheel = st.session_state.get("last_wheel_scan", pd.DataFrame())
    if wheel.empty:
        st.info("Run the wheel scan to generate cash-secured put candidates.")
        return
    display_cols = [
        "Option Symbol", "Underlying", "Underlying Price", "Strike", "Expiration", "DTE",
        "Delta", "Delta Source", "Bid", "Ask", "Mid", "Moneyness %", "Underlying Signal",
        "AI bullish probability", "Suitability score", "Collateral Required", "Buying Power OK", "Wheel Score",
    ]
    display_cols = [col for col in display_cols if col in wheel.columns]
    st.dataframe(
        wheel[display_cols].style.format(
            {
                "Underlying Price": "${:.2f}",
                "Strike": "${:.2f}",
                "Delta": "{:.2f}",
                "Bid": "${:.2f}",
                "Ask": "${:.2f}",
                "Mid": "${:.2f}",
                "Moneyness %": "{:.2f}%",
                "AI bullish probability": "{:.1f}%",
                "Collateral Required": "${:,.2f}",
                "Wheel Score": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    selected_symbol = st.selectbox("Wheel candidate to sell put", wheel["Option Symbol"].astype(str).tolist())
    candidate = wheel[wheel["Option Symbol"].astype(str) == selected_symbol].iloc[0]
    suggested_price = candidate.get("Mid")
    if pd.isna(suggested_price) or float(suggested_price or 0) <= 0:
        suggested_price = candidate.get("Bid")
    suggested_price = float(suggested_price) if pd.notna(suggested_price) and float(suggested_price) > 0 else 0.05
    c1, c2, c3 = st.columns(3)
    qty = c1.number_input("Wheel contracts", min_value=0, value=1, step=1)
    limit_price = c2.number_input("Wheel sell-put limit", min_value=0.0, value=round(suggested_price, 2), step=0.05, format="%.2f")
    collateral = float(candidate["Strike"]) * 100 * qty
    c3.metric("Cash collateral required", f"${collateral:,.2f}")
    if collateral > available_options_bp:
        st.error(f"Not enough options buying power for this cash-secured put. Required ${collateral:,.2f}, available ${available_options_bp:,.2f}.")
    emergency_stop = st.toggle("Wheel emergency stop", value=False, key="wheel_emergency_stop")
    can_submit = qty > 0 and limit_price > 0 and collateral <= available_options_bp and not emergency_stop
    if st.button("Submit wheel cash-secured put paper order", disabled=not can_submit):
        result = place_option_paper_order(selected_symbol, int(qty), float(limit_price), side="sell")
        (st.success if result["ok"] else st.error)(result["message"])


def page_risk_management() -> None:
    st.header("Risk Management")
    st.write("Built-in paper-bot constraints:")
    st.table(
        pd.DataFrame(
            [
                {"Rule": "Maximum risk per trade", "Value": f"{RISK_RULES['max_risk_per_trade']:.0%} of capital"},
                {"Rule": "Maximum exposure per stock", "Value": f"{RISK_RULES['max_exposure_per_stock']:.0%}"},
                {"Rule": "Maximum total exposure", "Value": f"{RISK_RULES['max_total_exposure']:.0%}"},
                {"Rule": "Minimum reward:risk", "Value": f"1:{RISK_RULES['min_reward_to_risk']:.0f}"},
            ]
        )
    )
    st.subheader("ATR-Based Stop / Target Calculator")
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.selectbox("Load ticker ATR", scanner_universe([], 50), index=0)
    use_live_atr = c2.toggle("Use latest ATR %", value=True)
    reward_ratio = c3.number_input("Reward:risk ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.25)
    lock_ratio = c4.toggle("2.0 ratio lock", value=True)
    if lock_ratio:
        reward_ratio = 2.0
    analysis = analyze_stock(ticker)
    latest_atr_pct = None
    latest_price = 100.0
    if "Error" not in analysis:
        latest_price = float(analysis.get("Price", latest_price))
        df = analysis.get("Data", pd.DataFrame()).dropna()
        if not df.empty:
            latest_atr_pct = float(df.iloc[-1].get("ATR_Pct", 0))
    r1, r2, r3 = st.columns(3)
    entry = r1.number_input("Entry price", min_value=0.01, value=round(latest_price, 2), step=0.01)
    manual_atr_pct = r2.number_input("ATR %", min_value=0.1, max_value=50.0, value=round(latest_atr_pct or 4.3, 2), step=0.1)
    capital = r3.number_input("Capital", min_value=1000.0, value=25000.0, step=1000.0)
    atr_pct = latest_atr_pct if use_live_atr and latest_atr_pct is not None else manual_atr_pct
    risk_per_share = entry * (atr_pct / 100)
    stop = max(0.01, entry - risk_per_share)
    target = entry + (risk_per_share * reward_ratio)
    stop_loss_pct = ((entry - stop) / entry) * 100 if entry else 0
    target_gain_pct = (target / entry - 1) * 100 if entry else 0
    calc = pd.DataFrame(
        [
            {"Metric": "ATR % used", "Value": f"{atr_pct:.2f}%"},
            {"Metric": "Stop loss", "Value": f"${stop:.2f} ({stop_loss_pct:.2f}% risk)"},
            {"Metric": "Take profit", "Value": f"${target:.2f} ({target_gain_pct:.2f}% target)"},
            {"Metric": "Reward:risk", "Value": f"1:{reward_ratio:.2f}"},
            {"Metric": "Position size", "Value": f"{position_size(capital, entry, stop)} shares"},
        ]
    )
    st.dataframe(calc, use_container_width=True, hide_index=True)
    st.caption("The default keeps the target gain at roughly double the ATR-based risk, matching a 2.0 reward:risk profile.")


def page_portfolio_allocation(selected: list[str], allow_penny: bool) -> None:
    st.header("Portfolio Allocation")
    if not st.button("Generate allocation from selected stocks"):
        st.info("Allocation uses suitable stocks with Buy or Strong Buy signals and caps each stock at 20%.")
        return
    table = scan_stocks(selected, limit=min(len(selected), 100), allow_penny_stocks=allow_penny)
    candidates = table[table["Final signal"].isin(["Strong Buy", "Buy"])].copy()
    if candidates.empty:
        st.warning("No allocation candidates passed the filters.")
        return
    candidates["Raw Weight"] = candidates["AI bullish probability"] * candidates["Suitability score"] * candidates["Risk-reward ratio"].clip(lower=1)
    candidates["Weight"] = candidates["Raw Weight"] / candidates["Raw Weight"].sum()
    candidates["Weight"] = candidates["Weight"].clip(upper=RISK_RULES["max_exposure_per_stock"])
    candidates["Weight"] = candidates["Weight"] / candidates["Weight"].sum() * min(1, RISK_RULES["max_total_exposure"])
    st.dataframe(candidates[["Ticker", "Final signal", "Weight", "AI bullish probability", "Risk-reward ratio"]].style.format({"Weight": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.bar_chart(candidates.set_index("Ticker")["Weight"])


def page_watchlist_manager(selected: list[str]) -> None:
    st.header("Watchlist Manager")
    st.write("Default watchlist")
    st.code(", ".join(DEFAULT_WATCHLIST))
    custom_name = st.text_input("Custom watchlist name", value="My Universal Watchlist")
    manual = st.text_area("Tickers", value=", ".join(selected or DEFAULT_WATCHLIST), height=120)
    if st.button("Save watchlist", type="primary"):
        save_watchlist(custom_name, manual.split(","))
        st.success("Saved custom watchlist.")
    st.subheader("Stock Screener")
    limit = st.select_slider("Screener universe size", options=[5, 10, 20, 50, 100], value=50)
    universe_text = st.text_area("Universe", value=", ".join(COMMON_UNIVERSE), height=100)
    if st.button("Run suitability screener"):
        result = screen_universe(normalize_tickers(universe_text.split(",")), limit=limit)
        st.dataframe(format_scan_table(result), use_container_width=True, hide_index=True)


def _date_range_from_preset(records: pd.DataFrame, preset: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.now().normalize()
    min_date = records["Record date"].min().normalize() if records["Record date"].notna().any() else today
    if preset == "Last 1 month":
        start = today - pd.DateOffset(months=1)
    elif preset == "Last 3 months":
        start = today - pd.DateOffset(months=3)
    elif preset == "Last 6 months":
        start = today - pd.DateOffset(months=6)
    elif preset == "Last 1 year":
        start = today - pd.DateOffset(years=1)
    elif preset == "Last 2 years":
        start = today - pd.DateOffset(years=2)
    elif preset == "Last 5 years":
        start = today - pd.DateOffset(years=5)
    else:
        start = min_date
    return pd.Timestamp(start).normalize(), today


def page_signal_accuracy_analysis(selected: list[str], allow_penny: bool) -> None:
    st.header("Signal Accuracy Analysis")
    st.caption("This page records daily scanner signals and keeps them as a long-term reference journal.")
    st.info(f"Recorded Signal Journal is saved permanently at {DAILY_ANALYSIS_FILE}. Run the recorder once per trading day to build daily history for years.")
    with st.expander("Restore previous daily journal data", expanded=False):
        uploaded_journal = st.file_uploader("Import previous signal journal CSV or Excel", type=["csv", "xlsx"])
        if st.button("Import journal history"):
            if uploaded_journal is None:
                st.warning("Choose a CSV or Excel journal file first.")
            else:
                try:
                    imported_count = import_signal_journal_file(uploaded_journal)
                    st.success(f"Imported {imported_count} journal row(s).")
                except Exception as exc:
                    st.error(f"Could not import journal: {exc}")

    auto_daily = st.checkbox("Auto save today's Top 50 scanner + technical data", value=True)
    force_auto = st.button("Force rerun today's Top 50 autosave")
    if auto_daily or force_auto:
        with st.spinner("Checking daily Top 50 autosave..."):
            autosave = run_daily_top50_autosave(selected, allow_penny, force=force_auto)
        if autosave["status"] == "saved":
            st.success(autosave["message"])
            if autosave.get("workbook_path"):
                st.caption(f"Excel report saved: {autosave['workbook_path']}")
            if autosave.get("technical_errors"):
                st.warning("Some technical rows could not be saved: " + "; ".join(autosave["technical_errors"][:8]))
        elif autosave["status"] == "skipped":
            st.caption(autosave["message"])
        else:
            st.warning(autosave["message"])

    with st.expander("Daily Signal Journal Recorder", expanded=False):
        journal_tickers = st.multiselect(
            "Stocks to record today",
            selected or DEFAULT_WATCHLIST,
            default=selected or DEFAULT_WATCHLIST,
            help="Records one row per selected stock for today's Recorded Signal Journal.",
        )
        limit_options = [5, 10, 20, 50, 100]
        default_limit = next((option for option in limit_options if option >= min(50, len(journal_tickers) if journal_tickers else 5)), 50)
        journal_limit = st.select_slider("Maximum stocks to record", options=limit_options, value=default_limit)
        if st.button("Record today's selected stock signals", type="primary"):
            if not journal_tickers:
                st.warning("Select at least one stock to record.")
            else:
                universe = scanner_universe(journal_tickers, int(journal_limit))
                with st.spinner("Scanning and recording today's signals..."):
                    table = scan_stocks(universe, limit=int(journal_limit), allow_penny_stocks=allow_penny)
                if table.empty:
                    st.warning("No signals were available to record.")
                else:
                    saved = record_daily_rows(table, "Daily Selected Stock Signals")
                    st.success(f"Recorded {saved} signal(s) for today's journal.")
                    st.dataframe(format_scan_table(table), use_container_width=True, hide_index=True)

    records = load_signal_accuracy_records()
    if records.empty:
        st.info("No daily records yet. Use the Daily Signal Journal Recorder above, run the Multi-Stock AI Scanner with recording enabled, or record Daily technical snapshots.")
        return
    records["Record date"] = pd.to_datetime(records["Record date"], errors="coerce")
    c_filter1, c_filter2, c_filter3 = st.columns(3)
    range_preset = c_filter1.selectbox(
        "Journal range",
        ["Last 1 month", "Last 3 months", "Last 6 months", "Last 1 year", "Last 2 years", "Last 5 years", "All history", "Custom"],
        index=3,
    )
    min_date = records["Record date"].min().date() if records["Record date"].notna().any() else pd.Timestamp.now().date()
    max_date = records["Record date"].max().date() if records["Record date"].notna().any() else pd.Timestamp.now().date()
    if range_preset == "Custom":
        date_range = c_filter2.date_input("Custom record date range", value=(min_date, max_date))
    else:
        preset_start, preset_end = _date_range_from_preset(records, range_preset)
        date_range = (preset_start.date(), preset_end.date())
        c_filter2.metric("Start date", preset_start.date().isoformat())
        c_filter3.metric("End date", preset_end.date().isoformat())
    source_options = sorted(records["Source"].dropna().astype(str).unique().tolist()) if "Source" in records else []
    selected_sources = st.multiselect("Sources", source_options, default=source_options)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        records = records[(records["Record date"] >= start_date) & (records["Record date"] <= end_date)]
    if selected_sources:
        records = records[records["Source"].astype(str).isin(selected_sources)]
    if records.empty:
        st.info("No recorded signals match the selected date/source filters.")
        return

    full_records = load_signal_accuracy_records()
    full_records["Record date"] = pd.to_datetime(full_records["Record date"], errors="coerce")
    s1, s2, s3 = st.columns(3)
    s1.metric("All saved journal rows", f"{len(full_records):,}")
    s2.metric("Saved trading dates", full_records["Record date"].dt.date.astype(str).nunique())
    s3.metric("Saved years", full_records["Record date"].dt.year.dropna().astype(int).nunique())

    mode = st.selectbox(
        "Accuracy calculation mode",
        ["Future return after holding period", "Manual same-day Daily Change %"],
    )
    threshold = st.number_input("Correctness threshold (%)", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
    if mode == "Future return after holding period":
        horizon_options = {
            "1 day": 1,
            "3 days": 3,
            "7 days": 7,
            "2 weeks": 10,
            "1 month": 20,
        }
        horizon_label = st.selectbox("Evaluation holding period", list(horizon_options.keys()), index=0)
        horizon = horizon_options[horizon_label]
        with st.spinner("Evaluating recorded signals against later prices..."):
            evaluated = evaluate_signal_records(records, horizon_days=horizon, threshold_pct=threshold)
    else:
        horizon = 1
        evaluated = evaluate_manual_daily_change_records(records, hold_threshold_pct=threshold)
    evaluated_done = evaluated[evaluated["Evaluation status"] == "Evaluated"].copy()
    waiting = evaluated[evaluated["Evaluation status"] == "Waiting"].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded signals", len(evaluated))
    c2.metric("Evaluated", len(evaluated_done))
    c3.metric("Waiting", len(waiting))
    accuracy = evaluated_done["Correct"].mean() if not evaluated_done.empty else 0
    c4.metric("Correct final signals", f"{accuracy:.1%}")
    if not evaluated_done.empty:
        summary_period = st.selectbox("Accuracy summary period", ["Daily", "Monthly", "Yearly"], index=1)
        evaluated_done["Summary period"] = evaluated_done["Record date"].dt.to_period({"Daily": "D", "Monthly": "M", "Yearly": "Y"}[summary_period]).astype(str)
        by_period = (
            evaluated_done.groupby("Summary period")
            .agg(Signals=("Ticker", "count"), Correct=("Correct", "sum"), Accuracy=("Correct", "mean"), Avg_Return=("Realized return %", "mean"))
            .reset_index()
            .sort_values("Summary period", ascending=False)
        )
        st.subheader(f"Accuracy by {summary_period}")
        st.dataframe(by_period.style.format({"Accuracy": "{:.1%}", "Avg_Return": "{:.2f}%"}), use_container_width=True, hide_index=True)
        by_signal = (
            evaluated_done.groupby("Final signal")
            .agg(Signals=("Ticker", "count"), Correct=("Correct", "sum"), Accuracy=("Correct", "mean"), Avg_Return=("Realized return %", "mean"))
            .reset_index()
            .sort_values("Accuracy", ascending=False)
        )
        st.subheader("Accuracy by Final Signal")
        st.dataframe(by_signal.style.format({"Accuracy": "{:.1%}", "Avg_Return": "{:.2f}%"}), use_container_width=True, hide_index=True)
        comparison_rows = []
        strong_buy_rows = evaluated_done[evaluated_done["Final signal"].astype(str) == "Strong Buy"].copy()
        if not strong_buy_rows.empty:
            synergy_rows = add_signal_synergy_columns(strong_buy_rows)
            comparison_rows.append(
                {
                    "Strategy": "Standard Strong Buy",
                    "Signals": len(strong_buy_rows),
                    "Correct": int(strong_buy_rows["Correct"].sum()),
                    "Accuracy": strong_buy_rows["Correct"].mean(),
                    "Avg_Return": strong_buy_rows["Realized return %"].mean(),
                }
            )
            synergy_only = synergy_rows[synergy_rows["Synergy Strong Buy"]].copy()
            if not synergy_only.empty:
                comparison_rows.append(
                    {
                        "Strategy": "Synergy Strong Buy",
                        "Signals": len(synergy_only),
                        "Correct": int(synergy_only["Correct"].sum()),
                        "Accuracy": synergy_only["Correct"].mean(),
                        "Avg_Return": synergy_only["Realized return %"].mean(),
                    }
                )
        hold_rows = evaluated_done[evaluated_done["Final signal"].astype(str) == "Hold"].copy()
        if not hold_rows.empty:
            trend_text = hold_rows.get("Trend", pd.Series(index=hold_rows.index, dtype=str)).astype(str)
            vwap_text = hold_rows.get("VWAP status", pd.Series(index=hold_rows.index, dtype=str)).astype(str)
            support_holds = hold_rows[
                vwap_text.str.contains("Above", case=False, na=False)
                & trend_text.str.contains("Downtrend|Sideways", case=False, na=False, regex=True)
            ].copy()
            if not support_holds.empty:
                comparison_rows.append(
                    {
                        "Strategy": "Hold support validation",
                        "Signals": len(support_holds),
                        "Correct": int(support_holds["Correct"].sum()),
                        "Accuracy": support_holds["Correct"].mean(),
                        "Avg_Return": support_holds["Realized return %"].mean(),
                    }
                )
        if comparison_rows:
            strategy_comparison = pd.DataFrame(comparison_rows)
            st.subheader("Strategy Comparison")
            st.caption("Compares normal Strong Buy rows against Synergy Strong Buy rows where RSI, MACD, and VWAP are aligned. Hold support validation checks Hold rows that are Above VWAP but still Downtrend/Sideways.")
            st.dataframe(strategy_comparison.style.format({"Accuracy": "{:.1%}", "Avg_Return": "{:.2f}%"}), use_container_width=True, hide_index=True)
        by_source = (
            evaluated_done.groupby("Source")
            .agg(Signals=("Ticker", "count"), Accuracy=("Correct", "mean"), Avg_Return=("Realized return %", "mean"))
            .reset_index()
        )
        st.subheader("Accuracy by Source")
        st.dataframe(by_source.style.format({"Accuracy": "{:.1%}", "Avg_Return": "{:.2f}%"}), use_container_width=True, hide_index=True)
    st.subheader("Recorded Signal Journal")
    show_cols = [
        "Record date", "Source", "Ticker", "Entry price", "Final signal", "Signal direction",
        "Evaluation status", "Wait reason", "Trading bars available", "Exit price", "Realized return %", "Correct", "RSI",
        "Manual rule", "MACD signal", "VWAP status", "Trend", "AI bullish probability", "AI bearish probability",
        "Suitability score",
    ]
    show_cols = [col for col in show_cols if col in evaluated.columns]
    st.dataframe(evaluated[show_cols], use_container_width=True, hide_index=True)
    evaluated_workbook = export_signal_workbook(evaluated, pd.Timestamp.now().date().isoformat(), f"signal_accuracy_{horizon}d")
    if evaluated_workbook:
        st.caption(f"Daily Signal Accuracy Excel saved: {evaluated_workbook}")
    st.download_button("Download filtered signal journal", evaluated.to_csv(index=False), "daily_signal_analysis_filtered.csv", "text/csv")
    st.download_button("Download full signal journal", full_records.to_csv(index=False), "daily_signal_analysis_full.csv", "text/csv")


def main() -> None:
    inject_style()
    st.title("Universal AI Stock Analysis Dashboard")
    selected = ticker_selector()
    allow_penny = st.sidebar.toggle("Manually allow penny stocks", value=False)
    page = st.sidebar.radio(
        "Dashboard pages",
        [
            "Market Overview",
            "Multi-Stock AI Scanner",
            "Short Strategy Stock Scanner",
            "Options AI Scanner",
            "Individual Stock Analysis",
            "Technical Indicator Dashboard",
            "AI Prediction Model",
            "Backtesting",
            "Signal Accuracy Analysis",
            "Paper Trading Bot",
            "Options Paper Trading",
            "Risk Management",
            "Portfolio Allocation",
            "Watchlist Manager",
        ],
    )

    if not selected:
        selected = scanner_universe([], 50)
    st.session_state["selected_stocks_for_options"] = selected

    if page == "Market Overview":
        page_market_overview(selected)
    elif page == "Multi-Stock AI Scanner":
        page_multi_stock_scanner(selected, allow_penny)
    elif page == "Short Strategy Stock Scanner":
        page_short_strategy_scanner(selected)
    elif page == "Options AI Scanner":
        page_options_ai_scanner(selected, allow_penny)
    elif page == "Individual Stock Analysis":
        page_individual_analysis(selected, allow_penny)
    elif page == "Technical Indicator Dashboard":
        page_technical_dashboard(selected)
    elif page == "AI Prediction Model":
        page_ai_prediction(selected)
    elif page == "Backtesting":
        page_backtesting(selected)
    elif page == "Signal Accuracy Analysis":
        page_signal_accuracy_analysis(selected, allow_penny)
    elif page == "Paper Trading Bot":
        page_paper_trading(selected, allow_penny)
    elif page == "Options Paper Trading":
        page_options_paper_trading()
    elif page == "Risk Management":
        page_risk_management()
    elif page == "Portfolio Allocation":
        page_portfolio_allocation(selected, allow_penny)
    elif page == "Watchlist Manager":
        page_watchlist_manager(selected)


if __name__ == "__main__":
    main()
