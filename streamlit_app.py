from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
        from app.core.data import fetch_last_prices, market_data_healthcheck, normalize_tickers
        from app.core.modeling import train_models
        from app.core.paper_trading import cancel_order, log_signal, open_orders_table, place_paper_order, position_size
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
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
        from alpaca.common.exceptions import APIError
    except Exception:
        TradingClient = None
        OrderSide = None
        QueryOrderStatus = None
        TimeInForce = None
        GetOrdersRequest = None
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

    def alpaca_client():
        if TradingClient is None:
            return None
        key = _secret("ALPACA_API_KEY")
        secret = _secret("ALPACA_SECRET_KEY")
        endpoint = _secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2").removesuffix("/").removesuffix("/v2")
        return TradingClient(key, secret, paper=True, url_override=endpoint) if key and secret else None

    def open_orders_table(ticker: str | None = None) -> pd.DataFrame:
        client = alpaca_client()
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

    def log_signal(row: dict) -> None:
        try:
            SIGNAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{**{"timestamp": datetime.utcnow().isoformat()}, **row}]).to_csv(SIGNAL_LOG_FILE, mode="a", header=not SIGNAL_LOG_FILE.exists(), index=False)
        except Exception:
            pass

load_dotenv()


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
    return df.style.format({k: v for k, v in numeric_cols.items() if k in df.columns})


def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", line=dict(width=1.2, dash="dot")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False)
    return fig


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
    run = st.button("Run scanner", type="primary")
    if not run:
        st.info("Choose a scan size and run the scanner.")
        return pd.DataFrame()
    with st.spinner("Scanning selected stocks..."):
        table = scan_stocks(universe, limit=scan_size, allow_penny_stocks=allow_penny)
    if table.empty:
        st.warning("No scan results were available.")
        return table
    st.dataframe(format_scan_table(table), use_container_width=True, hide_index=True)
    st.download_button("Download scanner results", table.to_csv(index=False), "scanner_results.csv", "text/csv")
    return table


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
    st.plotly_chart(price_chart(result["Data"].tail(252), ticker), use_container_width=True)


def page_technical_dashboard(selected: list[str]) -> None:
    st.header("Technical Indicator Dashboard")
    ticker = st.selectbox("Technical ticker", selected or DEFAULT_WATCHLIST)
    df = prepare_stock(ticker).dropna()
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
    entry = st.number_input("Entry price", value=100.0)
    stop = st.number_input("Stop loss", value=95.0)
    capital = st.number_input("Capital", value=25000.0)
    st.metric("Position size", f"{position_size(capital, entry, stop)} shares")


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
            "Individual Stock Analysis",
            "Technical Indicator Dashboard",
            "AI Prediction Model",
            "Backtesting",
            "Paper Trading Bot",
            "Risk Management",
            "Portfolio Allocation",
            "Watchlist Manager",
        ],
    )

    if not selected:
        selected = scanner_universe([], 50)

    if page == "Market Overview":
        page_market_overview(selected)
    elif page == "Multi-Stock AI Scanner":
        page_multi_stock_scanner(selected, allow_penny)
    elif page == "Individual Stock Analysis":
        page_individual_analysis(selected, allow_penny)
    elif page == "Technical Indicator Dashboard":
        page_technical_dashboard(selected)
    elif page == "AI Prediction Model":
        page_ai_prediction(selected)
    elif page == "Backtesting":
        page_backtesting(selected)
    elif page == "Paper Trading Bot":
        page_paper_trading(selected, allow_penny)
    elif page == "Risk Management":
        page_risk_management()
    elif page == "Portfolio Allocation":
        page_portfolio_allocation(selected, allow_penny)
    elif page == "Watchlist Manager":
        page_watchlist_manager(selected)


if __name__ == "__main__":
    main()
